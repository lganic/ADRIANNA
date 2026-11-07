#pragma once
// Executable network built from a NetworkGraph, with full introspection:
//
//  * named parameters (read / write / freeze / hook their gradients)
//  * forward hooks (observe or replace any node's activation)
//  * forward pre-hooks (observe or replace a node's merged input)
//  * backward hooks (observe or replace the gradient w.r.t. any activation)
//  * activation recording (activations and, after backward(), their gradients)
//  * recurrent state access (get / set / mask / detach)
//  * weight (de)serialisation and graph export
//
// One forward() call processes one time step for a batch. Recurrent layers keep
// their state between calls, so BPTT over an episode is just "forward T times,
// then backward()".

#include <map>
#include <memory>
#include <string>
#include <vector>

#include "codeepneat/graph.hpp"
#include "codeepneat/layers.hpp"

namespace cdn {

/// Read-only description of one node, passed to every hook and filter.
struct NodeInfo {
  int index = -1;      // position in execution order
  int graph_id = -1;   // id in the (pruned) graph
  std::string name;
  NodeKind kind = NodeKind::Layer;
  std::string type;    // layer type ("input", "dense", "conv2d", "lstm", "identity", ...)
  std::string merge;
  std::vector<std::string> input_names;
  std::vector<Shape> input_shapes;
  Shape merged_shape;  // shape fed to the layer (after merge)
  Shape output_shape;  // per sample
  HyperparamValues hp;
  NodeOrigin origin;
  size_t num_parameters = 0;  ///< layer + merge parameters of this node
};

/// Called after a node computed its output. May modify `output` in place (edit
/// values; bypasses autograd) or assign a new tensor (e.g. `out = out * mask`,
/// differentiable).
using ForwardHook = std::function<void(const NodeInfo& node, Tensor& output)>;
/// Called during backward() with the gradient of the loss w.r.t. a node's output.
using BackwardHook = std::function<void(const NodeInfo& node, Tensor& grad_output)>;
/// Selects nodes for a hook or for set_trainable. A null filter selects every node.
using NodeFilter = std::function<bool(const NodeInfo& node)>;

/// A built, trainable network.
///
/// Construction prunes nodes that are not on an input-to-output path, validates
/// the graph, infers all shapes and initialises the weights from `seed`. Invalid
/// graphs throw std::invalid_argument.
///
/// A Network starts in training mode. It is not thread-safe, and it is not
/// copyable: to duplicate one, build a second network from graph() and copy the
/// weights with state_dict() / load_state_dict(). Separate networks can run on
/// separate threads.
///
/// Do not add or remove hooks from inside a hook while forward() is running.
class Network {
 public:
  Network(const NetworkGraph& graph, const BuildOptions& options = BuildOptions(), uint64_t seed = 1);
  Network(const Network&) = delete;
  Network& operator=(const Network&) = delete;

  // ---- forward ----
  /// Inputs in graph input order, each [B, ...shape] (or unbatched [...shape]).
  /// Returns outputs in graph output order, each [B, size].
  ///
  /// Spell the vector out at the call site: `net.forward(std::vector<Tensor>{x})`.
  /// `net.forward({x})` with a single tensor selects the Tensor overload below,
  /// which returns only the first output; `net.forward({a, b})` does not compile.
  ///
  /// If the batch size differs from the current recurrent state's, the state is
  /// reset to zeros first.
  std::vector<Tensor> forward(const std::vector<Tensor>& inputs);
  /// Inputs and outputs by name. Prefixes are optional for inputs ("obs" or
  /// "input:obs"); output keys have the "head:" prefix removed.
  std::map<std::string, Tensor> forward(const std::map<std::string, Tensor>& inputs);
  /// Single input -> first output.
  Tensor forward(const Tensor& x);

  const std::vector<std::string>& input_names() const { return input_names_; }
  const std::vector<std::string>& output_names() const { return output_names_; }
  const std::vector<Shape>& input_shapes() const { return input_shapes_; }
  int output_index(const std::string& name) const;  // -1 if absent
  int input_index(const std::string& name) const;

  // ---- mode ----
  /// Training mode enables dropout. It does not affect gradient recording; use
  /// NoGradGuard for that.
  void train(bool on = true) { training_ = on; }
  void eval() { training_ = false; }
  bool is_training() const { return training_; }
  /// RNG used for dropout masks.
  Rng& rng() { return rng_; }

  // ---- recurrent state ----
  bool is_recurrent() const { return !recurrent_.empty(); }
  /// Zero all recurrent state for `batch` sequences.
  void reset_state(int batch = 1);
  /// Cut the state from the autograd graph (truncated BPTT). Without this, or a
  /// reset, the state keeps every earlier step's graph alive.
  void detach_state();
  /// Multiply every recurrent state row b by keep[b] (0 resets that sequence).
  void mask_state(const std::vector<float>& keep);
  /// State tensors named "<node>.h" and "<node>.c". These are handles, and they
  /// stay valid as a snapshot because each step replaces the tensors it updates.
  std::vector<NamedTensor> get_state() const;
  /// Restore a snapshot from get_state().
  void set_state(const std::vector<NamedTensor>& state);

  // ---- parameters ----
  /// All parameters in execution order, including merge projections and
  /// memory-cell skip parameters. Handles share storage with the network.
  std::vector<NamedTensor> named_parameters() const;
  std::vector<Tensor> parameters() const;
  /// Shares storage with the network: editing .data() edits the weights.
  Tensor parameter(const std::string& name) const;
  bool has_parameter(const std::string& name) const;
  size_t num_parameters() const;
  void zero_grad();
  /// Freeze/unfreeze all parameters whose name starts with `prefix`
  /// ("" = all). Returns the number of tensors affected.
  size_t set_trainable(const std::string& prefix, bool trainable);
  size_t set_trainable(const NodeFilter& filter, bool trainable);

  // ---- nodes ----
  /// All nodes in execution (topological) order.
  const std::vector<NodeInfo>& nodes() const { return info_; }
  /// Throws std::invalid_argument for unknown names.
  const NodeInfo& node(const std::string& name) const;
  bool has_node(const std::string& name) const;
  /// Null for input nodes / unknown names.
  Layer* layer(const std::string& name);
  const Layer* layer(const std::string& name) const;
  /// Merge op of a node with several inputs; null otherwise.
  const Merge* merge(const std::string& name) const;
  std::vector<Layer*> layers();
  /// The pruned graph this network was built from.
  const NetworkGraph& graph() const { return graph_; }

  // ---- hooks ----
  // Every add_* returns an id for remove_hook(). Hooks on one node run in the order
  // they were added. Identity nodes share their tensor with their parent, so an
  // in-place edit there also changes the parent's activation, and a backward hook
  // there sees the gradient of that shared tensor.

  /// Runs after a node computed its output, for every node the filter selects.
  int add_forward_hook(ForwardHook hook, NodeFilter filter = nullptr);
  int add_forward_hook(const std::string& node_name, ForwardHook hook);
  /// Runs on a node's merged input, before its layer. Not called for input nodes.
  int add_forward_pre_hook(ForwardHook hook, NodeFilter filter = nullptr);
  /// Runs during backward() with the gradient w.r.t. a node's output. Only tensors
  /// that require grad are hooked, so this never fires under NoGradGuard, and not
  /// for input nodes unless the input tensor itself requires grad.
  int add_backward_hook(BackwardHook hook, NodeFilter filter = nullptr);
  int add_backward_hook(const std::string& node_name, BackwardHook hook);
  /// Runs on a parameter's gradient, once per backward pass, before accumulation.
  int add_parameter_hook(const std::string& param_name, GradHook hook);
  /// Removes a hook of any kind. Backward hooks already attached to tensors of an
  /// earlier forward() stay on those tensors.
  void remove_hook(int id);
  void clear_hooks();

  // ---- recording ----
  /// When enabled, every node's output of the latest forward() is retained;
  /// after backward() their .grad() holds dLoss/dActivation.
  void record_activations(bool on) { record_ = on; }
  bool recording() const { return record_; }
  const std::map<std::string, Tensor>& activations() const { return activations_; }
  /// Recorded output of one node. Throws if nothing was recorded for it.
  Tensor activation(const std::string& node_name) const;

  // ---- I/O ----
  std::vector<NamedTensor> state_dict() const;  // detached copies
  /// Copies values for matching names. strict: every network parameter must be
  /// present with the same size.
  void load_state_dict(const std::vector<NamedTensor>& state, bool strict = true);
  /// Text weight file; see docs/file-formats.md. Architecture is not included.
  void save_weights(const std::string& path) const;
  void load_weights(const std::string& path, bool strict = true);
  /// Table of nodes with types, shapes, parameter counts and hyperparameters.
  std::string summary() const;
  /// Graphviz source of the graph. Red dashed edges are memory-cell skips.
  std::string to_dot() const { return graph_.to_dot(); }

 private:
  struct Runtime {
    std::shared_ptr<NodeInfo> info;
    std::unique_ptr<Merge> merge;
    std::unique_ptr<Layer> layer;
    int input_slot = -1;
    std::vector<int> inputs;  // runtime indices
    std::vector<int> fwd, pre, bwd;  // hook indices
  };
  struct HookEntry {
    int id;
    int kind;  // 0 fwd, 1 pre, 2 bwd
    NodeFilter filter;
    ForwardHook fhook;
    BackwardHook bhook;
  };
  void rebuild_hook_cache();
  int add_hook(int kind, NodeFilter filter, ForwardHook f, BackwardHook b);

  NetworkGraph graph_;
  BuildOptions options_;
  Rng rng_;
  bool training_ = true;
  bool record_ = false;
  std::vector<Runtime> rt_;
  std::vector<NodeInfo> info_;
  std::map<std::string, int> by_name_;
  std::vector<std::string> input_names_, output_names_;
  std::vector<Shape> input_shapes_;
  std::vector<int> output_rt_;
  std::vector<Layer*> recurrent_;
  int state_batch_ = -1;
  std::vector<HookEntry> hooks_;
  std::map<int, std::pair<Tensor, int>> param_hooks_;
  int next_hook_id_ = 1;
  std::map<std::string, Tensor> activations_;
};

}  // namespace cdn
