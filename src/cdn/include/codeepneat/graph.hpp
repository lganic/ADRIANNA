#pragma once
// NetworkGraph: a plain, inspectable description of a network (a DAG of layer
// nodes + LSTM memory-cell skip edges). It is the intermediate representation
// between genomes and executable Networks:
//
//   Genome(s) --assemble--> NetworkGraph --Network(graph)--> executable network
//
// Graphs can also be written by hand (e.g. for baselines or unit tests).

#include <string>
#include <vector>

#include "codeepneat/hyperparams.hpp"
#include "codeepneat/tensor.hpp"

namespace cdn {

enum class NodeKind { Input, Layer, Output };
const char* to_string(NodeKind k);

/// Where a node came from in the genotype (for filtering hooks / inspection).
struct NodeOrigin {
  std::string role;        // input | module_in | module_out | layer | body | head
  int blueprint_node = -1; // blueprint node id (CoDeepNEAT) or -1
  int species = -1;        // module species the blueprint node points to
  int module_id = -1;      // genome id of the module chosen for this network
  int module_node = -1;    // node id inside the module (DeepNEAT: genome node id)
};

struct GraphNode {
  int id = -1;
  std::string name;
  NodeKind kind = NodeKind::Layer;
  std::string layer_type = "identity";  // Layer nodes
  HyperparamValues hp;                  // Layer nodes
  std::string merge = "concat";         // how multiple inputs are merged
  std::vector<int> inputs;
  Shape shape;                          // Input nodes: per-sample shape (no batch axis)
  int out_size = 0;                     // Output nodes
  std::string activation = "linear";    // Output nodes
  float init_scale = 1.0f;              // Output nodes
  NodeOrigin origin;
};

/// Gated connection from one LSTM node's memory cell into another's (Fig. 3).
/// If `from` executes before `to` in a time step the current cell value is
/// used, otherwise (a feedback connection) the previous step's value.
struct CellSkipEdge {
  int from = -1;
  int to = -1;
};

/// A network description. Node names must be unique. A node's id is its index in
/// `nodes`. The add_* methods return that id, which is what later nodes list in
/// their inputs. Inputs and outputs are ordered by id.
class NetworkGraph {
 public:
  std::vector<GraphNode> nodes;
  std::vector<CellSkipEdge> cell_skips;

  /// Network input. `shape` is per sample: [F] or [C,H,W].
  int add_input(const std::string& name, const Shape& shape);
  /// Layer node. `type` must be registered in LayerRegistry; `hp` configures it.
  /// `merge` ("concat" or "sum") applies when there are several inputs.
  int add_layer(const std::string& name, const std::string& type, const HyperparamValues& hp,
                const std::vector<int>& inputs, const std::string& merge = "concat");
  /// Pass-through node, useful as a pure merge point.
  int add_identity(const std::string& name, const std::vector<int>& inputs, const std::string& merge = "concat");
  /// Output head: a dense layer of `size` units. `init_scale` multiplies its
  /// initial weights (small values give a near-uniform initial policy).
  int add_output(const std::string& name, int input, int size, const std::string& activation = "linear",
                 float init_scale = 1.0f);
  /// Memory-cell skip between two LSTM nodes. Ignored by Network if either end is
  /// not an LSTM layer.
  void add_cell_skip(int from, int to);

  const GraphNode& node(int id) const { return nodes.at(id); }
  GraphNode& node(int id) { return nodes.at(id); }
  int find(const std::string& name) const;  // -1 if absent
  std::vector<int> input_ids() const;
  std::vector<int> output_ids() const;

  /// Kahn topological order (ties broken by id). Throws on cycles.
  std::vector<int> topological_order() const;
  /// Copy without nodes that are not on any input->output path (inputs are kept).
  NetworkGraph pruned() const;
  /// Throws std::invalid_argument describing the first problem found.
  void validate() const;
  std::string to_dot() const;
};

}  // namespace cdn
