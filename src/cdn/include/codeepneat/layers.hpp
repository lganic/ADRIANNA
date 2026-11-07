#pragma once
// Layers (the phenotype of DeepNEAT/CoDeepNEAT node genes) and the merge
// operation applied when a node has several incoming connections.
//
// All parameters are ordinary leaf Tensors registered under a fully qualified
// name ("<node name>.<local name>"), so they can be read, written, frozen or
// hooked from the outside.

#include <functional>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "codeepneat/hyperparams.hpp"
#include "codeepneat/tensor.hpp"

namespace cdn {

/// A tensor with its fully qualified name, e.g. "bp3/m17/n5.weight". The tensor is
/// a handle: for parameters it shares storage with the layer that owns it.
struct NamedTensor {
  std::string name;
  Tensor tensor;
};

/// Options that apply to the whole network build (typically derived from global
/// hyperparameters).
struct BuildOptions {
  /// glorot_uniform | glorot_normal | he_normal | he_uniform | lecun_normal | uniform
  std::string weight_init = "glorot_uniform";
  float uniform_range = 0.05f;   // for weight_init == "uniform" (paper's LSTM setup)
  float lstm_forget_bias = 1.0f;
};

/// Per-call state handed to Layer::forward by the network.
struct LayerContext {
  bool training = true;  ///< false after Network::eval(); disables dropout
  Rng* rng = nullptr;    ///< the network's RNG; never null during Network::forward
};

/// Initialise a weight tensor (requires_grad = true) with the scheme named in
/// `opt.weight_init`. `scale` multiplies the scheme's limit or standard deviation;
/// layers pass their "weight_scale" hyperparameter here.
Tensor init_weight(const Shape& shape, int fan_in, int fan_out, const BuildOptions& opt, float scale, Rng& rng);

/// Base class of everything a graph node can execute.
///
/// To add a layer type:
///  1. Derive from Layer; forward the name, a type string, the input shape and
///     the hyperparameters to the protected constructor.
///  2. In your constructor set `out_shape_` and create parameters with add_param().
///  3. Override forward(). `x` has a leading batch axis; shapes elsewhere in this
///     class are per sample. Build the result from differentiable ops, or from an
///     op of your own, so gradients flow.
///  4. Register a factory with LayerRegistry and list the type in a search space.
/// docs/extending.md has a complete example.
class Layer {
 public:
  virtual ~Layer() = default;

  const std::string& name() const { return name_; }
  const std::string& type() const { return type_; }
  const Shape& input_shape() const { return in_shape_; }    // per sample
  const Shape& output_shape() const { return out_shape_; }  // per sample
  const HyperparamValues& hparams() const { return hp_; }

  /// Compute the output for one time step. `x` is [B, ...input_shape].
  virtual Tensor forward(const Tensor& x, LayerContext& ctx) = 0;

  // Recurrent state handling (no-ops for feed-forward layers).
  virtual bool is_recurrent() const { return false; }
  /// Zero the state for `batch` sequences.
  virtual void reset_state(int /*batch*/) {}
  /// Cut the state from the autograd graph (truncated BPTT).
  virtual void detach_state() {}
  /// Multiply state row b by keep[b]; 0 forgets that sequence.
  virtual void mask_state(const Tensor& /*keep*/) {}  // keep: [B,1]
  /// Current state tensors, named "<layer>.<part>".
  virtual std::vector<NamedTensor> state() const { return {}; }
  /// Restore state, in the same order as state().
  virtual void set_state(const std::vector<Tensor>& /*state*/) {}

  /// Registered parameters, in creation order, under their full names.
  std::vector<NamedTensor>& params() { return params_; }
  const std::vector<NamedTensor>& params() const { return params_; }
  /// Look up a parameter by its local name ("weight", "bias", ...). Null if absent.
  Tensor* param(const std::string& local_name);
  /// Total number of scalar parameters.
  size_t num_parameters() const;

 protected:
  Layer(std::string name, std::string type, Shape in_shape, HyperparamValues hp);
  /// Registers `t` as "<layer name>.<local_name>", enables its gradient and
  /// returns a handle to it, which the subclass usually stores.
  Tensor add_param(const std::string& local_name, Tensor t);

  std::string name_, type_;
  Shape in_shape_, out_shape_;
  HyperparamValues hp_;
  std::vector<NamedTensor> params_;
};

/// Pass-through used for module boundaries and the network "body". forward()
/// returns its input tensor itself, not a copy, so an identity node shares its
/// activation with the node before it (unless it has a merge).
class IdentityLayer : public Layer {
 public:
  IdentityLayer(const std::string& name, const Shape& in_shape);
  Tensor forward(const Tensor& x, LayerContext& ctx) override;
};

/// hp: units, activation, dropout, weight_scale, use_bias
class DenseLayer : public Layer {
 public:
  DenseLayer(const std::string& name, const HyperparamValues& hp, const Shape& in_shape, const BuildOptions& opt,
             Rng& rng);
  Tensor forward(const Tensor& x, LayerContext& ctx) override;
  Tensor weight() const { return weight_; }
  Tensor bias() const { return bias_; }  // undefined if use_bias == false
  const std::string& activation_name() const { return act_; }
  float dropout_rate() const { return dropout_; }

 private:
  Tensor weight_, bias_;
  std::string act_;
  float dropout_ = 0.0f;
};

/// hp: filters, kernel_size, stride, max_pool, activation, dropout, weight_scale
/// Flat inputs [F] are treated as [F,1,1] images.
class Conv2DLayer : public Layer {
 public:
  Conv2DLayer(const std::string& name, const HyperparamValues& hp, const Shape& in_shape, const BuildOptions& opt,
              Rng& rng);
  Tensor forward(const Tensor& x, LayerContext& ctx) override;

  Tensor weight() const { return weight_; }  // [filters, C, k, k]
  Tensor bias() const { return bias_; }      // undefined if use_bias == false
  int stride() const { return stride_; }
  int padding() const { return pad_; }
  bool max_pool() const { return pool_; }
  const std::string& activation_name() const { return act_; }
  float dropout_rate() const { return dropout_; }
  /// Input viewed as an image ([F] inputs become [F,1,1]).
  const Shape& image_shape() const { return img_shape_; }

 private:
  Tensor weight_, bias_;
  int stride_ = 1, pad_ = 0;
  bool pool_ = false;
  std::string act_;
  float dropout_ = 0.0f;
  Shape img_shape_;
};

/// Long short-term memory layer that is stepped once per forward() call. Its
/// hidden/cell state persists between calls (BPTT happens naturally through the
/// autograd graph; call detach_state() to truncate).
///
/// Supports gated memory-cell skip connections from other LSTM layers
/// (Sec. 4.1 / Fig. 3 of the paper):
///   c_t = f * c_{t-1} + i * g + sum_k  sigmoid(W_k [x_t, h_{t-1}] + b_k) * P_k c^{src_k}
/// where c^{src_k} is the source layer's current cell (if the source already ran
/// in this step) or its previous cell (feedback connection), and P_k is a learned
/// projection when the sizes differ.
class LSTMLayer : public Layer {
 public:
  LSTMLayer(const std::string& name, const HyperparamValues& hp, const Shape& in_shape, const BuildOptions& opt,
            Rng& rng);
  Tensor forward(const Tensor& x, LayerContext& ctx) override;

  bool is_recurrent() const override { return true; }
  void reset_state(int batch) override;
  void detach_state() override;
  void mask_state(const Tensor& keep) override;
  std::vector<NamedTensor> state() const override;
  void set_state(const std::vector<Tensor>& s) override;

  void add_cell_skip(LSTMLayer* source, const BuildOptions& opt, Rng& rng);
  size_t num_cell_skips() const { return skips_.size(); }
  int units() const { return units_; }
  const Tensor& hidden() const { return h_; }
  const Tensor& cell() const { return c_; }

  /// Gate activations of the last step (i, f, g, o), for inspection.
  const std::map<std::string, Tensor>& last_gates() const { return gates_; }

  int input_features() const { return in_features_; }
  float dropout_rate() const { return dropout_; }
  Tensor weight() const { return weight_; }  // [4H, I+H], gate order i, f, g, o
  Tensor bias() const { return bias_; }      // [4H]
  struct CellSkipInfo {
    const LSTMLayer* source;
    Tensor gate_weight, gate_bias;  // [H, I+H], [H]
    Tensor projection;              // [H, H_src], undefined if sizes match
  };
  std::vector<CellSkipInfo> cell_skips() const;

 private:
  struct Skip {
    LSTMLayer* src;
    Tensor gate_w, gate_b;
    Tensor proj;  // undefined if sizes match
  };
  int units_ = 0, in_features_ = 0;
  Tensor weight_, bias_;
  float dropout_ = 0.0f;
  Tensor h_, c_;
  std::vector<Skip> skips_;
  std::map<std::string, Tensor> gates_;
};

/// Merges the outputs of several parent layers (Sec. 3.1). Inputs of
/// mismatched size are first brought to the smallest size: spatial maps are max
/// pooled to the smallest H x W; for "sum" merges, mismatched channel/feature
/// counts are projected by a learned 1x1 convolution / dense bottleneck. Mixed
/// spatial and flat inputs are flattened.
class Merge {
 public:
  Merge(const std::string& node_name, const std::string& method, const std::vector<Shape>& in_shapes,
        const BuildOptions& opt, Rng& rng);
  Tensor forward(const std::vector<Tensor>& xs) const;
  const Shape& output_shape() const { return out_; }
  const std::string& method() const { return method_; }
  std::vector<NamedTensor>& params() { return params_; }
  const std::vector<NamedTensor>& params() const { return params_; }

  const std::vector<Shape>& input_shapes() const { return in_; }
  /// Mixed spatial/flat inputs: everything is flattened.
  bool flattens() const { return flatten_; }
  /// All inputs spatial: max pooled to pool_size() then merged on channels.
  bool spatial() const { return spatial_; }
  std::pair<int, int> pool_size() const { return {ph_, pw_}; }
  /// Per-input projection (1x1 conv weight [C_out, C_in, 1, 1] if spatial, else
  /// dense weight [F_out, F_in]); undefined where none is needed.
  const std::vector<Tensor>& projections() const { return proj_; }

 private:
  std::string method_;
  std::vector<Shape> in_;
  Shape out_;
  bool flatten_ = false, spatial_ = false;
  int ph_ = 0, pw_ = 0;
  std::vector<Tensor> proj_;  // undefined where no projection is needed
  std::vector<NamedTensor> params_;
};

using LayerFactory = std::function<std::unique_ptr<Layer>(
    const std::string& name, const HyperparamValues& hp, const Shape& in_shape, const BuildOptions& opt, Rng& rng)>;

/// Registry of layer types; register your own to extend the search space
/// (the node hyperparameter "type" selects the factory).
class LayerRegistry {
 public:
  static LayerRegistry& instance();
  void add(const std::string& type, LayerFactory factory);
  bool has(const std::string& type) const;
  std::unique_ptr<Layer> create(const std::string& type, const std::string& name, const HyperparamValues& hp,
                                const Shape& in_shape, const BuildOptions& opt, Rng& rng) const;
  std::vector<std::string> types() const;

 private:
  LayerRegistry();
  std::map<std::string, LayerFactory> factories_;
};

}  // namespace cdn
