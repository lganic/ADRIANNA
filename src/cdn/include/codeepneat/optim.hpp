#pragma once
// Gradient-descent optimizers operating on named parameter tensors. Parameters
// with requires_grad == false (frozen) or without a gradient are skipped.
// Optimizer buffers (momentum / Adam moments) are exposed for inspection.

#include <memory>
#include <string>
#include <vector>

#include "codeepneat/hyperparams.hpp"
#include "codeepneat/layers.hpp"

namespace cdn {

/// Per-parameter optimizer buffers. Empty until the parameter's first update.
struct ParamState {
  std::vector<float> m;  // momentum / first moment
  std::vector<float> v;  // second moment / square average
  long steps = 0;        ///< updates applied to this parameter
};

/// Base class. The usual loop is: zero_grad(), backward(), optional clipping, step().
class Optimizer {
 public:
  /// Holds handles to the tensors, so it updates the network's weights directly.
  Optimizer(std::vector<NamedTensor> params, float lr);
  virtual ~Optimizer() = default;
  /// Apply one update from the current gradients. Does not clear them.
  virtual void step() = 0;
  /// "sgd", "adam" or "rmsprop".
  virtual std::string type() const = 0;

  /// Clears the gradients of all managed parameters.
  void zero_grad();
  float learning_rate() const { return lr_; }
  void set_learning_rate(float lr) { lr_ = lr; }
  float weight_decay() const { return weight_decay_; }
  /// L2 penalty, added to the gradient (not decoupled as in AdamW).
  void set_weight_decay(float wd) { weight_decay_ = wd; }
  const std::vector<NamedTensor>& params() const { return params_; }
  const ParamState& state(size_t i) const { return states_.at(i); }
  /// Buffers of a parameter by name; null if the optimizer does not manage it.
  const ParamState* state(const std::string& param_name) const;
  /// Number of step() calls.
  long steps() const { return steps_; }

 protected:
  bool active(size_t i) const;
  std::vector<NamedTensor> params_;
  std::vector<ParamState> states_;
  float lr_;
  float weight_decay_ = 0.0f;
  long steps_ = 0;
};

/// v = momentum * v + g; w -= lr * v, or w -= lr * (g + momentum * v) with Nesterov.
/// Same convention as torch.optim.SGD.
class SGD : public Optimizer {
 public:
  SGD(std::vector<NamedTensor> params, float lr, float momentum = 0.0f, bool nesterov = false);
  void step() override;
  std::string type() const override { return "sgd"; }

 private:
  float momentum_;
  bool nesterov_;
};

/// Adam with bias correction. Buffers: m (first moment), v (second moment).
class Adam : public Optimizer {
 public:
  Adam(std::vector<NamedTensor> params, float lr = 1e-3f, float beta1 = 0.9f, float beta2 = 0.999f,
       float eps = 1e-8f);
  void step() override;
  std::string type() const override { return "adam"; }

 private:
  float b1_, b2_, eps_;
};

/// v = alpha * v + (1 - alpha) * g^2; w -= lr * g / (sqrt(v) + eps), with optional
/// momentum on the normalised update.
class RMSProp : public Optimizer {
 public:
  RMSProp(std::vector<NamedTensor> params, float lr = 1e-3f, float alpha = 0.99f, float eps = 1e-5f,
          float momentum = 0.0f);
  void step() override;
  std::string type() const override { return "rmsprop"; }

 private:
  float alpha_, eps_, momentum_;
};

/// Creates an optimizer by name (sgd | adam | rmsprop). `momentum` and `nesterov`
/// apply to SGD only; Adam and RMSProp use their constructor defaults.
std::unique_ptr<Optimizer> make_optimizer(const std::string& type, std::vector<NamedTensor> params, float lr,
                                          float momentum = 0.9f, bool nesterov = false);

/// Builds an optimizer from global hyperparameters: "optimizer" (sgd|adam|rmsprop),
/// "learning_rate", "momentum", "nesterov", "weight_decay". Missing keys use the
/// given defaults.
std::unique_ptr<Optimizer> make_optimizer_from_globals(const HyperparamValues& globals,
                                                       std::vector<NamedTensor> params,
                                                       const std::string& default_type = "adam",
                                                       float default_lr = 1e-3f);

}  // namespace cdn
