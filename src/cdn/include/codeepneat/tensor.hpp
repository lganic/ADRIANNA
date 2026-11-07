#pragma once
// A small, dependency-free, reverse-mode automatic differentiation engine.
//
// Design goals (driven by the "tight control" requirement):
//   * Every tensor owns plain std::vector<float> storage for its values and its
//     gradient, both directly readable and writable (data(), grad_data()).
//   * Every intermediate tensor (activation) keeps its gradient after backward(),
//     so activation gradients can be inspected without extra bookkeeping.
//   * Gradient hooks can be attached to *any* tensor (parameters or activations)
//     to observe, modify or replace the gradient flowing through it.
//   * No global state other than a thread_local "grad enabled" flag, so separate
//     networks can be trained concurrently on different threads.
//
// Layout is row-major. Images use NCHW. The first axis is the batch axis for
// all layer-level operations.

#include <cstddef>
#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "codeepneat/rng.hpp"

namespace cdn {

/// Tensor dimensions, outermost first. An empty Shape is a scalar with one element.
using Shape = std::vector<int>;
/// Number of elements in `shape`. Throws std::invalid_argument on negative sizes.
size_t shape_numel(const Shape& shape);
/// Formats a shape as "[2, 3]".
std::string shape_str(const Shape& shape);

class Tensor;

/// Gradient hook: called with the gradient flowing into a tensor during
/// backward(). The hook may read it, modify it in place, or assign a new tensor
/// of the same shape. For leaf tensors (parameters) the hook sees the gradient
/// contribution of the current backward pass only (before accumulation).
using GradHook = std::function<void(Tensor& grad)>;

/// Shared storage behind a Tensor handle, and one node of the autograd graph.
/// Most code should use Tensor; this is public so that custom operations can be
/// written and graphs can be walked.
struct TensorImpl {
  Shape shape;
  std::vector<float> data;  ///< values, row-major
  std::vector<float> grad;  ///< same size as data once backward() reached this node, else empty
  bool requires_grad = false;
  std::string name;         ///< optional label; parameters carry their full name
  const char* op = "leaf";  ///< name of the operation that produced this tensor
  std::vector<std::shared_ptr<TensorImpl>> parents;  ///< inputs of that operation (keeps them alive)
  /// Adds this node's contribution to its parents' grad buffers. Must capture
  /// parents as raw pointers only: `parents` owns them, and a shared_ptr capture
  /// would defeat the iterative release in the destructor.
  std::function<void(TensorImpl&)> backward_fn;
  std::vector<std::pair<int, GradHook>> hooks;  ///< (id, hook), run in registration order
  int next_hook_id = 1;

  /// A leaf has no backward function: parameters, inputs, constants, detached tensors.
  bool is_leaf() const { return !backward_fn; }

  TensorImpl() = default;
  TensorImpl(const TensorImpl&) = delete;
  TensorImpl& operator=(const TensorImpl&) = delete;
  ~TensorImpl();  // iterative release of long graphs (no recursion blow-up)
};

/// Reference-counted handle to a dense float32 array and its autograd node.
///
/// Copying a Tensor copies the handle, not the values: both copies see the same
/// data, gradient and hooks. Use clone() (differentiable) or detach() (not
/// connected to the graph) for a copy of the values.
///
/// Operations allocate a new tensor, with three deliberate exceptions that return
/// their input unchanged because they have nothing to do: activation("linear"),
/// dropout() in eval mode or with p = 0, and reshape()/flatten() to the same shape.
/// Keep that in mind before editing `.data()` in place.
///
/// Autograd reads saved values when backward() runs, so editing the data of a
/// tensor between forward and backward changes the gradients computed from it.
///
/// A default-constructed Tensor is "undefined". defined(), requires_grad(),
/// has_grad() and to_string() are safe on it; value and shape access throw
/// std::logic_error.
class Tensor {
 public:
  Tensor() = default;
  /// Wraps an existing node.
  explicit Tensor(std::shared_ptr<TensorImpl> impl);
  /// Throws std::invalid_argument if data.size() does not match `shape`.
  Tensor(Shape shape, std::vector<float> data, bool requires_grad = false);

  static Tensor zeros(const Shape& shape, bool requires_grad = false);
  static Tensor ones(const Shape& shape, bool requires_grad = false);
  static Tensor full(const Shape& shape, float value, bool requires_grad = false);
  /// Rank-0 tensor (Shape{}).
  static Tensor scalar(float value, bool requires_grad = false);
  /// Gaussian values with mean 0.
  static Tensor randn(const Shape& shape, Rng& rng, float stddev = 1.0f, bool requires_grad = false);
  /// Uniform values in [lo, hi).
  static Tensor uniform(const Shape& shape, Rng& rng, float lo, float hi, bool requires_grad = false);

  /// False for a default-constructed tensor.
  bool defined() const { return static_cast<bool>(impl_); }
  const Shape& shape() const;
  int dim() const;
  int size(int axis) const;  // negative axes allowed
  size_t numel() const;

  // ---- raw value access (always allowed; edits bypass autograd) ----
  std::vector<float>& data();
  const std::vector<float>& data() const;
  float* ptr();
  const float* ptr() const;
  /// The single value of a one-element tensor; throws otherwise.
  float item() const;
  /// Bounds-checked element access, e.g. t.at({1, 2}).
  float& at(const std::vector<int>& index);
  float at(const std::vector<int>& index) const;
  void fill(float value);
  void copy_from(const Tensor& other);  // copies values; shapes must have equal numel

  // ---- autograd ----
  bool requires_grad() const;
  /// Turn gradient tracking on or off for this tensor (how parameters are frozen).
  Tensor& set_requires_grad(bool on);
  bool is_leaf() const;
  /// Name of the producing operation, "leaf" for leaves.
  const char* op() const;
  const std::string& name() const;
  Tensor& set_name(const std::string& name);

  /// True once a gradient of the right size exists.
  bool has_grad() const;
  std::vector<float>& grad_data();          // empty if no gradient yet
  const std::vector<float>& grad_data() const;
  Tensor grad() const;                      // copy as a tensor (zeros if none)
  /// Clears the gradient. has_grad() becomes false, so optimizers skip this
  /// tensor until the next backward().
  void zero_grad();

  /// Backpropagate from a scalar tensor (seed = 1).
  ///
  /// Leaf gradients accumulate across calls until zero_grad(). Gradients of
  /// intermediate tensors are recomputed by each pass and stay readable until the
  /// tensor is released. The graph is not freed by backward(), so it can be
  /// traversed again; it is released when the last handle to its output goes away.
  void backward();
  /// Backpropagate with an explicit seed gradient of the same shape.
  void backward(const Tensor& seed);

  /// Adds a hook; returns an id for remove_hook(). See GradHook.
  int register_hook(GradHook hook);
  void remove_hook(int id);
  void clear_hooks();
  size_t num_hooks() const;

  /// New leaf tensor holding a copy of the values, not connected to the graph.
  Tensor detach() const;
  /// Differentiable copy.
  Tensor clone() const;

  /// Debug description: shape, op, name and the first `max_values` values.
  std::string to_string(size_t max_values = 12) const;

  TensorImpl* impl() const { return impl_.get(); }
  const std::shared_ptr<TensorImpl>& impl_ptr() const { return impl_; }
  /// True if both handles refer to the same storage.
  bool is_same(const Tensor& other) const { return impl_ == other.impl_; }

 private:
  TensorImpl& req() const;
  std::shared_ptr<TensorImpl> impl_;
};

/// Thread-local switch for graph recording. New threads start with recording on.
struct GradMode {
  static bool enabled();
  static void set(bool enabled);
};

/// RAII guard disabling graph recording (inference / acting in RL).
class NoGradGuard {
 public:
  NoGradGuard();
  ~NoGradGuard();
  NoGradGuard(const NoGradGuard&) = delete;
  NoGradGuard& operator=(const NoGradGuard&) = delete;

 private:
  bool prev_;
};

// ============================================================================
// Operations (all differentiable unless stated otherwise)
// ============================================================================

// Element-wise binary ops with numpy-style broadcasting.
Tensor add(const Tensor& a, const Tensor& b);
Tensor sub(const Tensor& a, const Tensor& b);
Tensor mul(const Tensor& a, const Tensor& b);
Tensor div(const Tensor& a, const Tensor& b);
Tensor maximum(const Tensor& a, const Tensor& b);
Tensor minimum(const Tensor& a, const Tensor& b);

Tensor add(const Tensor& a, float s);
Tensor mul(const Tensor& a, float s);
Tensor rsub(float s, const Tensor& a);  // s - a
Tensor rdiv(float s, const Tensor& a);  // s / a

Tensor operator+(const Tensor& a, const Tensor& b);
Tensor operator-(const Tensor& a, const Tensor& b);
Tensor operator*(const Tensor& a, const Tensor& b);
Tensor operator/(const Tensor& a, const Tensor& b);
Tensor operator+(const Tensor& a, float s);
Tensor operator+(float s, const Tensor& a);
Tensor operator-(const Tensor& a, float s);
Tensor operator-(float s, const Tensor& a);
Tensor operator*(const Tensor& a, float s);
Tensor operator*(float s, const Tensor& a);
Tensor operator/(const Tensor& a, float s);
Tensor operator/(float s, const Tensor& a);
Tensor operator-(const Tensor& a);

// Element-wise unary ops.
Tensor neg(const Tensor& x);
Tensor exp(const Tensor& x);
Tensor log(const Tensor& x);
Tensor sqrt(const Tensor& x);
Tensor abs(const Tensor& x);
Tensor square(const Tensor& x);
Tensor pow(const Tensor& x, float p);
Tensor clamp(const Tensor& x, float lo, float hi);
Tensor relu(const Tensor& x);
Tensor leaky_relu(const Tensor& x, float slope = 0.01f);
Tensor elu(const Tensor& x, float alpha = 1.0f);
Tensor tanh(const Tensor& x);
Tensor sigmoid(const Tensor& x);
Tensor softplus(const Tensor& x);

/// Apply an activation by name: linear|identity, relu, leaky_relu, elu, tanh,
/// sigmoid, softplus.
Tensor activation(const Tensor& x, const std::string& name);
bool is_activation(const std::string& name);
const std::vector<std::string>& activation_names();

// Linear algebra.
Tensor matmul(const Tensor& a, const Tensor& b);                     // [M,K]x[K,N]
/// y = x w^T + b, the PyTorch weight layout.
Tensor linear(const Tensor& x, const Tensor& w, const Tensor& b);    // x[B,I], w[O,I], b[O] (b may be undefined)

// Reductions. Without an axis the result is a rank-0 scalar. Negative axes count
// from the end.
Tensor sum(const Tensor& x);
Tensor mean(const Tensor& x);
Tensor sum(const Tensor& x, int axis, bool keepdim = false);
Tensor mean(const Tensor& x, int axis, bool keepdim = false);
/// Maximum along an axis; the gradient goes to the first maximal element.
Tensor max(const Tensor& x, int axis, bool keepdim = false);

// Shape ops. These copy data; there are no views.
Tensor reshape(const Tensor& x, const Shape& shape);  // one dim may be -1
Tensor flatten(const Tensor& x);                       // keeps axis 0
/// All other dimensions must match. A single input is returned unchanged.
Tensor concat(const std::vector<Tensor>& xs, int axis);
Tensor stack(const std::vector<Tensor>& xs);           // new axis 0
/// Slice [start, start + length) along an axis.
Tensor narrow(const Tensor& x, int axis, int start, int length);

// Probability helpers (last axis).
Tensor softmax(const Tensor& x);
Tensor log_softmax(const Tensor& x);
Tensor gather(const Tensor& x, const std::vector<int>& index);  // x[B,N] -> [B]

// Convolution / pooling (NCHW).
/// x[B,C,H,W], w[O,C,KH,KW], b[O] (may be undefined). Output size is
/// (H + 2*padding - KH) / stride + 1, and likewise for W.
Tensor conv2d(const Tensor& x, const Tensor& w, const Tensor& b, int stride = 1, int padding = 0);
/// Output size is (H - kernel) / stride + 1, and likewise for W.
Tensor max_pool2d(const Tensor& x, int kernel, int stride);
/// Pools to exactly out_h x out_w using the same bins as PyTorch's
/// adaptive_max_pool2d. Cannot upsample. Returns x unchanged if the size matches.
Tensor adaptive_max_pool2d(const Tensor& x, int out_h, int out_w);

// Regularisation.
/// Inverted dropout: survivors are scaled by 1 / (1 - p).
Tensor dropout(const Tensor& x, float p, Rng& rng, bool training);

// Losses (return scalars).
Tensor mse_loss(const Tensor& pred, const Tensor& target);
Tensor huber_loss(const Tensor& pred, const Tensor& target, float delta = 1.0f);
Tensor cross_entropy(const Tensor& logits, const std::vector<int>& targets);

// Non-differentiable helpers.
std::vector<int> argmax_rows(const Tensor& x);  // x[B,N]
Tensor one_hot(const std::vector<int>& index, int n);
float grad_norm(const std::vector<Tensor>& params);
/// Scales gradients so that their global L2 norm is at most max_norm.
/// Returns the norm before clipping.
float clip_grad_norm(const std::vector<Tensor>& params, float max_norm);

}  // namespace cdn
