// Tensor handles, reverse-mode backward(), and every differentiable operation.
// Backward functions capture parents as raw pointers (TensorImpl::parents owns them).

#include "codeepneat/tensor.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <unordered_set>

namespace cdn {

// ============================================================================
// Grad mode
// ============================================================================

namespace {
thread_local bool t_grad_enabled = true;
}

bool GradMode::enabled() { return t_grad_enabled; }
void GradMode::set(bool enabled) { t_grad_enabled = enabled; }
NoGradGuard::NoGradGuard() : prev_(t_grad_enabled) { t_grad_enabled = false; }
NoGradGuard::~NoGradGuard() { t_grad_enabled = prev_; }

// ============================================================================
// Shapes
// ============================================================================

size_t shape_numel(const Shape& shape) {
  size_t n = 1;
  for (int d : shape) {
    if (d < 0) throw std::invalid_argument("negative dimension in shape " + shape_str(shape));
    n *= static_cast<size_t>(d);
  }
  return n;
}

std::string shape_str(const Shape& shape) {
  std::ostringstream os;
  os << '[';
  for (size_t i = 0; i < shape.size(); ++i) {
    if (i) os << ", ";
    os << shape[i];
  }
  os << ']';
  return os.str();
}

// ============================================================================
// TensorImpl
// ============================================================================

TensorImpl::~TensorImpl() {
  // Release the parent chain iteratively; a recurrent network unrolled over a
  // long episode can produce graphs tens of thousands of nodes deep.
  std::vector<std::shared_ptr<TensorImpl>> stack;
  stack.swap(parents);
  while (!stack.empty()) {
    std::shared_ptr<TensorImpl> p = std::move(stack.back());
    stack.pop_back();
    if (p && p.use_count() == 1) {
      for (auto& q : p->parents) stack.push_back(std::move(q));
      p->parents.clear();
    }
  }
}

// ============================================================================
// Tensor
// ============================================================================

Tensor::Tensor(std::shared_ptr<TensorImpl> impl) : impl_(std::move(impl)) {}

Tensor::Tensor(Shape shape, std::vector<float> data, bool requires_grad)
    : impl_(std::make_shared<TensorImpl>()) {
  if (data.size() != shape_numel(shape))
    throw std::invalid_argument("Tensor: data size " + std::to_string(data.size()) +
                                " does not match shape " + shape_str(shape));
  impl_->shape = std::move(shape);
  impl_->data = std::move(data);
  impl_->requires_grad = requires_grad;
}

Tensor Tensor::zeros(const Shape& shape, bool rg) { return full(shape, 0.0f, rg); }
Tensor Tensor::ones(const Shape& shape, bool rg) { return full(shape, 1.0f, rg); }
Tensor Tensor::full(const Shape& shape, float value, bool rg) {
  return Tensor(shape, std::vector<float>(shape_numel(shape), value), rg);
}
Tensor Tensor::scalar(float value, bool rg) { return Tensor(Shape{}, {value}, rg); }
Tensor Tensor::randn(const Shape& shape, Rng& rng, float stddev, bool rg) {
  std::vector<float> d(shape_numel(shape));
  for (auto& v : d) v = static_cast<float>(rng.normal(0.0, stddev));
  return Tensor(shape, std::move(d), rg);
}
Tensor Tensor::uniform(const Shape& shape, Rng& rng, float lo, float hi, bool rg) {
  std::vector<float> d(shape_numel(shape));
  for (auto& v : d) v = static_cast<float>(rng.uniform(lo, hi));
  return Tensor(shape, std::move(d), rg);
}

TensorImpl& Tensor::req() const {
  if (!impl_) throw std::logic_error("operation on undefined tensor");
  return *impl_;
}

const Shape& Tensor::shape() const { return req().shape; }
int Tensor::dim() const { return static_cast<int>(req().shape.size()); }
int Tensor::size(int axis) const {
  int r = dim();
  if (axis < 0) axis += r;
  if (axis < 0 || axis >= r) throw std::out_of_range("Tensor::size: axis out of range");
  return req().shape[axis];
}
size_t Tensor::numel() const { return req().data.size(); }
std::vector<float>& Tensor::data() { return req().data; }
const std::vector<float>& Tensor::data() const { return req().data; }
float* Tensor::ptr() { return req().data.data(); }
const float* Tensor::ptr() const { return req().data.data(); }

float Tensor::item() const {
  if (numel() != 1) throw std::logic_error("item(): tensor has " + std::to_string(numel()) + " elements");
  return req().data[0];
}

static size_t flat_index(const Shape& shape, const std::vector<int>& index) {
  if (index.size() != shape.size()) throw std::out_of_range("at(): wrong number of indices");
  size_t off = 0;
  for (size_t i = 0; i < shape.size(); ++i) {
    if (index[i] < 0 || index[i] >= shape[i]) throw std::out_of_range("at(): index out of range");
    off = off * shape[i] + index[i];
  }
  return off;
}

float& Tensor::at(const std::vector<int>& index) { return req().data[flat_index(req().shape, index)]; }
float Tensor::at(const std::vector<int>& index) const { return req().data[flat_index(req().shape, index)]; }
void Tensor::fill(float value) { std::fill(req().data.begin(), req().data.end(), value); }
void Tensor::copy_from(const Tensor& other) {
  if (other.numel() != numel()) throw std::invalid_argument("copy_from: size mismatch");
  req().data = other.data();
}

bool Tensor::requires_grad() const { return impl_ && impl_->requires_grad; }
Tensor& Tensor::set_requires_grad(bool on) {
  req().requires_grad = on;
  return *this;
}
bool Tensor::is_leaf() const { return req().is_leaf(); }
const char* Tensor::op() const { return req().op; }
const std::string& Tensor::name() const { return req().name; }
Tensor& Tensor::set_name(const std::string& name) {
  req().name = name;
  return *this;
}

bool Tensor::has_grad() const { return impl_ && impl_->grad.size() == impl_->data.size() && !impl_->data.empty(); }
std::vector<float>& Tensor::grad_data() { return req().grad; }
const std::vector<float>& Tensor::grad_data() const { return req().grad; }
Tensor Tensor::grad() const {
  if (!has_grad()) return Tensor::zeros(shape());
  return Tensor(shape(), req().grad);
}
void Tensor::zero_grad() { req().grad.clear(); }

int Tensor::register_hook(GradHook hook) {
  auto& t = req();
  int id = t.next_hook_id++;
  t.hooks.emplace_back(id, std::move(hook));
  return id;
}
void Tensor::remove_hook(int id) {
  auto& h = req().hooks;
  h.erase(std::remove_if(h.begin(), h.end(), [id](const auto& p) { return p.first == id; }), h.end());
}
void Tensor::clear_hooks() { req().hooks.clear(); }
size_t Tensor::num_hooks() const { return req().hooks.size(); }

Tensor Tensor::detach() const {
  Tensor t(shape(), data(), false);
  t.impl_->name = req().name;
  return t;
}

std::string Tensor::to_string(size_t max_values) const {
  if (!impl_) return "Tensor(undefined)";
  std::ostringstream os;
  os << "Tensor(shape=" << shape_str(impl_->shape) << ", op=" << impl_->op;
  if (!impl_->name.empty()) os << ", name=" << impl_->name;
  if (impl_->requires_grad) os << ", requires_grad";
  os << ", data=[";
  size_t n = std::min(max_values, impl_->data.size());
  for (size_t i = 0; i < n; ++i) {
    if (i) os << ", ";
    os << impl_->data[i];
  }
  if (n < impl_->data.size()) os << ", ...";
  os << "])";
  return os.str();
}

// ============================================================================
// Backward
// ============================================================================

namespace {

void run_hooks(TensorImpl& n) {
  if (n.hooks.empty()) return;
  auto hooks = n.hooks;  // a hook may remove itself
  Tensor g(n.shape, std::move(n.grad));
  for (auto& h : hooks) h.second(g);
  if (!g.defined() || g.numel() != n.data.size())
    throw std::logic_error("gradient hook changed the gradient's size");
  n.grad = std::move(g.data());
}

}  // namespace

void Tensor::backward() { backward(Tensor()); }

void Tensor::backward(const Tensor& seed) {
  TensorImpl& root = req();
  if (!root.requires_grad) throw std::logic_error("backward(): tensor does not require grad");
  if (seed.defined() && seed.numel() != root.data.size())
    throw std::invalid_argument("backward(): seed size mismatch");
  if (!seed.defined() && root.data.size() != 1)
    throw std::logic_error("backward(): implicit seed requires a scalar tensor");

  // Iterative post-order DFS: parents are emitted before their consumers.
  std::vector<TensorImpl*> order;
  std::unordered_set<TensorImpl*> visited;
  std::vector<std::pair<TensorImpl*, size_t>> stack;
  stack.emplace_back(&root, 0);
  visited.insert(&root);
  while (!stack.empty()) {
    TensorImpl* node = stack.back().first;
    size_t i = stack.back().second;
    if (i < node->parents.size()) {
      stack.back().second = i + 1;
      TensorImpl* p = node->parents[i].get();
      if (p->requires_grad && visited.insert(p).second) stack.emplace_back(p, 0);
    } else {
      order.push_back(node);
      stack.pop_back();
    }
  }

  // Leaves accumulate across backward passes; stash their existing gradient so
  // hooks see only this pass's contribution.
  std::vector<std::vector<float>> saved(order.size());
  for (size_t k = 0; k < order.size(); ++k) {
    TensorImpl* n = order[k];
    if (n->is_leaf()) saved[k].swap(n->grad);
    n->grad.assign(n->data.size(), 0.0f);
  }
  if (seed.defined())
    root.grad = seed.data();
  else
    root.grad[0] = 1.0f;

  for (size_t k = order.size(); k-- > 0;) {
    TensorImpl* n = order[k];
    if (n->is_leaf()) continue;
    run_hooks(*n);
    n->backward_fn(*n);
  }
  for (size_t k = 0; k < order.size(); ++k) {
    TensorImpl* n = order[k];
    if (!n->is_leaf()) continue;
    run_hooks(*n);
    if (saved[k].size() == n->grad.size())
      for (size_t i = 0; i < n->grad.size(); ++i) n->grad[i] += saved[k][i];
  }
}

// ============================================================================
// Op helpers
// ============================================================================

namespace {

using BackwardFn = std::function<void(TensorImpl&)>;

Tensor make_out(Shape shape, std::vector<float> data, const std::vector<Tensor>& parents, const char* op,
                BackwardFn bw) {
  auto impl = std::make_shared<TensorImpl>();
  impl->shape = std::move(shape);
  impl->data = std::move(data);
  impl->op = op;
  if (GradMode::enabled()) {
    bool need = false;
    for (const auto& p : parents)
      if (p.defined() && p.requires_grad()) {
        need = true;
        break;
      }
    if (need) {
      impl->requires_grad = true;
      impl->parents.reserve(parents.size());
      for (const auto& p : parents)
        if (p.defined()) impl->parents.push_back(p.impl_ptr());
      impl->backward_fn = std::move(bw);
    }
  }
  return Tensor(std::move(impl));
}

inline float* gradp(TensorImpl* p) {
  if (!p || !p->requires_grad || p->grad.size() != p->data.size()) return nullptr;
  return p->grad.data();
}

int norm_axis(int axis, int rank) {
  int a = axis < 0 ? axis + rank : axis;
  if (a < 0 || a >= rank) throw std::out_of_range("axis " + std::to_string(axis) + " out of range for rank " +
                                                  std::to_string(rank));
  return a;
}

// C[M,N] += A[M,K] * B[K,N]
void gemm_nn(int M, int N, int K, const float* A, const float* B, float* C) {
  for (int i = 0; i < M; ++i) {
    float* c = C + static_cast<size_t>(i) * N;
    const float* a = A + static_cast<size_t>(i) * K;
    for (int k = 0; k < K; ++k) {
      float av = a[k];
      if (av == 0.0f) continue;
      const float* b = B + static_cast<size_t>(k) * N;
      for (int j = 0; j < N; ++j) c[j] += av * b[j];
    }
  }
}
// C[M,N] += A[M,K] * B[N,K]^T
void gemm_nt(int M, int N, int K, const float* A, const float* B, float* C) {
  for (int i = 0; i < M; ++i) {
    const float* a = A + static_cast<size_t>(i) * K;
    float* c = C + static_cast<size_t>(i) * N;
    for (int j = 0; j < N; ++j) {
      const float* b = B + static_cast<size_t>(j) * K;
      float s = 0.0f;
      for (int k = 0; k < K; ++k) s += a[k] * b[k];
      c[j] += s;
    }
  }
}
// C[M,N] += A[K,M]^T * B[K,N]
void gemm_tn(int M, int N, int K, const float* A, const float* B, float* C) {
  for (int k = 0; k < K; ++k) {
    const float* a = A + static_cast<size_t>(k) * M;
    const float* b = B + static_cast<size_t>(k) * N;
    for (int i = 0; i < M; ++i) {
      float av = a[i];
      if (av == 0.0f) continue;
      float* c = C + static_cast<size_t>(i) * N;
      for (int j = 0; j < N; ++j) c[j] += av * b[j];
    }
  }
}

Shape broadcast_shapes(const Shape& a, const Shape& b) {
  size_t n = std::max(a.size(), b.size());
  Shape out(n);
  for (size_t i = 0; i < n; ++i) {
    int da = i < n - a.size() ? 1 : a[i - (n - a.size())];
    int db = i < n - b.size() ? 1 : b[i - (n - b.size())];
    if (da != db && da != 1 && db != 1)
      throw std::invalid_argument("broadcast: incompatible shapes " + shape_str(a) + " and " + shape_str(b));
    out[i] = std::max(da, db);
  }
  return out;
}

std::vector<size_t> broadcast_map(const Shape& out, const Shape& in) {
  size_t n = out.size(), off = n - in.size();
  std::vector<size_t> stride(n, 0);
  size_t s = 1;
  for (size_t k = in.size(); k-- > 0;) {
    stride[k + off] = (in[k] == 1) ? 0 : s;
    s *= static_cast<size_t>(in[k]);
  }
  size_t total = shape_numel(out);
  std::vector<size_t> map(total);
  std::vector<int> ctr(n, 0);
  size_t cur = 0;
  for (size_t t = 0; t < total; ++t) {
    map[t] = cur;
    for (size_t d = n; d-- > 0;) {
      ++ctr[d];
      cur += stride[d];
      if (ctr[d] < out[d]) break;
      cur -= stride[d] * static_cast<size_t>(out[d]);
      ctr[d] = 0;
    }
  }
  return map;
}

template <class F, class DA, class DB>
Tensor binary_op(const Tensor& a, const Tensor& b, const char* op, F f, DA da, DB db) {
  Shape os = broadcast_shapes(a.shape(), b.shape());
  size_t n = shape_numel(os);
  std::vector<size_t> ia, ib;
  if (a.shape() != os) ia = broadcast_map(os, a.shape());
  if (b.shape() != os) ib = broadcast_map(os, b.shape());
  const float* A = a.ptr();
  const float* B = b.ptr();
  std::vector<float> out(n);
  for (size_t i = 0; i < n; ++i) out[i] = f(A[ia.empty() ? i : ia[i]], B[ib.empty() ? i : ib[i]]);
  TensorImpl* pa = a.impl();
  TensorImpl* pb = b.impl();
  return make_out(std::move(os), std::move(out), {a, b}, op,
                  [pa, pb, ia = std::move(ia), ib = std::move(ib), da, db](TensorImpl& self) {
                    float* ga = gradp(pa);
                    float* gb = gradp(pb);
                    const float* g = self.grad.data();
                    const float* z = self.data.data();
                    const float* A = pa->data.data();
                    const float* B = pb->data.data();
                    size_t n = self.data.size();
                    for (size_t i = 0; i < n; ++i) {
                      size_t ja = ia.empty() ? i : ia[i], jb = ib.empty() ? i : ib[i];
                      float x = A[ja], y = B[jb];
                      if (ga) ga[ja] += g[i] * da(x, y, z[i]);
                      if (gb) gb[jb] += g[i] * db(x, y, z[i]);
                    }
                  });
}

template <class F, class DF>
Tensor unary_op(const Tensor& x, const char* op, F f, DF df) {
  const auto& xd = x.data();
  std::vector<float> out(xd.size());
  for (size_t i = 0; i < xd.size(); ++i) out[i] = f(xd[i]);
  TensorImpl* px = x.impl();
  return make_out(x.shape(), std::move(out), {x}, op, [px, df](TensorImpl& self) {
    float* gx = gradp(px);
    if (!gx) return;
    const float* g = self.grad.data();
    const float* y = self.data.data();
    const float* xv = px->data.data();
    for (size_t i = 0; i < self.data.size(); ++i) gx[i] += g[i] * df(xv[i], y[i]);
  });
}

inline float sigmoidf(float x) {
  if (x >= 0.0f) {
    float e = std::exp(-x);
    return 1.0f / (1.0f + e);
  }
  float e = std::exp(x);
  return e / (1.0f + e);
}

struct AxisSplit {
  size_t outer, dim, inner;
};
AxisSplit split_axis(const Shape& s, int axis) {
  AxisSplit r{1, static_cast<size_t>(s[axis]), 1};
  for (int i = 0; i < axis; ++i) r.outer *= s[i];
  for (size_t i = axis + 1; i < s.size(); ++i) r.inner *= s[i];
  return r;
}
Shape reduced_shape(const Shape& s, int axis, bool keepdim) {
  Shape r = s;
  if (keepdim)
    r[axis] = 1;
  else
    r.erase(r.begin() + axis);
  return r;
}

}  // namespace

// ============================================================================
// Element-wise
// ============================================================================

Tensor add(const Tensor& a, const Tensor& b) {
  return binary_op(
      a, b, "add", [](float x, float y) { return x + y; }, [](float, float, float) { return 1.0f; },
      [](float, float, float) { return 1.0f; });
}
Tensor sub(const Tensor& a, const Tensor& b) {
  return binary_op(
      a, b, "sub", [](float x, float y) { return x - y; }, [](float, float, float) { return 1.0f; },
      [](float, float, float) { return -1.0f; });
}
Tensor mul(const Tensor& a, const Tensor& b) {
  return binary_op(
      a, b, "mul", [](float x, float y) { return x * y; }, [](float, float y, float) { return y; },
      [](float x, float, float) { return x; });
}
Tensor div(const Tensor& a, const Tensor& b) {
  return binary_op(
      a, b, "div", [](float x, float y) { return x / y; }, [](float, float y, float) { return 1.0f / y; },
      [](float x, float y, float) { return -x / (y * y); });
}
Tensor maximum(const Tensor& a, const Tensor& b) {
  return binary_op(
      a, b, "maximum", [](float x, float y) { return x >= y ? x : y; },
      [](float x, float y, float) { return x >= y ? 1.0f : 0.0f; },
      [](float x, float y, float) { return x >= y ? 0.0f : 1.0f; });
}
Tensor minimum(const Tensor& a, const Tensor& b) {
  return binary_op(
      a, b, "minimum", [](float x, float y) { return x <= y ? x : y; },
      [](float x, float y, float) { return x <= y ? 1.0f : 0.0f; },
      [](float x, float y, float) { return x <= y ? 0.0f : 1.0f; });
}

Tensor add(const Tensor& a, float s) {
  return unary_op(
      a, "add_scalar", [s](float x) { return x + s; }, [](float, float) { return 1.0f; });
}
Tensor mul(const Tensor& a, float s) {
  return unary_op(
      a, "mul_scalar", [s](float x) { return x * s; }, [s](float, float) { return s; });
}
Tensor rsub(float s, const Tensor& a) {
  return unary_op(
      a, "rsub_scalar", [s](float x) { return s - x; }, [](float, float) { return -1.0f; });
}
Tensor rdiv(float s, const Tensor& a) {
  return unary_op(
      a, "rdiv_scalar", [s](float x) { return s / x; }, [s](float x, float) { return -s / (x * x); });
}

Tensor operator+(const Tensor& a, const Tensor& b) { return add(a, b); }
Tensor operator-(const Tensor& a, const Tensor& b) { return sub(a, b); }
Tensor operator*(const Tensor& a, const Tensor& b) { return mul(a, b); }
Tensor operator/(const Tensor& a, const Tensor& b) { return div(a, b); }
Tensor operator+(const Tensor& a, float s) { return add(a, s); }
Tensor operator+(float s, const Tensor& a) { return add(a, s); }
Tensor operator-(const Tensor& a, float s) { return add(a, -s); }
Tensor operator-(float s, const Tensor& a) { return rsub(s, a); }
Tensor operator*(const Tensor& a, float s) { return mul(a, s); }
Tensor operator*(float s, const Tensor& a) { return mul(a, s); }
Tensor operator/(const Tensor& a, float s) { return mul(a, 1.0f / s); }
Tensor operator/(float s, const Tensor& a) { return rdiv(s, a); }
Tensor operator-(const Tensor& a) { return neg(a); }

Tensor Tensor::clone() const {
  return unary_op(
      *this, "clone", [](float v) { return v; }, [](float, float) { return 1.0f; });
}

Tensor neg(const Tensor& x) {
  return unary_op(
      x, "neg", [](float v) { return -v; }, [](float, float) { return -1.0f; });
}
Tensor exp(const Tensor& x) {
  return unary_op(
      x, "exp", [](float v) { return std::exp(v); }, [](float, float y) { return y; });
}
Tensor log(const Tensor& x) {
  return unary_op(
      x, "log", [](float v) { return std::log(v); }, [](float v, float) { return 1.0f / v; });
}
Tensor sqrt(const Tensor& x) {
  return unary_op(
      x, "sqrt", [](float v) { return std::sqrt(v); }, [](float, float y) { return y > 0 ? 0.5f / y : 0.0f; });
}
Tensor abs(const Tensor& x) {
  return unary_op(
      x, "abs", [](float v) { return std::fabs(v); },
      [](float v, float) { return v > 0 ? 1.0f : (v < 0 ? -1.0f : 0.0f); });
}
Tensor square(const Tensor& x) {
  return unary_op(
      x, "square", [](float v) { return v * v; }, [](float v, float) { return 2.0f * v; });
}
Tensor pow(const Tensor& x, float p) {
  return unary_op(
      x, "pow", [p](float v) { return std::pow(v, p); },
      [p](float v, float) { return p * std::pow(v, p - 1.0f); });
}
Tensor clamp(const Tensor& x, float lo, float hi) {
  return unary_op(
      x, "clamp", [lo, hi](float v) { return std::min(hi, std::max(lo, v)); },
      [lo, hi](float v, float) { return (v >= lo && v <= hi) ? 1.0f : 0.0f; });
}
Tensor relu(const Tensor& x) {
  return unary_op(
      x, "relu", [](float v) { return v > 0 ? v : 0.0f; }, [](float v, float) { return v > 0 ? 1.0f : 0.0f; });
}
Tensor leaky_relu(const Tensor& x, float slope) {
  return unary_op(
      x, "leaky_relu", [slope](float v) { return v > 0 ? v : slope * v; },
      [slope](float v, float) { return v > 0 ? 1.0f : slope; });
}
Tensor elu(const Tensor& x, float alpha) {
  return unary_op(
      x, "elu", [alpha](float v) { return v > 0 ? v : alpha * (std::exp(v) - 1.0f); },
      [alpha](float v, float y) { return v > 0 ? 1.0f : y + alpha; });
}
Tensor tanh(const Tensor& x) {
  return unary_op(
      x, "tanh", [](float v) { return std::tanh(v); }, [](float, float y) { return 1.0f - y * y; });
}
Tensor sigmoid(const Tensor& x) {
  return unary_op(
      x, "sigmoid", [](float v) { return sigmoidf(v); }, [](float, float y) { return y * (1.0f - y); });
}
Tensor softplus(const Tensor& x) {
  return unary_op(
      x, "softplus", [](float v) { return std::log1p(std::exp(-std::fabs(v))) + std::max(v, 0.0f); },
      [](float v, float) { return sigmoidf(v); });
}

const std::vector<std::string>& activation_names() {
  static const std::vector<std::string> names = {"linear", "relu", "leaky_relu", "elu",
                                                 "tanh",   "sigmoid", "softplus"};
  return names;
}
bool is_activation(const std::string& name) {
  if (name == "identity" || name.empty()) return true;
  const auto& n = activation_names();
  return std::find(n.begin(), n.end(), name) != n.end();
}
Tensor activation(const Tensor& x, const std::string& name) {
  if (name == "linear" || name == "identity" || name.empty()) return x;
  if (name == "relu") return relu(x);
  if (name == "leaky_relu") return leaky_relu(x);
  if (name == "elu") return elu(x);
  if (name == "tanh") return tanh(x);
  if (name == "sigmoid") return sigmoid(x);
  if (name == "softplus") return softplus(x);
  throw std::invalid_argument("unknown activation '" + name + "'");
}

// ============================================================================
// Linear algebra
// ============================================================================

Tensor matmul(const Tensor& a, const Tensor& b) {
  if (a.dim() != 2 || b.dim() != 2 || a.shape()[1] != b.shape()[0])
    throw std::invalid_argument("matmul: incompatible shapes " + shape_str(a.shape()) + " x " + shape_str(b.shape()));
  int M = a.shape()[0], K = a.shape()[1], N = b.shape()[1];
  std::vector<float> out(static_cast<size_t>(M) * N, 0.0f);
  gemm_nn(M, N, K, a.ptr(), b.ptr(), out.data());
  TensorImpl* pa = a.impl();
  TensorImpl* pb = b.impl();
  return make_out({M, N}, std::move(out), {a, b}, "matmul", [pa, pb, M, N, K](TensorImpl& self) {
    const float* g = self.grad.data();
    if (float* ga = gradp(pa)) gemm_nt(M, K, N, g, pb->data.data(), ga);
    if (float* gb = gradp(pb)) gemm_tn(K, N, M, pa->data.data(), g, gb);
  });
}

Tensor linear(const Tensor& x, const Tensor& w, const Tensor& b) {
  if (x.dim() != 2 || w.dim() != 2 || x.shape()[1] != w.shape()[1])
    throw std::invalid_argument("linear: incompatible shapes x" + shape_str(x.shape()) + " w" + shape_str(w.shape()));
  int B = x.shape()[0], I = x.shape()[1], O = w.shape()[0];
  if (b.defined() && (b.numel() != static_cast<size_t>(O)))
    throw std::invalid_argument("linear: bias size mismatch");
  std::vector<float> out(static_cast<size_t>(B) * O, 0.0f);
  gemm_nt(B, O, I, x.ptr(), w.ptr(), out.data());
  if (b.defined()) {
    const float* bb = b.ptr();
    for (int r = 0; r < B; ++r)
      for (int o = 0; o < O; ++o) out[static_cast<size_t>(r) * O + o] += bb[o];
  }
  TensorImpl* px = x.impl();
  TensorImpl* pw = w.impl();
  TensorImpl* pb = b.defined() ? b.impl() : nullptr;
  return make_out({B, O}, std::move(out), {x, w, b}, "linear", [px, pw, pb, B, I, O](TensorImpl& self) {
    const float* g = self.grad.data();
    if (float* gx = gradp(px)) gemm_nn(B, I, O, g, pw->data.data(), gx);
    if (float* gw = gradp(pw)) gemm_tn(O, I, B, g, px->data.data(), gw);
    if (float* gb = gradp(pb))
      for (int r = 0; r < B; ++r)
        for (int o = 0; o < O; ++o) gb[o] += g[static_cast<size_t>(r) * O + o];
  });
}

// ============================================================================
// Reductions
// ============================================================================

Tensor sum(const Tensor& x) {
  double s = 0.0;
  for (float v : x.data()) s += v;
  TensorImpl* px = x.impl();
  return make_out({}, {static_cast<float>(s)}, {x}, "sum", [px](TensorImpl& self) {
    float* gx = gradp(px);
    if (!gx) return;
    float g = self.grad[0];
    for (size_t i = 0; i < px->data.size(); ++i) gx[i] += g;
  });
}

Tensor mean(const Tensor& x) {
  size_t n = x.numel();
  if (n == 0) throw std::invalid_argument("mean of empty tensor");
  double s = 0.0;
  for (float v : x.data()) s += v;
  TensorImpl* px = x.impl();
  return make_out({}, {static_cast<float>(s / n)}, {x}, "mean", [px, n](TensorImpl& self) {
    float* gx = gradp(px);
    if (!gx) return;
    float g = self.grad[0] / static_cast<float>(n);
    for (size_t i = 0; i < n; ++i) gx[i] += g;
  });
}

static Tensor sum_axis_impl(const Tensor& x, int axis, bool keepdim, bool average) {
  axis = norm_axis(axis, x.dim());
  AxisSplit s = split_axis(x.shape(), axis);
  float scale = average ? 1.0f / static_cast<float>(s.dim) : 1.0f;
  std::vector<float> out(s.outer * s.inner, 0.0f);
  const float* X = x.ptr();
  for (size_t o = 0; o < s.outer; ++o)
    for (size_t d = 0; d < s.dim; ++d)
      for (size_t i = 0; i < s.inner; ++i) out[o * s.inner + i] += X[(o * s.dim + d) * s.inner + i];
  if (average)
    for (auto& v : out) v *= scale;
  TensorImpl* px = x.impl();
  return make_out(reduced_shape(x.shape(), axis, keepdim), std::move(out), {x}, average ? "mean_axis" : "sum_axis",
                  [px, s, scale](TensorImpl& self) {
                    float* gx = gradp(px);
                    if (!gx) return;
                    const float* g = self.grad.data();
                    for (size_t o = 0; o < s.outer; ++o)
                      for (size_t d = 0; d < s.dim; ++d)
                        for (size_t i = 0; i < s.inner; ++i)
                          gx[(o * s.dim + d) * s.inner + i] += g[o * s.inner + i] * scale;
                  });
}

Tensor sum(const Tensor& x, int axis, bool keepdim) { return sum_axis_impl(x, axis, keepdim, false); }
Tensor mean(const Tensor& x, int axis, bool keepdim) { return sum_axis_impl(x, axis, keepdim, true); }

Tensor max(const Tensor& x, int axis, bool keepdim) {
  axis = norm_axis(axis, x.dim());
  AxisSplit s = split_axis(x.shape(), axis);
  if (s.dim == 0) throw std::invalid_argument("max over empty axis");
  std::vector<float> out(s.outer * s.inner);
  std::vector<size_t> arg(s.outer * s.inner);
  const float* X = x.ptr();
  for (size_t o = 0; o < s.outer; ++o)
    for (size_t i = 0; i < s.inner; ++i) {
      size_t best = o * s.dim * s.inner + i;
      for (size_t d = 1; d < s.dim; ++d) {
        size_t idx = (o * s.dim + d) * s.inner + i;
        if (X[idx] > X[best]) best = idx;
      }
      out[o * s.inner + i] = X[best];
      arg[o * s.inner + i] = best;
    }
  TensorImpl* px = x.impl();
  return make_out(reduced_shape(x.shape(), axis, keepdim), std::move(out), {x}, "max_axis",
                  [px, arg = std::move(arg)](TensorImpl& self) {
                    float* gx = gradp(px);
                    if (!gx) return;
                    for (size_t k = 0; k < arg.size(); ++k) gx[arg[k]] += self.grad[k];
                  });
}

// ============================================================================
// Shape ops
// ============================================================================

Tensor reshape(const Tensor& x, const Shape& shape_in) {
  Shape shape = shape_in;
  int infer = -1;
  size_t known = 1;
  for (size_t i = 0; i < shape.size(); ++i) {
    if (shape[i] == -1) {
      if (infer >= 0) throw std::invalid_argument("reshape: more than one -1");
      infer = static_cast<int>(i);
    } else {
      known *= static_cast<size_t>(shape[i]);
    }
  }
  if (infer >= 0) {
    if (known == 0 || x.numel() % known != 0) throw std::invalid_argument("reshape: cannot infer dimension");
    shape[infer] = static_cast<int>(x.numel() / known);
  }
  if (shape_numel(shape) != x.numel())
    throw std::invalid_argument("reshape: " + shape_str(x.shape()) + " -> " + shape_str(shape));
  if (shape == x.shape()) return x;
  TensorImpl* px = x.impl();
  return make_out(shape, x.data(), {x}, "reshape", [px](TensorImpl& self) {
    float* gx = gradp(px);
    if (!gx) return;
    for (size_t i = 0; i < self.grad.size(); ++i) gx[i] += self.grad[i];
  });
}

Tensor flatten(const Tensor& x) {
  if (x.dim() == 2) return x;
  if (x.dim() < 1) throw std::invalid_argument("flatten: scalar");
  return reshape(x, {x.shape()[0], -1});
}

Tensor concat(const std::vector<Tensor>& xs, int axis) {
  if (xs.empty()) throw std::invalid_argument("concat: no inputs");
  if (xs.size() == 1) return xs[0];
  int rank = xs[0].dim();
  axis = norm_axis(axis, rank);
  Shape os = xs[0].shape();
  os[axis] = 0;
  for (const auto& t : xs) {
    if (t.dim() != rank) throw std::invalid_argument("concat: rank mismatch");
    for (int d = 0; d < rank; ++d)
      if (d != axis && t.shape()[d] != xs[0].shape()[d])
        throw std::invalid_argument("concat: shape mismatch " + shape_str(t.shape()) + " vs " +
                                    shape_str(xs[0].shape()));
    os[axis] += t.shape()[axis];
  }
  size_t outer = 1, inner = 1;
  for (int d = 0; d < axis; ++d) outer *= os[d];
  for (int d = axis + 1; d < rank; ++d) inner *= os[d];
  size_t total = static_cast<size_t>(os[axis]);
  std::vector<float> out(shape_numel(os));
  std::vector<size_t> offs;
  std::vector<TensorImpl*> ps;
  size_t off = 0;
  for (const auto& t : xs) {
    offs.push_back(off);
    ps.push_back(t.impl());
    size_t chunk = static_cast<size_t>(t.shape()[axis]) * inner;
    const float* src = t.ptr();
    for (size_t o = 0; o < outer; ++o)
      std::copy(src + o * chunk, src + (o + 1) * chunk, out.begin() + (o * total + off) * inner);
    off += t.shape()[axis];
  }
  std::vector<size_t> lens;
  for (const auto& t : xs) lens.push_back(static_cast<size_t>(t.shape()[axis]));
  return make_out(os, std::move(out), xs, "concat", [ps, offs, lens, outer, inner, total](TensorImpl& self) {
    for (size_t k = 0; k < ps.size(); ++k) {
      float* g = gradp(ps[k]);
      if (!g) continue;
      size_t chunk = lens[k] * inner;
      for (size_t o = 0; o < outer; ++o) {
        const float* src = self.grad.data() + (o * total + offs[k]) * inner;
        float* dst = g + o * chunk;
        for (size_t j = 0; j < chunk; ++j) dst[j] += src[j];
      }
    }
  });
}

Tensor stack(const std::vector<Tensor>& xs) {
  std::vector<Tensor> r;
  r.reserve(xs.size());
  for (const auto& t : xs) {
    Shape s = t.shape();
    s.insert(s.begin(), 1);
    r.push_back(reshape(t, s));
  }
  return concat(r, 0);
}

Tensor narrow(const Tensor& x, int axis, int start, int length) {
  axis = norm_axis(axis, x.dim());
  if (start < 0 || length < 0 || start + length > x.shape()[axis])
    throw std::out_of_range("narrow: range out of bounds");
  AxisSplit s = split_axis(x.shape(), axis);
  Shape os = x.shape();
  os[axis] = length;
  std::vector<float> out(shape_numel(os));
  const float* X = x.ptr();
  size_t chunk = static_cast<size_t>(length) * s.inner;
  for (size_t o = 0; o < s.outer; ++o)
    std::copy(X + (o * s.dim + start) * s.inner, X + (o * s.dim + start) * s.inner + chunk, out.begin() + o * chunk);
  TensorImpl* px = x.impl();
  return make_out(os, std::move(out), {x}, "narrow", [px, s, start, chunk](TensorImpl& self) {
    float* gx = gradp(px);
    if (!gx) return;
    for (size_t o = 0; o < s.outer; ++o) {
      float* dst = gx + (o * s.dim + start) * s.inner;
      const float* src = self.grad.data() + o * chunk;
      for (size_t j = 0; j < chunk; ++j) dst[j] += src[j];
    }
  });
}

// ============================================================================
// Softmax family
// ============================================================================

Tensor softmax(const Tensor& x) {
  if (x.dim() < 1) throw std::invalid_argument("softmax: scalar");
  size_t N = x.shape().back();
  size_t rows = x.numel() / N;
  std::vector<float> out(x.numel());
  const float* X = x.ptr();
  for (size_t r = 0; r < rows; ++r) {
    const float* xr = X + r * N;
    float* yr = out.data() + r * N;
    float m = *std::max_element(xr, xr + N);
    double s = 0.0;
    for (size_t j = 0; j < N; ++j) {
      yr[j] = std::exp(xr[j] - m);
      s += yr[j];
    }
    for (size_t j = 0; j < N; ++j) yr[j] = static_cast<float>(yr[j] / s);
  }
  TensorImpl* px = x.impl();
  return make_out(x.shape(), std::move(out), {x}, "softmax", [px, rows, N](TensorImpl& self) {
    float* gx = gradp(px);
    if (!gx) return;
    for (size_t r = 0; r < rows; ++r) {
      const float* y = self.data.data() + r * N;
      const float* g = self.grad.data() + r * N;
      float dot = 0.0f;
      for (size_t j = 0; j < N; ++j) dot += g[j] * y[j];
      for (size_t j = 0; j < N; ++j) gx[r * N + j] += y[j] * (g[j] - dot);
    }
  });
}

Tensor log_softmax(const Tensor& x) {
  if (x.dim() < 1) throw std::invalid_argument("log_softmax: scalar");
  size_t N = x.shape().back();
  size_t rows = x.numel() / N;
  std::vector<float> out(x.numel());
  const float* X = x.ptr();
  for (size_t r = 0; r < rows; ++r) {
    const float* xr = X + r * N;
    float* yr = out.data() + r * N;
    float m = *std::max_element(xr, xr + N);
    double s = 0.0;
    for (size_t j = 0; j < N; ++j) s += std::exp(xr[j] - m);
    float lse = m + static_cast<float>(std::log(s));
    for (size_t j = 0; j < N; ++j) yr[j] = xr[j] - lse;
  }
  TensorImpl* px = x.impl();
  return make_out(x.shape(), std::move(out), {x}, "log_softmax", [px, rows, N](TensorImpl& self) {
    float* gx = gradp(px);
    if (!gx) return;
    for (size_t r = 0; r < rows; ++r) {
      const float* y = self.data.data() + r * N;
      const float* g = self.grad.data() + r * N;
      float gs = 0.0f;
      for (size_t j = 0; j < N; ++j) gs += g[j];
      for (size_t j = 0; j < N; ++j) gx[r * N + j] += g[j] - std::exp(y[j]) * gs;
    }
  });
}

Tensor gather(const Tensor& x, const std::vector<int>& index) {
  if (x.dim() != 2) throw std::invalid_argument("gather: expects [B,N]");
  int B = x.shape()[0], N = x.shape()[1];
  if (static_cast<int>(index.size()) != B) throw std::invalid_argument("gather: index size mismatch");
  std::vector<float> out(B);
  for (int b = 0; b < B; ++b) {
    if (index[b] < 0 || index[b] >= N) throw std::out_of_range("gather: index out of range");
    out[b] = x.data()[static_cast<size_t>(b) * N + index[b]];
  }
  TensorImpl* px = x.impl();
  return make_out({B}, std::move(out), {x}, "gather", [px, index, N](TensorImpl& self) {
    float* gx = gradp(px);
    if (!gx) return;
    for (size_t b = 0; b < index.size(); ++b) gx[b * N + index[b]] += self.grad[b];
  });
}

// ============================================================================
// Convolution / pooling
// ============================================================================

namespace {

void im2col(const float* img, int C, int H, int W, int KH, int KW, int stride, int pad, int OH, int OW, float* cols) {
  size_t P = static_cast<size_t>(OH) * OW;
  for (int c = 0; c < C; ++c)
    for (int ki = 0; ki < KH; ++ki)
      for (int kj = 0; kj < KW; ++kj) {
        float* row = cols + (static_cast<size_t>(c * KH + ki) * KW + kj) * P;
        for (int oh = 0; oh < OH; ++oh) {
          int ih = oh * stride - pad + ki;
          for (int ow = 0; ow < OW; ++ow) {
            int iw = ow * stride - pad + kj;
            row[oh * OW + ow] =
                (ih >= 0 && ih < H && iw >= 0 && iw < W) ? img[(static_cast<size_t>(c) * H + ih) * W + iw] : 0.0f;
          }
        }
      }
}

void col2im_add(const float* cols, int C, int H, int W, int KH, int KW, int stride, int pad, int OH, int OW,
                float* img) {
  size_t P = static_cast<size_t>(OH) * OW;
  for (int c = 0; c < C; ++c)
    for (int ki = 0; ki < KH; ++ki)
      for (int kj = 0; kj < KW; ++kj) {
        const float* row = cols + (static_cast<size_t>(c * KH + ki) * KW + kj) * P;
        for (int oh = 0; oh < OH; ++oh) {
          int ih = oh * stride - pad + ki;
          if (ih < 0 || ih >= H) continue;
          for (int ow = 0; ow < OW; ++ow) {
            int iw = ow * stride - pad + kj;
            if (iw < 0 || iw >= W) continue;
            img[(static_cast<size_t>(c) * H + ih) * W + iw] += row[oh * OW + ow];
          }
        }
      }
}

}  // namespace

Tensor conv2d(const Tensor& x, const Tensor& w, const Tensor& b, int stride, int pad) {
  if (x.dim() != 4 || w.dim() != 4 || x.shape()[1] != w.shape()[1])
    throw std::invalid_argument("conv2d: incompatible shapes x" + shape_str(x.shape()) + " w" + shape_str(w.shape()));
  if (stride < 1 || pad < 0) throw std::invalid_argument("conv2d: bad stride/padding");
  int B = x.shape()[0], C = x.shape()[1], H = x.shape()[2], W = x.shape()[3];
  int O = w.shape()[0], KH = w.shape()[2], KW = w.shape()[3];
  int OH = (H + 2 * pad - KH) / stride + 1, OW = (W + 2 * pad - KW) / stride + 1;
  if (OH <= 0 || OW <= 0) throw std::invalid_argument("conv2d: kernel larger than padded input");
  if (b.defined() && b.numel() != static_cast<size_t>(O)) throw std::invalid_argument("conv2d: bias size mismatch");
  int P = OH * OW, CK = C * KH * KW;
  std::vector<float> out(static_cast<size_t>(B) * O * P, 0.0f);
  std::vector<float> cols(static_cast<size_t>(CK) * P);
  for (int bi = 0; bi < B; ++bi) {
    im2col(x.ptr() + static_cast<size_t>(bi) * C * H * W, C, H, W, KH, KW, stride, pad, OH, OW, cols.data());
    float* ob = out.data() + static_cast<size_t>(bi) * O * P;
    gemm_nn(O, P, CK, w.ptr(), cols.data(), ob);
    if (b.defined())
      for (int o = 0; o < O; ++o)
        for (int p = 0; p < P; ++p) ob[static_cast<size_t>(o) * P + p] += b.ptr()[o];
  }
  TensorImpl* px = x.impl();
  TensorImpl* pw = w.impl();
  TensorImpl* pb = b.defined() ? b.impl() : nullptr;
  return make_out({B, O, OH, OW}, std::move(out), {x, w, b}, "conv2d",
                  [=](TensorImpl& self) {
                    float* gx = gradp(px);
                    float* gw = gradp(pw);
                    float* gb = gradp(pb);
                    std::vector<float> c(static_cast<size_t>(CK) * P), dc;
                    if (gx) dc.resize(c.size());
                    for (int bi = 0; bi < B; ++bi) {
                      const float* g = self.grad.data() + static_cast<size_t>(bi) * O * P;
                      if (gw) {
                        im2col(px->data.data() + static_cast<size_t>(bi) * C * H * W, C, H, W, KH, KW, stride, pad,
                               OH, OW, c.data());
                        gemm_nt(O, CK, P, g, c.data(), gw);
                      }
                      if (gx) {
                        std::fill(dc.begin(), dc.end(), 0.0f);
                        gemm_tn(CK, P, O, pw->data.data(), g, dc.data());
                        col2im_add(dc.data(), C, H, W, KH, KW, stride, pad, OH, OW,
                                   gx + static_cast<size_t>(bi) * C * H * W);
                      }
                      if (gb)
                        for (int o = 0; o < O; ++o)
                          for (int p = 0; p < P; ++p) gb[o] += g[static_cast<size_t>(o) * P + p];
                    }
                  });
}

namespace {

Tensor pool_with_argmax(const Tensor& x, int OH, int OW, const std::vector<int>& hs, const std::vector<int>& he,
                        const std::vector<int>& ws, const std::vector<int>& we, const char* op) {
  int B = x.shape()[0], C = x.shape()[1], H = x.shape()[2], W = x.shape()[3];
  std::vector<float> out(static_cast<size_t>(B) * C * OH * OW);
  std::vector<size_t> arg(out.size());
  const float* X = x.ptr();
  size_t k = 0;
  for (int bc = 0; bc < B * C; ++bc) {
    const float* plane = X + static_cast<size_t>(bc) * H * W;
    for (int i = 0; i < OH; ++i)
      for (int j = 0; j < OW; ++j, ++k) {
        size_t best = static_cast<size_t>(hs[i]) * W + ws[j];
        for (int y = hs[i]; y < he[i]; ++y)
          for (int z = ws[j]; z < we[j]; ++z) {
            size_t idx = static_cast<size_t>(y) * W + z;
            if (plane[idx] > plane[best]) best = idx;
          }
        out[k] = plane[best];
        arg[k] = static_cast<size_t>(bc) * H * W + best;
      }
  }
  TensorImpl* px = x.impl();
  return make_out({B, C, OH, OW}, std::move(out), {x}, op, [px, arg = std::move(arg)](TensorImpl& self) {
    float* gx = gradp(px);
    if (!gx) return;
    for (size_t i = 0; i < arg.size(); ++i) gx[arg[i]] += self.grad[i];
  });
}

}  // namespace

Tensor max_pool2d(const Tensor& x, int kernel, int stride) {
  if (x.dim() != 4) throw std::invalid_argument("max_pool2d: expects NCHW");
  int H = x.shape()[2], W = x.shape()[3];
  if (kernel < 1 || stride < 1 || kernel > H || kernel > W) throw std::invalid_argument("max_pool2d: bad kernel");
  int OH = (H - kernel) / stride + 1, OW = (W - kernel) / stride + 1;
  std::vector<int> hs(OH), he(OH), ws(OW), we(OW);
  for (int i = 0; i < OH; ++i) hs[i] = i * stride, he[i] = i * stride + kernel;
  for (int j = 0; j < OW; ++j) ws[j] = j * stride, we[j] = j * stride + kernel;
  return pool_with_argmax(x, OH, OW, hs, he, ws, we, "max_pool2d");
}

Tensor adaptive_max_pool2d(const Tensor& x, int out_h, int out_w) {
  if (x.dim() != 4) throw std::invalid_argument("adaptive_max_pool2d: expects NCHW");
  int H = x.shape()[2], W = x.shape()[3];
  if (out_h < 1 || out_w < 1 || out_h > H || out_w > W)
    throw std::invalid_argument("adaptive_max_pool2d: output larger than input");
  if (out_h == H && out_w == W) return x;
  std::vector<int> hs(out_h), he(out_h), ws(out_w), we(out_w);
  for (int i = 0; i < out_h; ++i) {
    hs[i] = (i * H) / out_h;
    he[i] = ((i + 1) * H + out_h - 1) / out_h;
  }
  for (int j = 0; j < out_w; ++j) {
    ws[j] = (j * W) / out_w;
    we[j] = ((j + 1) * W + out_w - 1) / out_w;
  }
  return pool_with_argmax(x, out_h, out_w, hs, he, ws, we, "adaptive_max_pool2d");
}

// ============================================================================
// Dropout / losses / helpers
// ============================================================================

Tensor dropout(const Tensor& x, float p, Rng& rng, bool training) {
  if (!training || p <= 0.0f) return x;
  if (p >= 1.0f) return mul(x, 0.0f);
  float scale = 1.0f / (1.0f - p);
  std::vector<float> mask(x.numel());
  for (auto& m : mask) m = rng.uniform() < p ? 0.0f : scale;
  std::vector<float> out(x.numel());
  for (size_t i = 0; i < out.size(); ++i) out[i] = x.data()[i] * mask[i];
  TensorImpl* px = x.impl();
  return make_out(x.shape(), std::move(out), {x}, "dropout", [px, mask = std::move(mask)](TensorImpl& self) {
    float* gx = gradp(px);
    if (!gx) return;
    for (size_t i = 0; i < mask.size(); ++i) gx[i] += self.grad[i] * mask[i];
  });
}

Tensor mse_loss(const Tensor& pred, const Tensor& target) { return mean(square(sub(pred, target))); }

Tensor huber_loss(const Tensor& pred, const Tensor& target, float delta) {
  Tensor d = sub(pred, target);
  Tensor h = unary_op(
      d, "huber",
      [delta](float v) {
        float a = std::fabs(v);
        return a <= delta ? 0.5f * v * v : delta * (a - 0.5f * delta);
      },
      [delta](float v, float) { return std::min(delta, std::max(-delta, v)); });
  return mean(h);
}

Tensor cross_entropy(const Tensor& logits, const std::vector<int>& targets) {
  return neg(mean(gather(log_softmax(logits), targets)));
}

std::vector<int> argmax_rows(const Tensor& x) {
  if (x.dim() != 2) throw std::invalid_argument("argmax_rows: expects [B,N]");
  int B = x.shape()[0], N = x.shape()[1];
  std::vector<int> r(B);
  for (int b = 0; b < B; ++b) {
    const float* row = x.ptr() + static_cast<size_t>(b) * N;
    r[b] = static_cast<int>(std::max_element(row, row + N) - row);
  }
  return r;
}

Tensor one_hot(const std::vector<int>& index, int n) {
  Tensor t = Tensor::zeros({static_cast<int>(index.size()), n});
  for (size_t i = 0; i < index.size(); ++i) {
    if (index[i] < 0 || index[i] >= n) throw std::out_of_range("one_hot: index out of range");
    t.data()[i * n + index[i]] = 1.0f;
  }
  return t;
}

float grad_norm(const std::vector<Tensor>& params) {
  double s = 0.0;
  for (const auto& p : params)
    if (p.has_grad())
      for (float g : p.grad_data()) s += static_cast<double>(g) * g;
  return static_cast<float>(std::sqrt(s));
}

float clip_grad_norm(const std::vector<Tensor>& params, float max_norm) {
  float n = grad_norm(params);
  if (max_norm > 0.0f && n > max_norm) {
    float scale = max_norm / (n + 1e-6f);
    for (auto p : params)
      if (p.has_grad())
        for (auto& g : p.grad_data()) g *= scale;
  }
  return n;
}

}  // namespace cdn
