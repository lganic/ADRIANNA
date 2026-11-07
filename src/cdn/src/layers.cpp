// Weight initialisation, built-in layers, the merge operation, and the layer registry.

#include "codeepneat/layers.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace cdn {

Tensor init_weight(const Shape& shape, int fan_in, int fan_out, const BuildOptions& opt, float scale, Rng& rng) {
  const std::string& m = opt.weight_init;
  float fi = static_cast<float>(std::max(1, fan_in)), fo = static_cast<float>(std::max(1, fan_out));
  std::vector<float> d(shape_numel(shape));
  auto uni = [&](float lim) {
    for (auto& v : d) v = static_cast<float>(rng.uniform(-lim, lim));
  };
  auto nrm = [&](float sd) {
    for (auto& v : d) v = static_cast<float>(rng.normal(0.0, sd));
  };
  if (m == "glorot_uniform" || m == "xavier_uniform")
    uni(scale * std::sqrt(6.0f / (fi + fo)));
  else if (m == "glorot_normal" || m == "xavier_normal")
    nrm(scale * std::sqrt(2.0f / (fi + fo)));
  else if (m == "he_normal")
    nrm(scale * std::sqrt(2.0f / fi));
  else if (m == "he_uniform")
    uni(scale * std::sqrt(6.0f / fi));
  else if (m == "lecun_normal")
    nrm(scale * std::sqrt(1.0f / fi));
  else if (m == "uniform")
    uni(scale * opt.uniform_range);
  else
    throw std::invalid_argument("unknown weight_init '" + m + "'");
  return Tensor(shape, std::move(d), true);
}

// ============================================================================
// Layer base
// ============================================================================

Layer::Layer(std::string name, std::string type, Shape in_shape, HyperparamValues hp)
    : name_(std::move(name)), type_(std::move(type)), in_shape_(std::move(in_shape)), hp_(std::move(hp)) {}

Tensor Layer::add_param(const std::string& local_name, Tensor t) {
  t.set_requires_grad(true);
  t.set_name(name_ + "." + local_name);
  params_.push_back({name_ + "." + local_name, t});
  return t;
}

Tensor* Layer::param(const std::string& local_name) {
  std::string full = name_ + "." + local_name;
  for (auto& p : params_)
    if (p.name == full) return &p.tensor;
  return nullptr;
}

size_t Layer::num_parameters() const {
  size_t n = 0;
  for (const auto& p : params_) n += p.tensor.numel();
  return n;
}

static void check_dropout(float p, const std::string& name) {
  if (p < 0.0f || p >= 1.0f) throw std::invalid_argument(name + ": dropout must be in [0,1)");
}

// ============================================================================
// Identity
// ============================================================================

IdentityLayer::IdentityLayer(const std::string& name, const Shape& in_shape)
    : Layer(name, "identity", in_shape, HyperparamValues()) {
  out_shape_ = in_shape;
}

Tensor IdentityLayer::forward(const Tensor& x, LayerContext&) { return x; }

// ============================================================================
// Dense
// ============================================================================

DenseLayer::DenseLayer(const std::string& name, const HyperparamValues& hp, const Shape& in_shape,
                       const BuildOptions& opt, Rng& rng)
    : Layer(name, "dense", in_shape, hp) {
  int units = hp.i("units", 32);
  if (units <= 0) throw std::invalid_argument(name + ": units must be > 0");
  int in = static_cast<int>(shape_numel(in_shape));
  act_ = hp.s("activation", "relu");
  if (!is_activation(act_)) throw std::invalid_argument(name + ": unknown activation " + act_);
  dropout_ = static_cast<float>(hp.f("dropout", 0.0));
  check_dropout(dropout_, name);
  float scale = static_cast<float>(hp.f("weight_scale", 1.0));
  weight_ = add_param("weight", init_weight({units, in}, in, units, opt, scale, rng));
  if (hp.b("use_bias", true)) bias_ = add_param("bias", Tensor::zeros({units}));
  out_shape_ = {units};
}

Tensor DenseLayer::forward(const Tensor& x, LayerContext& ctx) {
  Tensor y = linear(flatten(x), weight_, bias_);
  y = activation(y, act_);
  return dropout(y, dropout_, *ctx.rng, ctx.training);
}

// ============================================================================
// Conv2D
// ============================================================================

Conv2DLayer::Conv2DLayer(const std::string& name, const HyperparamValues& hp, const Shape& in_shape,
                         const BuildOptions& opt, Rng& rng)
    : Layer(name, "conv2d", in_shape, hp) {
  if (in_shape.size() == 3)
    img_shape_ = in_shape;
  else if (in_shape.size() == 1)
    img_shape_ = {in_shape[0], 1, 1};
  else
    throw std::invalid_argument(name + ": conv2d expects [C,H,W] or [F] input, got " + shape_str(in_shape));
  int C = img_shape_[0], H = img_shape_[1], W = img_shape_[2];
  int filters = hp.i("filters", 32);
  int k = hp.i("kernel_size", 3);
  stride_ = hp.i("stride", 1);
  if (filters <= 0 || k <= 0 || stride_ <= 0) throw std::invalid_argument(name + ": bad conv2d hyperparameters");
  pad_ = hp.has("padding") ? hp.i("padding") : k / 2;
  pool_ = hp.b("max_pool", false);
  act_ = hp.s("activation", "relu");
  if (!is_activation(act_)) throw std::invalid_argument(name + ": unknown activation " + act_);
  dropout_ = static_cast<float>(hp.f("dropout", 0.0));
  check_dropout(dropout_, name);
  float scale = static_cast<float>(hp.f("weight_scale", 1.0));
  weight_ = add_param("weight", init_weight({filters, C, k, k}, C * k * k, filters * k * k, opt, scale, rng));
  if (hp.b("use_bias", true)) bias_ = add_param("bias", Tensor::zeros({filters}));
  int OH = (H + 2 * pad_ - k) / stride_ + 1, OW = (W + 2 * pad_ - k) / stride_ + 1;
  if (OH <= 0 || OW <= 0) throw std::invalid_argument(name + ": kernel larger than input");
  if (pool_ && (OH < 2 || OW < 2)) pool_ = false;  // nothing left to pool
  if (pool_) {
    OH /= 2;
    OW /= 2;
  }
  out_shape_ = {filters, OH, OW};
}

Tensor Conv2DLayer::forward(const Tensor& x, LayerContext& ctx) {
  Tensor xi = x;
  if (x.dim() == 2) xi = reshape(x, {x.shape()[0], img_shape_[0], img_shape_[1], img_shape_[2]});
  Tensor y = conv2d(xi, weight_, bias_, stride_, pad_);
  y = activation(y, act_);
  if (pool_) y = max_pool2d(y, 2, 2);
  return dropout(y, dropout_, *ctx.rng, ctx.training);
}

// ============================================================================
// LSTM
// ============================================================================

LSTMLayer::LSTMLayer(const std::string& name, const HyperparamValues& hp, const Shape& in_shape,
                     const BuildOptions& opt, Rng& rng)
    : Layer(name, "lstm", in_shape, hp) {
  units_ = hp.i("units", 32);
  if (units_ <= 0) throw std::invalid_argument(name + ": units must be > 0");
  in_features_ = static_cast<int>(shape_numel(in_shape));
  dropout_ = static_cast<float>(hp.f("dropout", 0.0));
  check_dropout(dropout_, name);
  float scale = static_cast<float>(hp.f("weight_scale", 1.0));
  int H = units_, Z = in_features_ + units_;
  weight_ = add_param("weight", init_weight({4 * H, Z}, Z, 4 * H, opt, scale, rng));
  Tensor b = Tensor::zeros({4 * H});
  for (int j = H; j < 2 * H; ++j) b.data()[j] = opt.lstm_forget_bias;  // gate order: i, f, g, o
  bias_ = add_param("bias", b);
  out_shape_ = {units_};
  reset_state(1);
}

void LSTMLayer::add_cell_skip(LSTMLayer* source, const BuildOptions& opt, Rng& rng) {
  if (!source || source == this) throw std::invalid_argument(name_ + ": invalid cell skip source");
  int H = units_, Z = in_features_ + units_;
  std::string base = "skip" + std::to_string(skips_.size());
  Skip s;
  s.src = source;
  s.gate_w = add_param(base + ".gate_weight", init_weight({H, Z}, Z, H, opt, 1.0f, rng));
  // Start mostly closed so a new skip connection perturbs the cell gently.
  s.gate_b = add_param(base + ".gate_bias", Tensor::full({H}, -1.0f));
  if (source->units() != H)
    s.proj = add_param(base + ".proj", init_weight({H, source->units()}, source->units(), H, opt, 1.0f, rng));
  skips_.push_back(s);
}

std::vector<LSTMLayer::CellSkipInfo> LSTMLayer::cell_skips() const {
  std::vector<CellSkipInfo> r;
  for (const auto& s : skips_) r.push_back({s.src, s.gate_w, s.gate_b, s.proj});
  return r;
}

void LSTMLayer::reset_state(int batch) {
  h_ = Tensor::zeros({batch, units_});
  c_ = Tensor::zeros({batch, units_});
}

void LSTMLayer::detach_state() {
  h_ = h_.detach();
  c_ = c_.detach();
}

void LSTMLayer::mask_state(const Tensor& keep) {
  h_ = mul(h_, keep);
  c_ = mul(c_, keep);
}

std::vector<NamedTensor> LSTMLayer::state() const { return {{name_ + ".h", h_}, {name_ + ".c", c_}}; }

void LSTMLayer::set_state(const std::vector<Tensor>& s) {
  if (s.size() != 2) throw std::invalid_argument(name_ + ": set_state expects {h, c}");
  if (s[0].dim() != 2 || s[0].shape()[1] != units_ || s[1].shape() != s[0].shape())
    throw std::invalid_argument(name_ + ": state shape mismatch");
  h_ = s[0];
  c_ = s[1];
}

Tensor LSTMLayer::forward(const Tensor& x, LayerContext& ctx) {
  Tensor xf = flatten(x);
  int B = xf.shape()[0];
  if (h_.shape()[0] != B) reset_state(B);
  int H = units_;
  Tensor xh = concat({xf, h_}, 1);
  Tensor z = linear(xh, weight_, bias_);
  Tensor i = sigmoid(narrow(z, 1, 0, H));
  Tensor f = sigmoid(narrow(z, 1, H, H));
  Tensor g = tanh(narrow(z, 1, 2 * H, H));
  Tensor o = sigmoid(narrow(z, 1, 3 * H, H));
  Tensor c = add(mul(f, c_), mul(i, g));
  for (auto& s : skips_) {
    Tensor src = s.src->cell();
    if (src.shape()[0] != B) continue;  // source not yet sized for this batch
    if (s.proj.defined()) src = linear(src, s.proj, Tensor());
    Tensor k = sigmoid(linear(xh, s.gate_w, s.gate_b));
    c = add(c, mul(k, src));
  }
  Tensor h = mul(o, tanh(c));
  gates_["i"] = i;
  gates_["f"] = f;
  gates_["g"] = g;
  gates_["o"] = o;
  h_ = h;
  c_ = c;
  return dropout(h, dropout_, *ctx.rng, ctx.training);
}

// ============================================================================
// Merge
// ============================================================================

Merge::Merge(const std::string& node_name, const std::string& method, const std::vector<Shape>& in_shapes,
             const BuildOptions& opt, Rng& rng)
    : method_(method == "add" ? "sum" : method), in_(in_shapes) {
  if (in_.empty()) throw std::invalid_argument(node_name + ": merge without inputs");
  if (method_ != "concat" && method_ != "sum") throw std::invalid_argument(node_name + ": unknown merge " + method);
  proj_.resize(in_.size());
  bool any_flat = false, any_spatial = false;
  for (const auto& s : in_) {
    if (s.size() == 3)
      any_spatial = true;
    else if (s.size() == 1)
      any_flat = true;
    else
      throw std::invalid_argument(node_name + ": merge supports [F] or [C,H,W] inputs, got " + shape_str(s));
  }
  if (in_.size() == 1) {
    out_ = in_[0];
    return;
  }
  flatten_ = any_flat && any_spatial;
  spatial_ = any_spatial && !flatten_;
  auto add_proj = [&](size_t k, Tensor w) {
    w.set_requires_grad(true);
    std::string n = node_name + ".merge.proj" + std::to_string(k);
    w.set_name(n);
    params_.push_back({n, w});
    proj_[k] = w;
  };
  if (spatial_) {
    ph_ = in_[0][1];
    pw_ = in_[0][2];
    int mc = in_[0][0], sc = 0;
    for (const auto& s : in_) {
      ph_ = std::min(ph_, s[1]);
      pw_ = std::min(pw_, s[2]);
      mc = std::min(mc, s[0]);
      sc += s[0];
    }
    if (method_ == "concat") {
      out_ = {sc, ph_, pw_};
    } else {
      for (size_t k = 0; k < in_.size(); ++k)
        if (in_[k][0] != mc) add_proj(k, init_weight({mc, in_[k][0], 1, 1}, in_[k][0], mc, opt, 1.0f, rng));
      out_ = {mc, ph_, pw_};
    }
  } else {
    int mf = -1, sf = 0;
    std::vector<int> f(in_.size());
    for (size_t k = 0; k < in_.size(); ++k) {
      f[k] = static_cast<int>(shape_numel(in_[k]));
      mf = mf < 0 ? f[k] : std::min(mf, f[k]);
      sf += f[k];
    }
    if (method_ == "concat") {
      out_ = {sf};
    } else {
      for (size_t k = 0; k < in_.size(); ++k)
        if (f[k] != mf) add_proj(k, init_weight({mf, f[k]}, f[k], mf, opt, 1.0f, rng));
      out_ = {mf};
    }
  }
}

Tensor Merge::forward(const std::vector<Tensor>& xs) const {
  if (xs.size() != in_.size()) throw std::invalid_argument("merge: wrong number of inputs");
  if (xs.size() == 1) return xs[0];
  std::vector<Tensor> ys;
  ys.reserve(xs.size());
  for (size_t k = 0; k < xs.size(); ++k) {
    Tensor x = xs[k];
    if (spatial_) {
      x = adaptive_max_pool2d(x, ph_, pw_);
      if (proj_[k].defined()) x = conv2d(x, proj_[k], Tensor(), 1, 0);
    } else {
      x = flatten(x);
      if (proj_[k].defined()) x = linear(x, proj_[k], Tensor());
    }
    ys.push_back(x);
  }
  if (method_ == "concat") return concat(ys, 1);
  Tensor s = ys[0];
  for (size_t k = 1; k < ys.size(); ++k) s = add(s, ys[k]);
  return s;
}

// ============================================================================
// Registry
// ============================================================================

LayerRegistry::LayerRegistry() {
  factories_["identity"] = [](const std::string& name, const HyperparamValues&, const Shape& in, const BuildOptions&,
                              Rng&) { return std::unique_ptr<Layer>(new IdentityLayer(name, in)); };
  factories_["dense"] = [](const std::string& name, const HyperparamValues& hp, const Shape& in,
                           const BuildOptions& opt, Rng& rng) {
    return std::unique_ptr<Layer>(new DenseLayer(name, hp, in, opt, rng));
  };
  factories_["conv2d"] = [](const std::string& name, const HyperparamValues& hp, const Shape& in,
                            const BuildOptions& opt, Rng& rng) {
    return std::unique_ptr<Layer>(new Conv2DLayer(name, hp, in, opt, rng));
  };
  factories_["lstm"] = [](const std::string& name, const HyperparamValues& hp, const Shape& in,
                          const BuildOptions& opt, Rng& rng) {
    return std::unique_ptr<Layer>(new LSTMLayer(name, hp, in, opt, rng));
  };
}

LayerRegistry& LayerRegistry::instance() {
  static LayerRegistry r;
  return r;
}

void LayerRegistry::add(const std::string& type, LayerFactory factory) { factories_[type] = std::move(factory); }
bool LayerRegistry::has(const std::string& type) const { return factories_.count(type) != 0; }

std::unique_ptr<Layer> LayerRegistry::create(const std::string& type, const std::string& name,
                                             const HyperparamValues& hp, const Shape& in_shape,
                                             const BuildOptions& opt, Rng& rng) const {
  auto it = factories_.find(type);
  if (it == factories_.end()) throw std::invalid_argument("unknown layer type '" + type + "' for node " + name);
  return it->second(name, hp, in_shape, opt, rng);
}

std::vector<std::string> LayerRegistry::types() const {
  std::vector<std::string> r;
  for (const auto& kv : factories_) r.push_back(kv.first);
  return r;
}

}  // namespace cdn
