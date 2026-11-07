// SGD, Adam and RMSProp, plus factories that read global hyperparameters.

#include "codeepneat/optim.hpp"

#include <cmath>
#include <stdexcept>

namespace cdn {

Optimizer::Optimizer(std::vector<NamedTensor> params, float lr) : params_(std::move(params)), lr_(lr) {
  states_.resize(params_.size());
}

void Optimizer::zero_grad() {
  for (auto& p : params_) p.tensor.zero_grad();
}

const ParamState* Optimizer::state(const std::string& param_name) const {
  for (size_t i = 0; i < params_.size(); ++i)
    if (params_[i].name == param_name) return &states_[i];
  return nullptr;
}

bool Optimizer::active(size_t i) const {
  const Tensor& t = params_[i].tensor;
  return t.requires_grad() && t.has_grad();
}

SGD::SGD(std::vector<NamedTensor> params, float lr, float momentum, bool nesterov)
    : Optimizer(std::move(params), lr), momentum_(momentum), nesterov_(nesterov) {}

void SGD::step() {
  ++steps_;
  for (size_t i = 0; i < params_.size(); ++i) {
    if (!active(i)) continue;
    Tensor& p = params_[i].tensor;
    auto& w = p.data();
    const auto& g = p.grad_data();
    auto& st = states_[i];
    ++st.steps;
    if (momentum_ > 0.0f && st.m.size() != w.size()) st.m.assign(w.size(), 0.0f);
    for (size_t k = 0; k < w.size(); ++k) {
      float gk = g[k] + weight_decay_ * w[k];
      float upd = gk;
      if (momentum_ > 0.0f) {
        st.m[k] = momentum_ * st.m[k] + gk;
        upd = nesterov_ ? gk + momentum_ * st.m[k] : st.m[k];
      }
      w[k] -= lr_ * upd;
    }
  }
}

Adam::Adam(std::vector<NamedTensor> params, float lr, float beta1, float beta2, float eps)
    : Optimizer(std::move(params), lr), b1_(beta1), b2_(beta2), eps_(eps) {}

void Adam::step() {
  ++steps_;
  for (size_t i = 0; i < params_.size(); ++i) {
    if (!active(i)) continue;
    Tensor& p = params_[i].tensor;
    auto& w = p.data();
    const auto& g = p.grad_data();
    auto& st = states_[i];
    if (st.m.size() != w.size()) {
      st.m.assign(w.size(), 0.0f);
      st.v.assign(w.size(), 0.0f);
    }
    ++st.steps;
    float bc1 = 1.0f - std::pow(b1_, static_cast<float>(st.steps));
    float bc2 = 1.0f - std::pow(b2_, static_cast<float>(st.steps));
    for (size_t k = 0; k < w.size(); ++k) {
      float gk = g[k] + weight_decay_ * w[k];
      st.m[k] = b1_ * st.m[k] + (1.0f - b1_) * gk;
      st.v[k] = b2_ * st.v[k] + (1.0f - b2_) * gk * gk;
      float mh = st.m[k] / bc1, vh = st.v[k] / bc2;
      w[k] -= lr_ * mh / (std::sqrt(vh) + eps_);
    }
  }
}

RMSProp::RMSProp(std::vector<NamedTensor> params, float lr, float alpha, float eps, float momentum)
    : Optimizer(std::move(params), lr), alpha_(alpha), eps_(eps), momentum_(momentum) {}

void RMSProp::step() {
  ++steps_;
  for (size_t i = 0; i < params_.size(); ++i) {
    if (!active(i)) continue;
    Tensor& p = params_[i].tensor;
    auto& w = p.data();
    const auto& g = p.grad_data();
    auto& st = states_[i];
    if (st.v.size() != w.size()) st.v.assign(w.size(), 0.0f);
    if (momentum_ > 0.0f && st.m.size() != w.size()) st.m.assign(w.size(), 0.0f);
    ++st.steps;
    for (size_t k = 0; k < w.size(); ++k) {
      float gk = g[k] + weight_decay_ * w[k];
      st.v[k] = alpha_ * st.v[k] + (1.0f - alpha_) * gk * gk;
      float upd = gk / (std::sqrt(st.v[k]) + eps_);
      if (momentum_ > 0.0f) {
        st.m[k] = momentum_ * st.m[k] + upd;
        upd = st.m[k];
      }
      w[k] -= lr_ * upd;
    }
  }
}

std::unique_ptr<Optimizer> make_optimizer(const std::string& type, std::vector<NamedTensor> params, float lr,
                                          float momentum, bool nesterov) {
  if (type == "sgd") return std::unique_ptr<Optimizer>(new SGD(std::move(params), lr, momentum, nesterov));
  if (type == "adam") return std::unique_ptr<Optimizer>(new Adam(std::move(params), lr));
  if (type == "rmsprop") return std::unique_ptr<Optimizer>(new RMSProp(std::move(params), lr));
  throw std::invalid_argument("unknown optimizer '" + type + "'");
}

std::unique_ptr<Optimizer> make_optimizer_from_globals(const HyperparamValues& g, std::vector<NamedTensor> params,
                                                       const std::string& default_type, float default_lr) {
  std::string type = g.s("optimizer", default_type);
  float lr = static_cast<float>(g.f("learning_rate", default_lr));
  float momentum = static_cast<float>(g.f("momentum", 0.9));
  bool nesterov = g.b("nesterov", false);
  auto opt = make_optimizer(type, std::move(params), lr, momentum, nesterov);
  opt->set_weight_decay(static_cast<float>(g.f("weight_decay", 0.0)));
  return opt;
}

}  // namespace cdn
