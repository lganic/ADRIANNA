// Built-in environments, the batched A2C trainer, and policy evaluation.

#include "codeepneat/rl.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace cdn {
namespace rl {

// ============================================================================
// Environments
// ============================================================================

std::vector<float> CartPole::obs() const {
  return {static_cast<float>(x_), static_cast<float>(x_dot_), static_cast<float>(theta_),
          static_cast<float>(theta_dot_)};
}

std::vector<float> CartPole::reset(Rng& rng) {
  x_ = rng.uniform(-0.05, 0.05);
  x_dot_ = rng.uniform(-0.05, 0.05);
  theta_ = rng.uniform(-0.05, 0.05);
  theta_dot_ = rng.uniform(-0.05, 0.05);
  steps_ = 0;
  return obs();
}

StepResult CartPole::step(int action, Rng&) {
  const double gravity = 9.8, masscart = 1.0, masspole = 0.1, total_mass = masscart + masspole, length = 0.5;
  const double polemass_length = masspole * length, force_mag = 10.0, tau = 0.02;
  double force = action == 1 ? force_mag : -force_mag;
  double ct = std::cos(theta_), st = std::sin(theta_);
  double temp = (force + polemass_length * theta_dot_ * theta_dot_ * st) / total_mass;
  double thetaacc = (gravity * st - ct * temp) / (length * (4.0 / 3.0 - masspole * ct * ct / total_mass));
  double xacc = temp - polemass_length * thetaacc * ct / total_mass;
  x_ += tau * x_dot_;
  x_dot_ += tau * xacc;
  theta_ += tau * theta_dot_;
  theta_dot_ += tau * thetaacc;
  ++steps_;
  bool fail = std::fabs(x_) > 2.4 || std::fabs(theta_) > 12.0 * 2.0 * 3.14159265358979 / 360.0;
  StepResult r;
  r.observation = obs();
  r.reward = 1.0f;
  r.done = fail || steps_ >= max_steps_;
  return r;
}

std::vector<float> Catch::obs() const {
  std::vector<float> o(static_cast<size_t>(rows_) * cols_, 0.0f);
  o[static_cast<size_t>(ball_r_) * cols_ + ball_c_] = 1.0f;
  o[static_cast<size_t>(rows_ - 1) * cols_ + paddle_] = 1.0f;
  return o;
}

std::vector<float> Catch::reset(Rng& rng) {
  ball_r_ = 0;
  ball_c_ = rng.randint(0, cols_ - 1);
  paddle_ = cols_ / 2;
  return obs();
}

StepResult Catch::step(int action, Rng&) {
  paddle_ = std::min(cols_ - 1, std::max(0, paddle_ + action - 1));
  ++ball_r_;
  StepResult r;
  r.done = ball_r_ >= rows_ - 1;
  r.reward = r.done ? (ball_c_ == paddle_ ? 1.0f : -1.0f) : 0.0f;
  r.observation = obs();
  return r;
}

std::vector<float> MemoryCue::reset(Rng& rng) {
  t_ = 0;
  cue_ = rng.randint(0, 1);
  return {cue_ == 0 ? 1.0f : 0.0f, cue_ == 1 ? 1.0f : 0.0f, 0.0f};
}

StepResult MemoryCue::step(int action, Rng&) {
  ++t_;
  StepResult r;
  if (t_ <= delay_) {
    r.observation = {0.0f, 0.0f, t_ == delay_ ? 1.0f : 0.0f};
    return r;
  }
  r.observation = {0.0f, 0.0f, 0.0f};
  r.reward = action == cue_ ? 1.0f : -1.0f;
  r.done = true;
  return r;
}

// ============================================================================
// Helpers
// ============================================================================

IOSpec actor_critic_io(const Environment& env, float policy_init_scale) {
  IOSpec io;
  io.inputs.push_back({"obs", env.observation_shape()});
  io.outputs.push_back({"policy", env.num_actions(), "linear", policy_init_scale});
  io.outputs.push_back({"value", 1, "linear", 1.0f});
  return io;
}

Tensor batch_observations(const std::vector<std::vector<float>>& obs, const Shape& shape) {
  size_t n = shape_numel(shape);
  std::vector<float> d;
  d.reserve(obs.size() * n);
  for (const auto& o : obs) {
    if (o.size() != n) throw std::invalid_argument("batch_observations: observation size mismatch");
    d.insert(d.end(), o.begin(), o.end());
  }
  Shape s = shape;
  s.insert(s.begin(), static_cast<int>(obs.size()));
  return Tensor(s, std::move(d));
}

double TrainStats::recent_return() const {
  if (episode_returns.empty()) return 0.0;
  size_t n = std::min<size_t>(20, episode_returns.size());
  double s = 0.0;
  for (size_t i = episode_returns.size() - n; i < episode_returns.size(); ++i) s += episode_returns[i];
  return s / n;
}

// ============================================================================
// A2C
// ============================================================================

A2CTrainer::A2CTrainer(Network& net, EnvFactory make_env, A2CConfig cfg, uint64_t seed)
    : net_(net), make_env_(std::move(make_env)), cfg_(std::move(cfg)), rng_(seed) {}

void A2CTrainer::apply_globals(const HyperparamValues& g) {
  cfg_.learning_rate = static_cast<float>(g.f("learning_rate", cfg_.learning_rate));
  cfg_.optimizer = g.s("optimizer", cfg_.optimizer);
  cfg_.momentum = static_cast<float>(g.f("momentum", cfg_.momentum));
  cfg_.nesterov = g.b("nesterov", cfg_.nesterov);
  cfg_.entropy_coef = static_cast<float>(g.f("entropy_coef", cfg_.entropy_coef));
  cfg_.value_coef = static_cast<float>(g.f("value_coef", cfg_.value_coef));
  cfg_.max_grad_norm = static_cast<float>(g.f("max_grad_norm", cfg_.max_grad_norm));
  cfg_.gamma = static_cast<float>(g.f("gamma", cfg_.gamma));
}

TrainStats A2CTrainer::train() {
  const int B = std::max(1, cfg_.num_envs);
  std::vector<std::unique_ptr<Environment>> envs;
  for (int b = 0; b < B; ++b) envs.push_back(make_env_());
  const Shape shape = envs[0]->observation_shape();
  if (net_.input_names().size() != 1) throw std::invalid_argument("A2CTrainer: network must have one input");
  int pi = net_.output_index(cfg_.policy_head), vi = net_.output_index(cfg_.value_head);
  if (pi < 0 || vi < 0) throw std::invalid_argument("A2CTrainer: network needs 'policy' and 'value' heads");
  opt_ = make_optimizer(cfg_.optimizer, net_.named_parameters(), cfg_.learning_rate, cfg_.momentum, cfg_.nesterov);
  std::vector<Tensor> params = net_.parameters();

  std::vector<std::vector<float>> obs(B);
  for (int b = 0; b < B; ++b) obs[b] = envs[b]->reset(rng_);
  std::vector<double> ep_ret(B, 0.0);
  net_.train(true);
  net_.reset_state(B);
  TrainStats stats;
  const int T = std::max(1, cfg_.rollout_length);

  while (stats.steps < cfg_.total_steps) {
    net_.detach_state();  // truncated BPTT at rollout boundaries
    std::vector<Tensor> logps, vals, ents;
    std::vector<std::vector<float>> rewards, notdone;
    for (int t = 0; t < T; ++t) {
      std::vector<Tensor> outs = net_.forward(std::vector<Tensor>{batch_observations(obs, shape)});
      Tensor lp = log_softmax(outs[pi]);
      int A = lp.shape()[1];
      std::vector<int> act(B);
      std::vector<float> probs(A);
      for (int b = 0; b < B; ++b) {
        for (int a = 0; a < A; ++a) probs[a] = std::exp(lp.data()[static_cast<size_t>(b) * A + a]);
        act[b] = rng_.categorical(probs);
      }
      logps.push_back(gather(lp, act));
      vals.push_back(reshape(outs[vi], {B}));
      ents.push_back(neg(sum(mul(exp(lp), lp), 1)));
      std::vector<float> r(B), nd(B);
      bool any_done = false;
      for (int b = 0; b < B; ++b) {
        StepResult res = envs[b]->step(act[b], rng_);
        r[b] = res.reward;
        ep_ret[b] += res.reward;
        nd[b] = res.done ? 0.0f : 1.0f;
        if (res.done) {
          any_done = true;
          stats.episode_returns.push_back(ep_ret[b]);
          ep_ret[b] = 0.0;
          obs[b] = envs[b]->reset(rng_);
        } else {
          obs[b] = std::move(res.observation);
        }
      }
      rewards.push_back(r);
      notdone.push_back(nd);
      if (any_done && net_.is_recurrent()) net_.mask_state(nd);
      stats.steps += B;
    }

    // Bootstrap value of the state after the rollout (state is restored so the
    // next rollout continues from the same recurrent state).
    std::vector<float> running(B);
    {
      NoGradGuard ng;
      std::vector<NamedTensor> saved = net_.get_state();
      std::vector<Tensor> outs = net_.forward(std::vector<Tensor>{batch_observations(obs, shape)});
      for (int b = 0; b < B; ++b) running[b] = outs[vi].data()[b];
      if (net_.is_recurrent()) net_.set_state(saved);
    }
    const int steps_taken = static_cast<int>(rewards.size());
    std::vector<float> R(static_cast<size_t>(steps_taken) * B);
    for (int t = steps_taken - 1; t >= 0; --t)
      for (int b = 0; b < B; ++b) {
        running[b] = rewards[t][b] + cfg_.gamma * running[b] * notdone[t][b];
        R[static_cast<size_t>(t) * B + b] = running[b];
      }
    Tensor ret({steps_taken * B}, std::move(R));
    Tensor V = concat(vals, 0);
    Tensor LP = concat(logps, 0);
    Tensor E = concat(ents, 0);
    Tensor adv = sub(ret, V.detach());
    Tensor policy_loss = neg(mean(mul(LP, adv)));
    Tensor value_loss = mean(square(sub(ret, V)));
    Tensor entropy = mean(E);
    Tensor loss = add(add(policy_loss, mul(value_loss, cfg_.value_coef)), mul(entropy, -cfg_.entropy_coef));

    opt_->zero_grad();
    loss.backward();
    float gn = clip_grad_norm(params, cfg_.max_grad_norm);
    ++stats.updates;
    if (on_update) {
      UpdateInfo info;
      info.steps = stats.steps;
      info.update = stats.updates;
      info.policy_loss = policy_loss.item();
      info.value_loss = value_loss.item();
      info.entropy = entropy.item();
      info.grad_norm = gn;
      info.recent_return = stats.recent_return();
      on_update(info, net_);
    }
    opt_->step();
  }
  return stats;
}

double evaluate_policy(Network& net, const EnvFactory& make_env, int episodes, bool greedy, uint64_t seed,
                       const std::string& policy_head, int max_steps) {
  auto env = make_env();
  Rng rng(seed);
  bool was_training = net.is_training();
  net.eval();
  NoGradGuard ng;
  int pi = net.output_index(policy_head);
  if (pi < 0) throw std::invalid_argument("evaluate_policy: no head named " + policy_head);
  const Shape shape = env->observation_shape();
  double total = 0.0;
  for (int e = 0; e < episodes; ++e) {
    std::vector<float> obs = env->reset(rng);
    net.reset_state(1);
    double ret = 0.0;
    for (int s = 0; s < max_steps; ++s) {
      std::vector<Tensor> outs = net.forward(std::vector<Tensor>{batch_observations({obs}, shape)});
      int action;
      if (greedy) {
        action = argmax_rows(outs[pi])[0];
      } else {
        Tensor p = softmax(outs[pi]);
        action = rng.categorical(p.data());
      }
      StepResult r = env->step(action, rng);
      ret += r.reward;
      if (r.done) break;
      obs = std::move(r.observation);
    }
    total += ret;
  }
  net.train(was_training);
  return episodes > 0 ? total / episodes : 0.0;
}

}  // namespace rl
}  // namespace cdn
