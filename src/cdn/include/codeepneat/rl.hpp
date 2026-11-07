#pragma once
// Reinforcement-learning helpers for using CoDeepNEAT as an outer loop around
// gradient-based RL: an environment interface, a few small built-in
// environments, and a batched advantage actor-critic (A2C) trainer that works
// with feed-forward and recurrent (LSTM) evolved networks.
//
// Typical evaluator:
//   [](cdn::EvalContext& ctx) {
//     cdn::rl::A2CTrainer trainer(ctx.net, make_env, cfg, ctx.rng.next_u64());
//     trainer.apply_globals(ctx.globals);   // evolved lr / optimizer / coefs
//     trainer.train();
//     return cdn::rl::evaluate_policy(ctx.net, make_env, 10, true, ctx.rng.next_u64());
//   }

#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "codeepneat/codeepneat.hpp"
#include "codeepneat/network.hpp"
#include "codeepneat/optim.hpp"

namespace cdn {
namespace rl {

/// What Environment::step returns.
struct StepResult {
  std::vector<float> observation;  ///< flattened row-major; ignored by the trainer when done
  float reward = 0.0f;
  bool done = false;               ///< the episode ended; reset() is called next
};

/// Discrete-action environment. An instance is used from one thread at a time,
/// but evaluators run concurrently, so create a new instance per evaluation
/// (that is what EnvFactory is for) and don't share mutable state between them.
class Environment {
 public:
  virtual ~Environment() = default;
  virtual std::string name() const = 0;
  /// Per-sample observation shape: [F] or [C,H,W].
  virtual Shape observation_shape() const = 0;
  virtual int num_actions() const = 0;
  /// Start an episode and return the first observation.
  virtual std::vector<float> reset(Rng& rng) = 0;
  /// Apply `action`, an index in [0, num_actions()).
  virtual StepResult step(int action, Rng& rng) = 0;
};

/// Creates a fresh environment. Called once per parallel environment and per
/// evaluate_policy() call, possibly from several threads.
using EnvFactory = std::function<std::unique_ptr<Environment>()>;

/// Classic cart-pole balancing (Barto et al.); 500 step limit, reward 1 per step.
class CartPole : public Environment {
 public:
  explicit CartPole(int max_steps = 500) : max_steps_(max_steps) {}
  std::string name() const override { return "CartPole"; }
  Shape observation_shape() const override { return {4}; }
  int num_actions() const override { return 2; }
  std::vector<float> reset(Rng& rng) override;
  StepResult step(int action, Rng& rng) override;

 private:
  double x_ = 0, x_dot_ = 0, theta_ = 0, theta_dot_ = 0;
  int steps_ = 0, max_steps_;
  std::vector<float> obs() const;
};

/// "Catch": a ball falls down a rows x cols grid; move the paddle (left / stay /
/// right) to catch it. Observation is a [1, rows, cols] image. Reward +1 / -1.
class Catch : public Environment {
 public:
  Catch(int rows = 10, int cols = 5) : rows_(rows), cols_(cols) {}
  std::string name() const override { return "Catch"; }
  Shape observation_shape() const override { return {1, rows_, cols_}; }
  int num_actions() const override { return 3; }
  std::vector<float> reset(Rng& rng) override;
  StepResult step(int action, Rng& rng) override;

 private:
  int rows_, cols_, ball_r_ = 0, ball_c_ = 0, paddle_ = 0;
  std::vector<float> obs() const;
};

/// Memory task requiring recurrence: a cue (one of two) is shown in the first
/// observation, and a "go" signal arrives `delay` steps later (so there are
/// `delay - 1` blank observations in between). The action taken at the go signal
/// must match the cue (+1, else -1); earlier actions are ignored. Observations are
/// [cue is 0, cue is 1, go]. Memoryless policies average 0 reward.
class MemoryCue : public Environment {
 public:
  explicit MemoryCue(int delay = 4) : delay_(delay) {}
  std::string name() const override { return "MemoryCue"; }
  Shape observation_shape() const override { return {3}; }
  int num_actions() const override { return 2; }
  std::vector<float> reset(Rng& rng) override;
  StepResult step(int action, Rng& rng) override;

 private:
  int delay_, t_ = 0, cue_ = 0;
};

/// IOSpec for actor-critic: input "obs", heads "policy" (logits) and "value".
IOSpec actor_critic_io(const Environment& env, float policy_init_scale = 0.1f);

/// Stack observations into a [B, ...shape] tensor.
Tensor batch_observations(const std::vector<std::vector<float>>& obs, const Shape& shape);

struct A2CConfig {
  long total_steps = 20000;  // environment steps (summed over parallel envs)
  int num_envs = 8;          ///< parallel environments = batch size of forward()
  int rollout_length = 16;   ///< steps per update; also the truncated-BPTT window
  float gamma = 0.99f;       ///< discount factor
  std::string optimizer = "adam";  ///< sgd | adam | rmsprop
  float learning_rate = 1e-3f;
  float momentum = 0.9f;     ///< SGD only
  bool nesterov = false;     ///< SGD only
  float entropy_coef = 0.01f;  ///< weight of the entropy bonus
  float value_coef = 0.5f;     ///< weight of the value loss
  float max_grad_norm = 1.0f;  ///< global gradient-norm clip; <= 0 disables it
  std::string policy_head = "policy";  ///< head producing action logits
  std::string value_head = "value";    ///< head producing a state value, size 1
};

/// Snapshot passed to A2CTrainer::on_update.
struct UpdateInfo {
  long steps = 0;   ///< environment steps so far
  int update = 0;   ///< gradient updates so far, counting this one
  float policy_loss = 0, value_loss = 0, entropy = 0;
  float grad_norm = 0;  ///< global gradient norm before clipping
  double recent_return = 0;  // mean of the last (up to) 20 finished episodes
};

/// Result of A2CTrainer::train.
struct TrainStats {
  std::vector<double> episode_returns;  ///< undiscounted, in completion order
  long steps = 0;
  int updates = 0;
  /// Mean of the last (up to) 20 entries of episode_returns, 0 if there are none.
  double recent_return() const;
};

/// Synchronous advantage actor-critic for discrete actions.
///
/// Each update collects `rollout_length` steps from `num_envs` environments in
/// one batch, bootstraps the return from the value of the following state, and
/// minimises
///   -mean(log pi(a|s) * (R - V)) + value_coef * mean((R - V)^2) - entropy_coef * H
/// with the advantage treated as a constant.
///
/// Recurrent networks work unchanged: the state is detached at rollout boundaries
/// (truncated BPTT) and masked per environment when an episode ends. For memory
/// tasks make rollout_length at least as long as the dependency to be learned.
///
/// The network must have exactly one input plus the two heads named in the config.
/// The trainer leaves the network in training mode.
class A2CTrainer {
 public:
  /// `net` must outlive the trainer. `seed` drives environments and action sampling.
  A2CTrainer(Network& net, EnvFactory make_env, A2CConfig cfg, uint64_t seed);

  /// Override config fields from (evolved) global hyperparameters, if present:
  /// learning_rate, optimizer, momentum, nesterov, entropy_coef, value_coef,
  /// max_grad_norm, gamma.
  void apply_globals(const HyperparamValues& g);
  /// Run until `total_steps` is reached. Creates a new optimizer and fresh
  /// environments on each call; network weights carry over between calls.
  TrainStats train();

  /// Editable until train() is called.
  A2CConfig& config() { return cfg_; }
  /// Available after train() started; lets callers inspect optimizer buffers.
  Optimizer* optimizer() { return opt_.get(); }
  /// Called after every gradient step (after clipping, before optimizer.step()),
  /// with full access to the network's gradients.
  std::function<void(const UpdateInfo&, Network&)> on_update;

 private:
  Network& net_;
  EnvFactory make_env_;
  A2CConfig cfg_;
  Rng rng_;
  std::unique_ptr<Optimizer> opt_;
};

/// Mean undiscounted return over `episodes` (batch size 1, eval mode, no graph
/// recording). `greedy` takes the argmax action, otherwise actions are sampled.
/// `max_steps` caps each episode. The network's training mode is restored, but its
/// recurrent state is left at batch size 1.
double evaluate_policy(Network& net, const EnvFactory& make_env, int episodes, bool greedy, uint64_t seed,
                       const std::string& policy_head = "policy", int max_steps = 100000);

}  // namespace rl
}  // namespace cdn
