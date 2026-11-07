# Reinforcement learning

CoDeepNEAT was designed for supervised learning, but its structure fits RL well: evolution
searches for an architecture and hyperparameters, and an RL algorithm trains each candidate. The
"fitness" is how well the policy performs after that training.

```text
CoDeepNEAT           for each generation, for each assembled network:
  |
  +-- evaluator        train the policy for a fixed budget  (e.g. A2C)
        |              measure its return
        +-- fitness
```

## A complete program

```cpp
#include <iostream>

#include "codeepneat/all.hpp"

using namespace cdn;

int main() {
  rl::EnvFactory make_env = [] { return std::unique_ptr<rl::Environment>(new rl::CartPole()); };

  CoDeepNEATConfig cfg;
  cfg.io = rl::actor_critic_io(*make_env());  // input "obs", heads "policy" and "value"
  cfg.module_node_space = spaces::dense_layers(8, 64);
  cfg.module_node_space.find("dropout")->hi = 0.0;
  cfg.global_space = spaces::rl_globals();
  cfg.modules.size = 16;
  cfg.blueprints.size = 8;
  cfg.networks_per_generation = 16;
  cfg.num_threads = 4;

  CoDeepNEAT evo(cfg);
  evo.run(
      3,
      [&](EvalContext& ctx) {
        rl::A2CConfig ac;
        ac.total_steps = 5000;
        rl::A2CTrainer trainer(ctx.net, make_env, ac, ctx.rng.next_u64());
        trainer.apply_globals(ctx.globals);  // use the evolved hyperparameters
        trainer.train();
        return rl::evaluate_policy(ctx.net, make_env, 5, /*greedy=*/true, ctx.rng.next_u64());
      },
      [](const GenerationStats& s) {
        std::cout << s.to_string() << "\n";
        return true;
      });
  return 0;
}
```

`examples/cartpole_codeepneat.cpp` extends this with saving, retraining and export.

## Environments

An environment has discrete actions and implements `rl::Environment`:

| Method | Contract |
|---|---|
| `name()` | Display name |
| `observation_shape()` | Per-sample shape: `{F}` or `{C, H, W}` |
| `num_actions()` | Number of discrete actions |
| `reset(rng)` | Starts an episode, returns the first observation, flattened row-major |
| `step(action, rng)` | Returns `StepResult{observation, reward, done}` |

Use the `rng` argument for randomness, so runs stay reproducible. After `done`, the next call is `reset`.

Evaluators run on several threads at once. Create a new environment inside the factory, and don't share
mutable state between instances.

### Built-in environments

| Class | Observation | Actions | Notes |
|---|---|---|---|
| `CartPole(max_steps = 500)` | `{4}` | 2 | +1 per step; a maximum return of `max_steps` |
| `Catch(rows = 10, cols = 5)` | `{1, rows, cols}` image | 3 (left, stay, right) | +1 or -1 when the ball lands |
| `MemoryCue(delay = 4)` | `{3}`: cue 0, cue 1, go | 2 | Needs memory; see below |

`MemoryCue` shows a cue first and a "go" signal `delay` steps later, with blank observations in between.
The action taken on seeing "go" must match the cue, for +1, otherwise -1. Other actions don't matter. A
policy without memory averages 0, which makes this a clean test for recurrence.

### Writing your own

This environment rewards moving to a target on a line:

```cpp
#include <cmath>
#include <iostream>

#include "codeepneat/all.hpp"

using namespace cdn;

class ReachTarget : public rl::Environment {
 public:
  std::string name() const override { return "ReachTarget"; }
  Shape observation_shape() const override { return {2}; }
  int num_actions() const override { return 2; }  // 0 = left, 1 = right

  std::vector<float> reset(Rng& rng) override {
    position_ = 0.0f;
    target_ = rng.bernoulli(0.5) ? 1.0f : -1.0f;
    steps_ = 0;
    return {position_, target_};
  }

  rl::StepResult step(int action, Rng&) override {
    position_ += action == 1 ? 0.1f : -0.1f;
    ++steps_;
    rl::StepResult r;
    r.observation = {position_, target_};
    r.reward = -std::fabs(target_ - position_) * 0.1f;
    r.done = steps_ >= 20;
    return r;
  }

 private:
  float position_ = 0.0f, target_ = 1.0f;
  int steps_ = 0;
};

int main() {
  rl::EnvFactory make_env = [] { return std::unique_ptr<rl::Environment>(new ReachTarget()); };
  auto proto = make_env();

  NetworkGraph g;
  HyperparamValues hp;
  hp.set_int("units", 16).set_choice("activation", "tanh");
  int in = g.add_input("obs", proto->observation_shape());
  int h = g.add_layer("h", "dense", hp, {in});
  g.add_output("head:policy", h, proto->num_actions(), "linear", 0.1f);
  g.add_output("head:value", h, 1);
  Network net(g);

  rl::A2CConfig ac;
  ac.total_steps = 8000;
  ac.learning_rate = 1e-2f;
  rl::A2CTrainer trainer(net, make_env, ac, 1);
  trainer.train();
  std::cout << "return: " << rl::evaluate_policy(net, make_env, 20, true, 2) << "\n";
  return 0;
}
```

## The A2C trainer

`rl::A2CTrainer` is a synchronous advantage actor-critic. Each update:

1. Detaches recurrent state, which truncates backpropagation at the rollout boundary.
2. Runs `num_envs` environments for `rollout_length` steps, as one batch. It samples actions from the
   policy, and resets finished environments and their state rows.
3. Estimates the value of the state after the rollout, without disturbing recurrent state.
4. Computes discounted returns `R`, cutting them at episode ends.
5. Minimises `policy_loss + value_coef * value_loss - entropy_coef * entropy`:

```text
policy_loss = -mean(log pi(a|s) * (R - V(s)))     (the advantage is a constant)
value_loss  =  mean((R - V(s))^2)
entropy     =  mean(-sum_a pi(a|s) log pi(a|s))
```

6. Clips the gradient norm and takes an optimizer step.

The network needs one input and two heads, named `policy` and `value` by default.
`rl::actor_critic_io(env, policy_init_scale = 0.1f)` creates the right `IOSpec`.

### A2CConfig

| Field | Default | Meaning | Global key |
|---|---|---|---|
| `total_steps` | 20000 | Environment steps, summed over environments | |
| `num_envs` | 8 | Parallel environments (the batch size) | |
| `rollout_length` | 16 | Steps per update, also the BPTT window | |
| `gamma` | 0.99 | Discount | `gamma` |
| `optimizer` | `adam` | `sgd`, `adam` or `rmsprop` | `optimizer` |
| `learning_rate` | 1e-3 | | `learning_rate` |
| `momentum` | 0.9 | SGD only | `momentum` |
| `nesterov` | false | SGD only | `nesterov` |
| `entropy_coef` | 0.01 | Entropy bonus weight | `entropy_coef` |
| `value_coef` | 0.5 | Value loss weight | `value_coef` |
| `max_grad_norm` | 1.0 | Gradient clip; 0 or less disables it | `max_grad_norm` |
| `policy_head`, `value_head` | `policy`, `value` | Head names | |

`apply_globals(ctx.globals)` overrides fields for which the global key exists.

Learning rates matter more than you might expect here. With a small dense network on `Catch`, a rate of
3e-3 with Adam barely left the uniform policy in 25,000 steps, while 1e-2 solved it. This is the kind of
choice evolution makes for you, as long as `learning_rate`'s range allows it.

### Training callback

`on_update` runs after clipping and before the optimizer step, so you can see or edit gradients:

```cpp
rl::A2CTrainer trainer(net, make_env, rl::A2CConfig{}, 1);
trainer.on_update = [](const rl::UpdateInfo& u, Network&) {
  if (u.update % 50 == 0) std::cout << u.steps << " return " << u.recent_return << " entropy " << u.entropy << "\n";
};
rl::TrainStats stats = trainer.train();
std::cout << stats.episode_returns.size() << " episodes, recent return " << stats.recent_return() << "\n";
```

`train()` returns all episode returns, the step count and the update count.

## Recurrent policies

Use a search space with LSTM nodes:

```cpp
cfg.module_node_space = spaces::recurrent_layers(8, 32);
cfg.modules.mutation.add_cell_skip = 0.15;
```

`A2CTrainer` and `evaluate_policy` handle the state themselves. Two points need attention:

- **`rollout_length` limits what can be learned.** Gradients don't cross rollout boundaries. If the cue and
  the decision are further apart than the rollout, the link is hard to learn. `examples/memory_lstm.cpp`
  uses `delay + 2`.
- **State is carried across rollouts and masked at episode ends.** In your own trainer, call
  `net.mask_state(keep)` when an episode finishes in a batch, and `reset_state` between episodes.

## Designing the fitness

The evaluator's return value drives everything, so it deserves some thought.

**Measure after training, in eval mode.** `evaluate_policy` does both.

**Use enough episodes.** Five episodes on a task with high variance gives a noisy fitness, and evolution then
selects lucky networks. Evaluation is usually cheap compared with training.

**Greedy or sampled?** Greedy evaluation (`greedy = true`) measures the policy's best guess. Sampled evaluation
matches how it trained, and it's a better choice if the policy depends on randomness.

**Reward learning speed if that's what you want.** To favour fast learners, use the mean return during training:

```cpp
Evaluator evaluate = [&](EvalContext& ctx) {
  rl::A2CConfig ac;
  ac.total_steps = 5000;
  rl::A2CTrainer trainer(ctx.net, make_env, ac, ctx.rng.next_u64());
  rl::TrainStats stats = trainer.train();
  double total = 0;
  for (double r : stats.episode_returns) total += r;
  return stats.episode_returns.empty() ? 0.0 : total / stats.episode_returns.size();
};
```

**Add other goals.** Fitness is just a number, so you can trade off performance against size or speed. The
[configuration guide](configuration.md#tuning-advice) has an example.

**Keep evaluators deterministic.** Draw all seeds from `ctx.rng`. Then a run depends only on `cfg.seed`.

**Expect ties and noise.** `evo.best()` is the single highest score seen, and a lucky run can hold that position.
Before trusting a champion, retrain it a few times with different seeds:

```cpp
double total = 0;
for (uint64_t seed = 1; seed <= 5; ++seed) {
  auto candidate = evo.build(evo.best(), seed);
  rl::A2CConfig ac;
  rl::A2CTrainer trainer(*candidate, make_env, ac, seed);
  trainer.apply_globals(evo.best().globals());
  trainer.train();
  total += rl::evaluate_policy(*candidate, make_env, 20, true, seed);
}
std::cout << "mean over 5 retrainings: " << total / 5 << "\n";
```

## Using your own RL algorithm

Nothing in the evolution code depends on A2C. An evaluator can do anything with `ctx.net`. Here is REINFORCE on
one environment, written directly against the network API:

```cpp
Evaluator reinforce = [&](EvalContext& ctx) {
  auto env = make_env();
  auto opt = make_optimizer_from_globals(ctx.globals, ctx.net.named_parameters());
  const Shape shape = env->observation_shape();
  double last_return = 0;

  for (int episode = 0; episode < 200; ++episode) {
    std::vector<float> obs = env->reset(ctx.rng);
    ctx.net.reset_state(1);
    std::vector<Tensor> log_probs;
    std::vector<float> rewards;
    bool done = false;
    while (!done) {
      Tensor logits = ctx.net.forward(rl::batch_observations({obs}, shape));
      Tensor lp = log_softmax(logits);
      int action = ctx.rng.categorical(exp(lp).data());
      log_probs.push_back(gather(lp, {action}));
      rl::StepResult r = env->step(action, ctx.rng);
      rewards.push_back(r.reward);
      done = r.done;
      obs = r.observation;
    }
    // Discounted return for every step.
    std::vector<float> returns(rewards.size());
    float running = 0;
    for (size_t t = rewards.size(); t-- > 0;) returns[t] = running = rewards[t] + 0.99f * running;
    Tensor R({static_cast<int>(returns.size())}, returns);
    Tensor loss = neg(mean(concat(log_probs, 0) * R));
    opt->zero_grad();
    loss.backward();
    opt->step();
    last_return = 0;
    for (float r : rewards) last_return += r;
  }
  return last_return;
};
```

`ctx.net.forward(tensor)` returns the first head, which is the policy here. To train on a single head, use an `IOSpec`
with only that head.

### Value-based methods (DQN)

Use one output head of size `num_actions`, and keep a target network as a second `Network`:

```cpp
Evaluator dqn = [&](EvalContext& ctx) {
  Network target(ctx.net.graph(), BuildOptions(), 1);
  target.load_state_dict(ctx.net.state_dict());  // repeat every N steps
  // Your replay buffer and loop go here. Compute targets under NoGradGuard, then:
  //   Tensor q = gather(ctx.net.forward(states), actions);
  //   Tensor loss = huber_loss(q, targets);
  return 0.0;
};
```

### Continuous actions

`A2CTrainer` handles discrete actions only. For continuous control, write your own trainer as above. A typical setup
has a `mean` head of action size and a `log_std` head, and uses the Gaussian log-probability built from tensor operations:

```cpp
CoDeepNEATConfig continuous;
continuous.io.inputs.push_back({"obs", {8}});
continuous.io.outputs.push_back({"mean", 2, "tanh", 0.1f});
continuous.io.outputs.push_back({"log_std", 2});
continuous.io.outputs.push_back({"value", 1});
```

### Several inputs

`A2CTrainer` also requires a single input. Networks with several inputs, such as an image plus a state vector, work with
`net.forward(std::vector<Tensor>{image, state})` in your own loop.
