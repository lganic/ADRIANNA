#include "memory_organism.hpp"

#include <cstddef>
#include <utility>

namespace memory_chase {

constexpr int MEMORY_UNITS = 32;
constexpr int MIX_UNITS = 64;

// How many steps go in to one update, which is also how far back gradients reach.
constexpr int STRETCH_STEPS = 32;

constexpr float GAMMA = .98f;

// Faster than this learns the first few seconds sooner, but then it falls apart once the spells get long.
// At 1e-3, 4 of 6 seeds had lost track of the mood 83 sim hours in, against 1 of 6 at this rate.
constexpr float LEARNING_RATE = 3e-4f;

constexpr float VALUE_WEIGHT = .5f;
constexpr float ENTROPY_WEIGHT = 1e-3f;
constexpr float MAX_GRAD_NORM = 1;

// With GAMMA above, values come out about 50 times bigger than scores, which swamps everything else.
constexpr float SCORE_SCALE = .1f;

constexpr float MIN_LOG_STD = -2;
constexpr float MAX_LOG_STD = 1;

static cdn::NetworkGraph build_graph(int num_inputs, int num_outputs)
{
    cdn::HyperparamValues memory;
    memory.set_int("units", MEMORY_UNITS);

    cdn::HyperparamValues mix;
    mix.set_int("units", MIX_UNITS);
    mix.set_choice("activation", "tanh");

    cdn::NetworkGraph graph;

    const int input = graph.add_input("state", {num_inputs});
    const int lstm = graph.add_layer("memory", "lstm", memory, {input});

    // The memory has to be able to flip what the state means: towards the target when calm, and
    // away from it when angry. That needs a nonlinear layer that gets to see both of them.
    const int body = graph.add_layer("mix", "dense", mix, {input, lstm});

    // Small starting weights, so that it starts out with no opinion on what to do.
    graph.add_output("mean", body, num_outputs, "linear", .1f);
    graph.add_output("log_std", body, num_outputs, "linear", .1f);
    graph.add_output("value", body, 1);

    return graph;
}

MemoryOrganism::MemoryOrganism(int num_inputs, int num_outputs, int num_worlds, uint64_t seed)
    : num_worlds(num_worlds),
      rng(seed),
      net(build_graph(num_inputs, num_outputs), cdn::BuildOptions(), seed),
      optimizer(cdn::make_optimizer("adam", net.named_parameters(), LEARNING_RATE)),
      mean_head(net.output_index("mean")),
      log_std_head(net.output_index("log_std")),
      value_head(net.output_index("value"))
{
    net.reset_state(num_worlds);
}

void MemoryOrganism::wipe_memory()
{
    net.reset_state(num_worlds);
}

cdn::Tensor MemoryOrganism::run(const std::vector<float>& scores, const cdn::Tensor& states)
{
    if (acted) {
        for (float score : scores) {
            rewards.push_back(score * SCORE_SCALE);
        }
    }

    if (static_cast<int>(values.size()) == STRETCH_STEPS) {
        learn(states);
    }

    const std::vector<cdn::Tensor> outputs = net.forward(std::vector<cdn::Tensor>{states});

    const cdn::Tensor& mean = outputs[mean_head];
    const cdn::Tensor log_std = cdn::clamp(outputs[log_std_head], MIN_LOG_STD, MAX_LOG_STD);
    const cdn::Tensor std_dev = cdn::exp(log_std);

    // Detached, since the actions are something that happened, rather than something to differentiate through.
    const cdn::Tensor action = (mean + std_dev * cdn::Tensor::randn(mean.shape(), rng)).detach();

    // Log probability of that action. The constant in it is left out, since it has no gradient.
    const cdn::Tensor z = (action - mean) / std_dev;

    log_probs.push_back(cdn::sum(-.5f * cdn::square(z) - log_std, 1));
    entropies.push_back(cdn::sum(log_std, 1));
    values.push_back(cdn::reshape(outputs[value_head], {num_worlds}));

    acted = true;

    return action;
}

void MemoryOrganism::learn(const cdn::Tensor& next_states)
{
    // What the state after the stretch is worth. The memory is put back afterwards, since this
    // step is about to go through the network again for real.
    std::vector<float> running(num_worlds);

    {
        cdn::NoGradGuard no_grad;

        const std::vector<cdn::NamedTensor> saved = net.get_state();
        const std::vector<cdn::Tensor> outputs = net.forward(std::vector<cdn::Tensor>{next_states});

        for (int world = 0; world < num_worlds; world++) {
            running[world] = outputs[value_head].data()[world];
        }

        net.set_state(saved);
    }

    std::vector<float> returns(rewards.size());

    for (int step = STRETCH_STEPS - 1; step >= 0; step--) {
        for (int world = 0; world < num_worlds; world++) {
            const std::size_t index = static_cast<std::size_t>(step) * num_worlds + world;

            running[world] = rewards[index] + GAMMA * running[world];
            returns[index] = running[world];
        }
    }

    const cdn::Tensor target({STRETCH_STEPS * num_worlds}, std::move(returns));

    const cdn::Tensor value = cdn::concat(values, 0);
    const cdn::Tensor advantage = target - value.detach();

    const cdn::Tensor policy_loss = -cdn::mean(cdn::concat(log_probs, 0) * advantage);
    const cdn::Tensor value_loss = cdn::mean(cdn::square(target - value));
    const cdn::Tensor entropy = cdn::mean(cdn::concat(entropies, 0));

    cdn::Tensor loss = policy_loss + value_loss * VALUE_WEIGHT - entropy * ENTROPY_WEIGHT;

    optimizer->zero_grad();
    loss.backward();
    cdn::clip_grad_norm(net.parameters(), MAX_GRAD_NORM);
    optimizer->step();

    log_probs.clear();
    values.clear();
    entropies.clear();
    rewards.clear();

    // Otherwise the next stretch would hold on to this entire graph.
    net.detach_state();
}

}
