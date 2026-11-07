#include <sim/models.hpp>

#include <algorithm>
#include <cstddef>
#include <utility>
#include <vector>

namespace sim {

Model::Model(int num_inputs, int num_outputs, cdn::Rng& rng)
    : data(cdn::Tensor::randn({num_outputs, num_inputs}, rng, 0.05f)) {}

cdn::Tensor Model::get(const cdn::Tensor& input_vector) const
{
    return cdn::matmul(data, input_vector);
}

void Model::update(const cdn::Tensor& gradient, float scale)
{
    // Done in place, since this runs twice a step and skipping the temporary tensors is a good chunk faster.
    float* weights = data.ptr();
    const float* delta = gradient.ptr();

    for (std::size_t i = 0; i < data.numel(); i++) {
        weights[i] += scale * delta[i];
    }
}

// One output for mean, one for stddev.
Actor::Actor(int num_inputs, int num_outputs, cdn::Rng& rng, float alpha)
    : Model(num_inputs, num_outputs * 2, rng), num_actions(num_outputs), alpha(alpha), rng(rng) {}

cdn::Tensor Actor::run(const cdn::Tensor& current_state)
{
    const cdn::Tensor output = get(current_state);

    const cdn::Tensor mean = cdn::narrow(output, 0, 0, num_actions);
    const cdn::Tensor log_std = cdn::clamp(cdn::narrow(output, 0, num_actions, num_actions), -2.0f, 1.0f);
    const cdn::Tensor std_dev = cdn::exp(log_std);

    const cdn::Tensor action = mean + std_dev * cdn::Tensor::randn(mean.shape(), rng);

    // Differentiate log pi(action | state), holding action fixed, then scale by the
    // inverse Fisher information (natural gradient). The plain gradient has a 1 / std^2
    // term, so a small std gives huge noisy steps that shove thrusters far below zero,
    // where they are clamped and can never recover.
    const cdn::Tensor grad_mean = action - mean;
    const cdn::Tensor grad_log_std = 0.5f * (cdn::square(grad_mean / std_dev) - 1.0f);
    const cdn::Tensor output_gradient = cdn::concat({grad_mean, grad_log_std}, 0);

    // Save the gradient for this action before the reward. The column x row matmul is the outer product.
    last_gradient = cdn::matmul(output_gradient, cdn::reshape(current_state, {1, -1}));

    return action;
}

void Actor::learn(float td_error)
{
    td_error = std::min(2.0f, std::max(-2.0f, td_error));

    if (last_gradient.defined()) {
        update(last_gradient, alpha * td_error);
        last_gradient = cdn::Tensor(); // Prevent from running twice.
    }
}

// Bias, plus the state, plus the upper triangle of the state's outer product.
static int feature_count(int num_inputs)
{
    return 1 + num_inputs + num_inputs * (num_inputs + 1) / 2;
}

// Num outputs is assumed to be one, since we are just getting the discounted return here.
Critic::Critic(int num_inputs, cdn::Rng& rng, float alpha, float gamma)
    : Model(feature_count(num_inputs), 1, rng), alpha(alpha), gamma(gamma) {}

cdn::Tensor Critic::features(const cdn::Tensor& state) const
{
    // Value peaks at the target, so it depends on things like dx^2 and dx * vx. A plain
    // linear map of the state cannot express that, so give the critic the pairwise products.
    const std::vector<float>& values = state.data();
    const int num_inputs = static_cast<int>(values.size());
    const int count = feature_count(num_inputs);

    std::vector<float> result;
    result.reserve(count);

    result.push_back(1.0f);
    result.insert(result.end(), values.begin(), values.end());

    for (int i = 0; i < num_inputs; i++) {
        for (int j = i; j < num_inputs; j++) {
            result.push_back(values[i] * values[j]);
        }
    }

    return cdn::Tensor({count, 1}, std::move(result));
}

float Critic::run(float current_score, const cdn::Tensor& current_state)
{
    cdn::Tensor state = features(current_state);

    if (!last_state.defined()) {
        last_state = std::move(state);
        return 0;
    }

    // Just using TD(0) for right now.
    const float new_prediction = get(state).item();
    const float last_prediction = get(last_state).item();

    const float error = current_score + gamma * new_prediction - last_prediction;

    update(cdn::reshape(last_state, {1, -1}), alpha * error);

    last_state = std::move(state);

    return error;
}

Organism::Organism(int num_inputs, int num_outputs, uint64_t seed)
    : rng(seed), actor(num_inputs, num_outputs, rng), critic(num_inputs, rng) {}

cdn::Tensor Organism::run(float current_score, const cdn::Tensor& current_state)
{
    // First, we need to do the scoring.
    const float td_error = critic.run(current_score, current_state);

    // Then update the learning.
    actor.learn(td_error);

    // Run the actor, and return its actions.
    return actor.run(current_state);
}

}
