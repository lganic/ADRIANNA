#ifndef SIM_MODELS_HPP
#define SIM_MODELS_HPP

// C++ port of tests/test_3/models.py, on top of the CDN tensors.
//
// The actor takes in a state, and spits out a distribution for each action (mean, std).
// The critic estimates the value of the current and next states, and uses TD to get more accurate.
//
// State start
// This happens if we have previous state score from critic
// | Apply error reduction to critic.
// | Use the TD score to adjust actor weights
// Pass state to actor
// Actor makes move
// Update state
// Critic scores new state
// Repeat back to start
//
// State vectors are column vectors, with shape [n, 1]. Nothing here sets requires_grad, because
// the gradients are worked out by hand, so CDN skips building an autograd graph for any of this.

#include <codeepneat/tensor.hpp>

#include <cstdint>

namespace sim {

// Parent class, since the two models look very similar architecturally.
class Model {
public:
    Model(int num_inputs, int num_outputs, cdn::Rng& rng);

    cdn::Tensor get(const cdn::Tensor& input_vector) const;

protected:
    // Move the weights along a gradient: data += scale * gradient
    void update(const cdn::Tensor& gradient, float scale);

    cdn::Tensor data; // [num_outputs, num_inputs]
};

class Actor : public Model {
public:
    Actor(int num_inputs, int num_outputs, cdn::Rng& rng, float alpha = 2e-2f);

    // Sample an action for this state. Returns a [num_outputs, 1] vector.
    cdn::Tensor run(const cdn::Tensor& current_state);

    void learn(float td_error);

private:
    int num_actions;
    float alpha;
    cdn::Rng& rng;

    cdn::Tensor last_gradient; // Undefined when there is nothing left to learn from.
};

class Critic : public Model {
public:
    Critic(int num_inputs, cdn::Rng& rng, float alpha = 0.05f, float gamma = 0.98f);

    // Score the new state, learn from it, and return the TD error.
    float run(float current_score, const cdn::Tensor& current_state);

private:
    cdn::Tensor features(const cdn::Tensor& state) const;

    float alpha;
    float gamma;

    cdn::Tensor last_state; // Undefined until the first state has been seen.
};

class Organism {
public:
    Organism(int num_inputs, int num_outputs, uint64_t seed);

    cdn::Tensor run(float current_score, const cdn::Tensor& current_state);

private:
    cdn::Rng rng; // Has to stay above the models, since they draw their starting weights from it.

    Actor actor;
    Critic critic;
};

}

#endif // SIM_MODELS_HPP
