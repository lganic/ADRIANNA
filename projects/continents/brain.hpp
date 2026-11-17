#ifndef CONTINENTS_BRAIN_HPP
#define CONTINENTS_BRAIN_HPP

#include <codeepneat/codeepneat.hpp>
#include <codeepneat/genome.hpp>
#include <codeepneat/network.hpp>
#include <codeepneat/optim.hpp>
#include <codeepneat/rng.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace continents {

constexpr int CHANNELS = 5; // That an agent can call out on, and hear others on.
constexpr int EYES = 5;

// See sense() in agent_pool.cpp for what every one of these is.
constexpr int SELF_INPUTS = 14;
constexpr int EYE_INPUTS = 5; // For each eye.
constexpr int CHANNEL_INPUTS = 3; // For each channel.

constexpr int INPUT_SIZE = SELF_INPUTS + EYES * EYE_INPUTS + CHANNELS * CHANNEL_INPUTS;

// Left thrust, right thrust, reproduction effort, attack, hold on, and then a call on each channel.
constexpr int ACT_LEFT = 0;
constexpr int ACT_RIGHT = 1;
constexpr int ACT_REPRODUCE = 2;
constexpr int ACT_ATTACK = 3;
constexpr int ACT_HOLD = 4;
constexpr int ACT_CALL = 5;

constexpr int ACTION_SIZE = ACT_CALL + CHANNELS;

using State = std::array<float, INPUT_SIZE>;
using Action = std::array<float, ACTION_SIZE>;

// What came of the last action.
struct Outcome {
    float score = 0;
    float bite_score = 0; // The part of the score that came from biting.
    bool could_bite = false; // Whether there was anything in reach to bite.
};

// How the brain learns, which can change as it goes. See Brain::learn() for all of these.
struct Learning {
    float lambda = 1; // How much the rewards after an action count towards judging it.
    float bite_noise = 1; // The most the log of how far the bite is picked from its mean can be.
    float bite_focus = 0; // How much the bite is judged by just what it got straight away, when it could bite.
};

// Everything that gets handed down: how the network is put together, and what it had learned by then.
struct Heritage {
    cdn::AssembledGenome genome;
    std::vector<cdn::NamedTensor> weights;
};

// The CoDeepNEAT side of things, the same as projects/island_evolution, except that layers can be
// LSTMs as well as dense ones, and how wide the eyes look evolves along with the learning rate.
//
// Each agent carries a whole network's genotype on its own: a blueprint, and one module for each
// module species that it points to. Every child is its parent's genotype put through the same
// mutations that CoDeepNEAT uses, with nothing in the way of crossover, culling, or species quotas.
// Who lives long enough to have children is all the selection there is.
class Genetics {
public:
    Genetics();

    const cdn::IOSpec& io() const { return io_spec; }

    // An LSTM, and then a dense layer that sees both the LSTM and the inputs, the same as
    // projects/memory_chase, only smaller. The inputs also go straight to the heads.
    cdn::AssembledGenome founder(cdn::Rng& rng);

    // A mutated copy of the parent. rate_scale multiplies the chance of every mutation.
    cdn::AssembledGenome offspring(const cdn::AssembledGenome& parent, double rate_scale, cdn::Rng& rng);

    int species_made() const { return next_species; }

private:
    cdn::Genome new_module(int species, cdn::Rng& rng);

    cdn::IOSpec io_spec;

    cdn::HyperparamSpace layer_space;
    cdn::HyperparamSpace blueprint_space;
    cdn::HyperparamSpace global_space;

    cdn::MutationParams module_mutation;
    cdn::MutationParams blueprint_mutation;

    cdn::InnovationTracker module_innovations;
    cdn::InnovationTracker blueprint_innovations;

    int next_species = 0;
};

// One agent's network, and the actor-critic that trains it as it goes, from nothing but what
// happens to it.
//
// The network has memory, so it can't be run over a stretch as one batch the way
// projects/island_evolution does. Keeping every step's graph around until the end of the stretch,
// the way projects/memory_chase does, would take too much memory with hundreds of agents. So each
// step runs without a graph, and the brain only keeps what went in, what it did, and what its
// memory was at the start of the stretch. At the end, it runs the whole stretch through again from
// that memory, which comes out exactly the same since the weights only change between stretches,
// and learns from that. Gradients only reach back one stretch, but the memory carries on across them.
class Brain {
public:
    // Inherited is whatever the parent had learned. Anything still the same shape gets copied over
    // as it is. A layer that changed size keeps what overlaps, with any new inputs starting off at
    // zero, so that growing doesn't change what the network does. Everything else starts out fresh.
    Brain(cdn::AssembledGenome genome, const cdn::IOSpec& io, uint64_t seed, const std::vector<cdn::NamedTensor>* inherited);

    // Takes what came of the last action, and the new state, and returns the next action.
    Action run(const Outcome& outcome, const State& state, const Learning& learning);

    // Whether the next run() finishes a stretch and learns from it, which makes it a lot slower than usual.
    bool learns_next() const;

    Heritage heritage() const;

    const cdn::AssembledGenome& genome() const { return genotype; }
    const cdn::Network& network() const { return *net; }

    std::size_t parameter_count() const { return parameters; }
    int layer_count() const { return layers; }
    int memory_count() const { return memories; } // How many of the layers are LSTMs.
    float learning_rate() const { return rate; }
    double eye_spread() const { return spread; } // Radians between the middle eye and the outer ones.

    // What the last action was picked around, and how far from it, for each output.
    const Action& mean() const { return last_mean; }
    const Action& noise() const { return last_spread; }

private:
    void inherit(const std::vector<cdn::NamedTensor>& inherited);
    void learn(float bootstrap);

    cdn::Tensor input(const float* state) const;

    cdn::AssembledGenome genotype;

    cdn::Rng rng;

    std::unique_ptr<cdn::Network> net;
    std::unique_ptr<cdn::Optimizer> optimizer;

    int mean_head;
    int log_std_head;
    int value_head;

    float rate;
    float entropy_weight;
    double spread;

    std::size_t parameters;
    int layers;
    int memories;

    bool acted = false; // Whether the score coming in is for an action of ours yet.

    // What the stretch so far was picked and is to be judged with, from the last Learning.
    float lambda = 1;
    float bite_cap;
    float bite_focus = 0;

    Action last_mean{};
    Action last_spread{};

    // The stretch so far, one step after another, and the memory from before the first of them.
    std::vector<cdn::NamedTensor> stretch_memory;
    std::vector<float> states;
    std::vector<float> actions;
    std::vector<float> rewards;
    std::vector<float> bite_rewards; // The part of each reward that came from biting, if it could bite, or else 0.
};

}

#endif // CONTINENTS_BRAIN_HPP
