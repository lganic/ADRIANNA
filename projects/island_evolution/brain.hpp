#ifndef ISLAND_EVOLUTION_BRAIN_HPP
#define ISLAND_EVOLUTION_BRAIN_HPP

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

namespace island_evolution {

constexpr int INPUT_SIZE = 14;
constexpr int ACTION_SIZE = 3; // Left thrust, right thrust, reproduction effort.

using State = std::array<float, INPUT_SIZE>;
using Action = std::array<float, ACTION_SIZE>;

// Everything that gets handed down: how the network is put together, and what it had learned by then.
struct Heritage {
    cdn::AssembledGenome genome;
    std::vector<cdn::NamedTensor> weights;
};

// The CoDeepNEAT side of things, without the generations. Each agent carries a whole network's
// genotype on its own: a blueprint, and one module for each module species that it points to. Every
// child is its parent's genotype put through the same mutations that CoDeepNEAT uses, with nothing
// in the way of crossover, culling, or species quotas. Who lives long enough to have children is
// all the selection there is.
//
// The innovation numbers are shared by everybody, the same as one CoDeepNEAT population, so that the
// same structure turning up in two families lines up the same way.
class Genetics {
public:
    Genetics();

    const cdn::IOSpec& io() const { return io_spec; }

    // The simplest network there is, apart from its inputs going straight to the heads as well. One
    // module, which is one small dense layer, plus a skip around it, so that the policy starts out as
    // capable as the linear one in projects/island_food, with a little extra on the side.
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

// One agent's network, and the actor-critic that trains it as it goes.
//
// This works like projects/memory_chase, except that each agent learns on its own, from nothing but
// what happens to it. It collects a short stretch of steps, and then runs the whole stretch through
// the network again as one batch to learn from it. There is no memory in these networks, and the
// weights only change between stretches, so that batch gives exactly what the steps got at the time.
class Brain {
public:
    // Inherited is whatever the parent had learned. Anything still the same shape gets copied over
    // as it is. A layer that changed size keeps what overlaps, with any new inputs starting off at
    // zero, so that growing doesn't change what the network does. Everything else starts out fresh.
    Brain(cdn::AssembledGenome genome, const cdn::IOSpec& io, uint64_t seed, const std::vector<cdn::NamedTensor>* inherited);

    // Takes the score for the last action, and the new state, and returns the next action.
    Action run(float score, const State& state);

    // Whether the next run() finishes a stretch and learns from it, which makes it a lot slower than usual.
    bool learns_next() const;

    Heritage heritage() const;

    const cdn::AssembledGenome& genome() const { return genotype; }
    const cdn::Network& network() const { return *net; }

    std::size_t parameter_count() const { return parameters; }
    int layer_count() const { return layers; }
    float learning_rate() const { return rate; }

private:
    void inherit(const std::vector<cdn::NamedTensor>& inherited);
    void learn(const State& next_state);

    cdn::AssembledGenome genotype;

    cdn::Rng rng;

    std::unique_ptr<cdn::Network> net;
    std::unique_ptr<cdn::Optimizer> optimizer;

    int mean_head;
    int log_std_head;
    int value_head;

    float rate;
    float entropy_weight;

    std::size_t parameters;
    int layers;

    bool acted = false; // Whether the score coming in is for an action of ours yet.

    // The stretch so far, one step after another.
    std::vector<float> states;
    std::vector<float> actions;
    std::vector<float> rewards;
};

}

#endif // ISLAND_EVOLUTION_BRAIN_HPP
