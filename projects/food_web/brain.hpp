#ifndef FOOD_WEB_BRAIN_HPP
#define FOOD_WEB_BRAIN_HPP

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <random>
#include <vector>

namespace food_web {

constexpr int MOST_HIDDEN = 32;

// How big a brain is. Each species has its own, since each senses and does different things.
struct Layout {
    int inputs = 0;
    int hidden = 0;
    int outputs = 0;

    // Where everything is in Genome::weights.
    std::size_t input_weights() const { return 0; } // [hidden][inputs]
    std::size_t recurrent_weights() const { return input_weights() + static_cast<std::size_t>(hidden * inputs); } // [hidden][hidden]
    std::size_t hidden_bias() const { return recurrent_weights() + static_cast<std::size_t>(hidden * hidden); } // [hidden]
    std::size_t leak() const { return hidden_bias() + static_cast<std::size_t>(hidden); } // [hidden], before squashing
    std::size_t output_weights() const { return leak() + static_cast<std::size_t>(hidden); } // [outputs][hidden + inputs]
    std::size_t output_bias() const { return output_weights() + static_cast<std::size_t>(outputs * (hidden + inputs)); } // [outputs]
    std::size_t genes() const { return output_bias() + static_cast<std::size_t>(outputs); }

    // The weight straight from an input to an output, past the hidden units.
    std::size_t direct(int output, int input) const { return output_weights() + static_cast<std::size_t>(output * (hidden + inputs) + hidden + input); }
    std::size_t bias(int output) const { return output_bias() + static_cast<std::size_t>(output); }
};

// Everything a brain is, which is all handed down. Every weight is a gene, and changes only by
// mutation, the same as projects/hail_mary, which explains why.
struct Genome {
    Layout layout;
    std::vector<float> weights;

    // How far each weight moves when it mutates, which is handed down and mutates as well.
    float mutation_size = .03f;

    int generation = 0;

    // Squashed from Layout::leak(), for speed: how much of each hidden unit is replaced every step,
    // from .05, which remembers for about 2 seconds, to 1, which doesn't remember at all.
    std::vector<float> leak;

    // Small random weights, scaled to how many inputs each unit has.
    static Genome random(const Layout& layout, std::mt19937_64& rng);

    // A copy with every weight moved by a normal amount of mutation_size, which first mutates itself.
    // See projects/hail_mary/brain.hpp for why every weight a little, rather than a few a lot.
    Genome mutated(double chance, std::mt19937_64& rng) const;

    // Shrinks every weight from the inputs, and the memory, so that wiring put in by hand straight
    // from the inputs to the outputs decides what a founder does. See the ancestors in world.cpp.
    void quieten(float factor);

    // After changing weights by hand.
    void squash();
};

// One animal's brain: a genome, which a whole wolf pack shares, and its own memory. One layer of
// leaky recurrent units, which can hold on to things for a few seconds, and outputs that see the
// units and the inputs both, through tanh.
class Brain {
public:
    explicit Brain(std::shared_ptr<const Genome> genome) : genes(std::move(genome)) {}

    void think(const float* in, float* out);

    const std::shared_ptr<const Genome>& genome() const { return genes; }

private:
    std::shared_ptr<const Genome> genes;
    std::array<float, MOST_HIDDEN> memory{};
};

}

#endif // FOOD_WEB_BRAIN_HPP
