#ifndef HAIL_MARY_BRAIN_HPP
#define HAIL_MARY_BRAIN_HPP

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <random>
#include <vector>

namespace hail_mary {

// Voices a wolf can call in, which every wolf hears, packmates apart from strangers. The noise of
// biting and feeding is on channel 0 as well, whether the wolf means it or not.
constexpr int CHANNELS = 3;

// See Continent::think() for what every one of these is.
constexpr int SELF_INPUTS = 11;
constexpr int SIGHT_INPUTS = 21;
constexpr int SOUND_INPUTS = 3; // How loud, and which way it is coming from, ahead and to the left.
constexpr int SOUNDS = 2 * CHANNELS; // Each channel from packmates, then from strangers.

constexpr int INPUTS = SELF_INPUTS + SIGHT_INPUTS + SOUNDS * SOUND_INPUTS;

constexpr int OUT_SPEED = 0;
constexpr int OUT_TURN = 1;
constexpr int OUT_BITE = 2;
constexpr int OUT_CALL = 3; // And one more for each channel after it.

constexpr int OUTPUTS = OUT_CALL + CHANNELS;

constexpr int HIDDEN = 24;

// Where each input is. See Continent::think() for what they mean.
constexpr int IN_SPEED = 0;
constexpr int IN_SIDEWAYS = 1;
constexpr int IN_TURN = 2;
constexpr int IN_HEALTH = 3;
constexpr int IN_FOOD = 4;
constexpr int IN_PAIN = 5;
constexpr int IN_AT_BISON = 6;
constexpr int IN_AT_HARE = 7;
constexpr int IN_AT_STRANGER = 8;
constexpr int IN_EATING = 9;
constexpr int IN_CONSTANT = 10;

// Each thing seen takes three inputs: how close it is, and which way it is, as how far ahead
// and how far to the left, on a circle of 1. All three are 0 when there is nothing to see.
constexpr int SEEN_CLOSE = 0;
constexpr int SEEN_AHEAD = 1;
constexpr int SEEN_LEFT = 2;

constexpr int IN_HARE = SELF_INPUTS;
constexpr int IN_BISON = IN_HARE + 3;
constexpr int IN_BISON_FACING = IN_BISON + 3;
constexpr int IN_BISON_WEAK = IN_BISON_FACING + 1;
constexpr int IN_CARCASS = IN_BISON_WEAK + 1;
constexpr int IN_MATE = IN_CARCASS + 3;
constexpr int IN_SECOND_MATE = IN_MATE + 3;
constexpr int IN_STRANGER = IN_SECOND_MATE + 3;
constexpr int IN_MATES_SEEN = IN_STRANGER + 3;

constexpr int IN_SOUNDS = IN_MATES_SEEN + 1;

static_assert(IN_SOUNDS == SELF_INPUTS + SIGHT_INPUTS, "The sight inputs don't add up");

using Inputs = std::array<float, INPUTS>;
using Outputs = std::array<float, OUTPUTS>;

// Where everything is in Genome::weights.
constexpr std::size_t GENE_INPUT = 0; // [HIDDEN][INPUTS]
constexpr std::size_t GENE_RECURRENT = GENE_INPUT + HIDDEN * INPUTS; // [HIDDEN][HIDDEN]
constexpr std::size_t GENE_HIDDEN_BIAS = GENE_RECURRENT + HIDDEN * HIDDEN; // [HIDDEN]
constexpr std::size_t GENE_LEAK = GENE_HIDDEN_BIAS + HIDDEN; // [HIDDEN], before squashing
constexpr std::size_t GENE_OUTPUT = GENE_LEAK + HIDDEN; // [OUTPUTS][HIDDEN + INPUTS]
constexpr std::size_t GENE_OUTPUT_BIAS = GENE_OUTPUT + OUTPUTS * (HIDDEN + INPUTS); // [OUTPUTS]
constexpr std::size_t GENES = GENE_OUTPUT_BIAS + OUTPUTS;

// Everything a wolf's brain is, which is all handed down. Every weight is a gene, and changes only
// by mutation, so evolution shapes what the network does directly, not just how it learns.
//
// That is the big change from projects/continents. There, weights only ever moved by each agent's
// own actor-critic, over a life of minutes, and the bite never learned to go with what was in
// reach, because contact was 1-2% of the time and its reward was lost in the noise of everything
// else. Selection here judges a whole life at once, by how many children it ends up as, which
// takes in every rare moment that mattered without needing to pick it out of the noise.
struct Genome {
    std::vector<float> weights;

    // How far each weight moves when it mutates. It is handed down and mutates as well, the way
    // evolution strategies tune their own step size.
    float mutation_size = .03f;

    int generation = 0;

    // Squashed from GENE_LEAK, for speed: how much of each hidden unit is replaced every step,
    // from .05, which remembers for about 2 seconds, to 1, which doesn't remember at all.
    std::vector<float> leak;

    // Small random weights, scaled to how many inputs each unit has.
    static Genome random(std::mt19937_64& rng);

    // A plain predator, which is where pack hunters come from: it runs at any hare or bison it
    // sees, and bites whatever it catches, goes to carcasses and eats, and otherwise ambles along. It takes no notice of other wolves, or of which way a bison is facing,
    // and never calls. All of that is wired straight from the inputs to the outputs, and the memory
    // starts out small and random, so that evolution has something to build on, but nothing about
    // hunting together or calling to start with.
    static Genome ancestor(std::mt19937_64& rng);

    // A copy, with each weight moved with this chance, by a normal amount of mutation_size, which
    // first mutates itself. Moving every weight a little works better than moving a few a lot: for
    // the same amount of disruption, it moves each weight that selection cares about further.
    Genome mutated(double chance, std::mt19937_64& rng) const;

private:
    void squash();
};

// One wolf's brain. The genome is the whole pack's, and only the memory is the wolf's own.
//
// One layer of leaky recurrent units, which can hold on to things for a few seconds, and the
// outputs, which see the units and the inputs both. Everything comes out through tanh.
class Brain {
public:
    explicit Brain(std::shared_ptr<const Genome> genome) : genes(std::move(genome)) {}

    void think(const Inputs& in, Outputs& out);

    const std::shared_ptr<const Genome>& genome() const { return genes; }

private:
    std::shared_ptr<const Genome> genes;
    std::array<float, HIDDEN> memory{};
};

}

#endif // HAIL_MARY_BRAIN_HPP
