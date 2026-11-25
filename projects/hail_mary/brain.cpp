#include "brain.hpp"

#include <algorithm>
#include <cmath>

namespace hail_mary {

constexpr float MIN_LEAK = .05f;

constexpr float SMALLEST_MUTATION = .003f;
constexpr float BIGGEST_MUTATION = .3f;

// How much the mutation size itself mutates, in log space, per child. About 10% either way.
constexpr double MUTATION_SIZE_DRIFT = .1;

// Very rarely, a weight starts over from scratch instead of being nudged.
constexpr double RESET_SHARE = .001;

void Genome::squash()
{
    leak.resize(HIDDEN);

    for (int unit = 0; unit < HIDDEN; unit++) {
        const float raw = weights[GENE_LEAK + unit];

        leak[unit] = MIN_LEAK + (1 - MIN_LEAK) / (1 + std::exp(-raw));
    }
}

Genome Genome::random(std::mt19937_64& rng)
{
    Genome genome;

    genome.weights.assign(GENES, 0);

    std::normal_distribution<float> normal(0, 1);

    const auto fill = [&](std::size_t start, std::size_t count, float scale) {
        for (std::size_t i = 0; i < count; i++) {
            genome.weights[start + i] = normal(rng) * scale;
        }
    };

    fill(GENE_INPUT, HIDDEN * INPUTS, 1 / std::sqrt(static_cast<float>(INPUTS)));
    fill(GENE_RECURRENT, HIDDEN * HIDDEN, .5f / std::sqrt(static_cast<float>(HIDDEN)));
    fill(GENE_LEAK, HIDDEN, 1);
    fill(GENE_OUTPUT, OUTPUTS * (HIDDEN + INPUTS), 1 / std::sqrt(static_cast<float>(HIDDEN + INPUTS)));

    genome.squash();

    return genome;
}

Genome Genome::ancestor(std::mt19937_64& rng)
{
    Genome genome = random(rng);

    // The memory is there, but faint, so that the wiring below decides what it does at first.
    for (std::size_t i = GENE_INPUT; i < GENE_HIDDEN_BIAS; i++) {
        genome.weights[i] *= .3f;
    }

    const auto direct = [&](int output, int input) -> float& {
        return genome.weights[GENE_OUTPUT + static_cast<std::size_t>(output) * (HIDDEN + INPUTS) + HIDDEN + static_cast<std::size_t>(input)];
    };

    const auto bias = [&](int output) -> float& { return genome.weights[GENE_OUTPUT_BIAS + static_cast<std::size_t>(output)]; };

    for (int output = 0; output < OUTPUTS; output++) {
        for (int input = 0; input < INPUTS; input++) {
            direct(output, input) *= .3f;
        }
    }

    // Turn towards hares, bison, and carcasses, by how far to the left they are.
    // Bison the same as hares. One wolf can only bring down a calf or an old one this way, but
    // packmates that all run at the same bison bring down the rest, without meaning to work
    // together at all: cooperative hunting as a side effect of living in a group, which is where
    // Packer and Ruttan (1988) think it usually starts.
    for (const int seen : {IN_HARE, IN_BISON, IN_CARCASS}) {
        direct(OUT_TURN, seen + SEEN_LEFT) = 3;
    }

    // Run flat out at prey, and slow down for carcasses.
    direct(OUT_SPEED, IN_HARE + SEEN_CLOSE) = 3;
    direct(OUT_SPEED, IN_BISON + SEEN_CLOSE) = 3;
    direct(OUT_SPEED, IN_CARCASS + SEEN_CLOSE) = .5f;

    bias(OUT_SPEED) = .4f;
    direct(OUT_SPEED, IN_EATING) = -3;

    bias(OUT_BITE) = -1.5f;
    direct(OUT_BITE, IN_AT_HARE) = 3;
    direct(OUT_BITE, IN_AT_BISON) = 3;

    // Just short of calling at all, the way animals make all sorts of sounds before any of them
    // mean anything. Any small change can tip one in to calling, in some situation or other.
    for (int channel = 0; channel < CHANNELS; channel++) {
        bias(OUT_CALL + channel) = -.3f;
    }

    genome.squash();

    return genome;
}

Genome Genome::mutated(double chance, std::mt19937_64& rng) const
{
    Genome child = *this;

    child.generation = generation + 1;

    std::normal_distribution<double> normal(0, 1);
    std::uniform_real_distribution<double> uniform(0, 1);

    child.mutation_size = static_cast<float>(std::clamp(mutation_size * std::exp(MUTATION_SIZE_DRIFT * normal(rng)),
                                                        static_cast<double>(SMALLEST_MUTATION), static_cast<double>(BIGGEST_MUTATION)));

    // Picking which weights mutate by skipping ahead a geometric number each time, rather than
    // rolling for every one of them, since only a few percent change.
    if (chance <= 0) {
        child.squash();
        return child;
    }

    std::geometric_distribution<std::size_t> skip(std::min(1.0, chance));

    for (std::size_t gene = skip(rng); gene < GENES; gene += 1 + skip(rng)) {
        if (uniform(rng) < RESET_SHARE) {
            child.weights[gene] = static_cast<float>(normal(rng) * .3);
        } else {
            child.weights[gene] += static_cast<float>(normal(rng) * child.mutation_size);
        }
    }

    child.squash();

    return child;
}

void Brain::think(const Inputs& in, Outputs& out)
{
    const float* w = genes->weights.data();
    const float* leak = genes->leak.data();

    std::array<float, HIDDEN> next;

    for (int unit = 0; unit < HIDDEN; unit++) {
        const float* from_input = w + GENE_INPUT + static_cast<std::size_t>(unit) * INPUTS;
        const float* from_memory = w + GENE_RECURRENT + static_cast<std::size_t>(unit) * HIDDEN;

        float sum = w[GENE_HIDDEN_BIAS + unit];

        for (int i = 0; i < INPUTS; i++) {
            sum += from_input[i] * in[i];
        }

        for (int i = 0; i < HIDDEN; i++) {
            sum += from_memory[i] * memory[i];
        }

        next[unit] = memory[unit] + leak[unit] * (std::tanh(sum) - memory[unit]);
    }

    memory = next;

    for (int o = 0; o < OUTPUTS; o++) {
        const float* row = w + GENE_OUTPUT + static_cast<std::size_t>(o) * (HIDDEN + INPUTS);

        float sum = w[GENE_OUTPUT_BIAS + o];

        for (int i = 0; i < HIDDEN; i++) {
            sum += row[i] * memory[i];
        }

        for (int i = 0; i < INPUTS; i++) {
            sum += row[HIDDEN + i] * in[i];
        }

        out[o] = std::tanh(sum);
    }

    // A wolf whose memory has gone wrong just forgets, rather than carrying a NaN around.
    for (float& unit : memory) {
        if (!std::isfinite(unit)) {
            unit = 0;
        }
    }
}

}
