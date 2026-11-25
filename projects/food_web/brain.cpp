#include "brain.hpp"

#include <algorithm>
#include <cmath>

namespace food_web {

constexpr float MIN_LEAK = .05f;

constexpr float SMALLEST_MUTATION = .003f;
constexpr float BIGGEST_MUTATION = .3f;

// How much the mutation size itself mutates, in log space, per child. About 10% either way.
constexpr double MUTATION_SIZE_DRIFT = .1;

// Very rarely, a weight starts over from scratch instead of being nudged.
constexpr double RESET_SHARE = .001;

void Genome::squash()
{
    leak.resize(static_cast<std::size_t>(layout.hidden));

    for (int unit = 0; unit < layout.hidden; unit++) {
        const float raw = weights[layout.leak() + static_cast<std::size_t>(unit)];

        leak[static_cast<std::size_t>(unit)] = MIN_LEAK + (1 - MIN_LEAK) / (1 + std::exp(-raw));
    }
}

Genome Genome::random(const Layout& layout, std::mt19937_64& rng)
{
    Genome genome;

    genome.layout = layout;
    genome.weights.assign(layout.genes(), 0);

    std::normal_distribution<float> normal(0, 1);

    const auto fill = [&](std::size_t start, std::size_t count, float scale) {
        for (std::size_t i = 0; i < count; i++) {
            genome.weights[start + i] = normal(rng) * scale;
        }
    };

    const std::size_t hidden = static_cast<std::size_t>(layout.hidden);
    const std::size_t inputs = static_cast<std::size_t>(layout.inputs);
    const std::size_t outputs = static_cast<std::size_t>(layout.outputs);

    fill(layout.input_weights(), hidden * inputs, 1 / std::sqrt(static_cast<float>(inputs)));
    fill(layout.recurrent_weights(), hidden * hidden, .5f / std::sqrt(static_cast<float>(hidden)));
    fill(layout.leak(), hidden, 1);
    fill(layout.output_weights(), outputs * (hidden + inputs), 1 / std::sqrt(static_cast<float>(hidden + inputs)));

    genome.squash();

    return genome;
}

void Genome::quieten(float factor)
{
    for (std::size_t i = layout.input_weights(); i < layout.hidden_bias(); i++) {
        weights[i] *= factor;
    }

    for (int output = 0; output < layout.outputs; output++) {
        for (int input = 0; input < layout.inputs; input++) {
            weights[layout.direct(output, input)] *= factor;
        }
    }
}

Genome Genome::mutated(double chance, std::mt19937_64& rng) const
{
    Genome child = *this;

    child.generation = generation + 1;

    std::normal_distribution<double> normal(0, 1);
    std::uniform_real_distribution<double> uniform(0, 1);

    child.mutation_size = static_cast<float>(std::clamp(mutation_size * std::exp(MUTATION_SIZE_DRIFT * normal(rng)), static_cast<double>(SMALLEST_MUTATION),
                                                        static_cast<double>(BIGGEST_MUTATION)));

    if (chance > 0) {
        // Skipping ahead a geometric number each time, which comes to the same as rolling for each
        // weight, and is quicker when only a few of them change.
        std::geometric_distribution<std::size_t> skip(std::min(1.0, chance));

        for (std::size_t gene = skip(rng); gene < child.weights.size(); gene += 1 + skip(rng)) {
            if (uniform(rng) < RESET_SHARE) {
                child.weights[gene] = static_cast<float>(normal(rng) * .3);
            } else {
                child.weights[gene] += static_cast<float>(normal(rng) * child.mutation_size);
            }
        }
    }

    child.squash();

    return child;
}

void Brain::think(const float* in, float* out)
{
    const Layout& layout = genes->layout;
    const float* w = genes->weights.data();
    const float* leak = genes->leak.data();

    const int hidden = layout.hidden;
    const int inputs = layout.inputs;

    std::array<float, MOST_HIDDEN> next;

    for (int unit = 0; unit < hidden; unit++) {
        const float* from_input = w + layout.input_weights() + static_cast<std::size_t>(unit * inputs);
        const float* from_memory = w + layout.recurrent_weights() + static_cast<std::size_t>(unit * hidden);

        float sum = w[layout.hidden_bias() + static_cast<std::size_t>(unit)];

        for (int i = 0; i < inputs; i++) {
            sum += from_input[i] * in[i];
        }

        for (int i = 0; i < hidden; i++) {
            sum += from_memory[i] * memory[static_cast<std::size_t>(i)];
        }

        next[static_cast<std::size_t>(unit)] = memory[static_cast<std::size_t>(unit)] + leak[unit] * (std::tanh(sum) - memory[static_cast<std::size_t>(unit)]);
    }

    for (int unit = 0; unit < hidden; unit++) {
        const float value = next[static_cast<std::size_t>(unit)];

        // A brain whose memory has gone wrong just forgets, rather than carrying a NaN around.
        memory[static_cast<std::size_t>(unit)] = std::isfinite(value) ? value : 0;
    }

    for (int o = 0; o < layout.outputs; o++) {
        const float* row = w + layout.output_weights() + static_cast<std::size_t>(o * (hidden + inputs));

        float sum = w[layout.bias(o)];

        for (int i = 0; i < hidden; i++) {
            sum += row[i] * memory[static_cast<std::size_t>(i)];
        }

        for (int i = 0; i < inputs; i++) {
            sum += row[hidden + i] * in[i];
        }

        out[o] = std::tanh(sum);
    }
}

}
