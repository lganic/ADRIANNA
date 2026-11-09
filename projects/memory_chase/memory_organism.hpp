#ifndef MEMORY_CHASE_MEMORY_ORGANISM_HPP
#define MEMORY_CHASE_MEMORY_ORGANISM_HPP

#include <codeepneat/network.hpp>
#include <codeepneat/optim.hpp>
#include <codeepneat/rng.hpp>
#include <codeepneat/tensor.hpp>

#include <cstdint>
#include <memory>
#include <vector>

namespace memory_chase {

// sim::Organism, but with an LSTM in it, so that it can hold on to something it saw a while ago.
//
// The actor and critic share the LSTM, so that it gets taught by both. The critic is what really
// teaches it, since what the next few seconds are worth depends so heavily on the mood.
//
// Learning works the same way as the A2C trainer in CDN, except with continuous actions. It collects
// a stretch of steps from every world at once, backpropagates through that stretch, and then cuts the
// memory off from the graph. So gradients only reach back one stretch, but the memory itself
// carries on across them.
class MemoryOrganism {
public:
    MemoryOrganism(int num_inputs, int num_outputs, int num_worlds, uint64_t seed);

    // Takes the score each world got for the last action, and the states as [num_worlds, num_inputs].
    // Returns the actions as [num_worlds, num_outputs].
    cdn::Tensor run(const std::vector<float>& scores, const cdn::Tensor& states);

    // Empty out the LSTM.
    void wipe_memory();

private:
    void learn(const cdn::Tensor& next_states);

    int num_worlds;

    cdn::Rng rng;

    cdn::Network net;
    std::unique_ptr<cdn::Optimizer> optimizer;

    int mean_head;
    int log_std_head;
    int value_head;

    bool acted = false; // Whether the scores coming in are for an action of ours yet.

    // The stretch so far. One entry per step, where each one has a value for every world.
    std::vector<cdn::Tensor> log_probs;
    std::vector<cdn::Tensor> values;
    std::vector<cdn::Tensor> entropies;
    std::vector<float> rewards; // [steps * num_worlds]
};

}

#endif // MEMORY_CHASE_MEMORY_ORGANISM_HPP
