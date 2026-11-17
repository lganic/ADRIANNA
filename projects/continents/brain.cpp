#include "brain.hpp"

#include <algorithm>
#include <cmath>
#include <iterator>
#include <map>
#include <set>
#include <string>
#include <utility>

namespace continents {

// What founders start with. The learning rate and how wide the eyes look are the evolving parts of
// learning and sensing, as global hyperparameters on the blueprint, and this is where they start.
constexpr int FOUNDER_MEMORY_UNITS = 16;
constexpr int FOUNDER_MIX_UNITS = 16;
// projects/island_evolution found 1.5e-3 best for its dense founders, and projects/memory_chase found
// that LSTMs fell apart at 1e-3 when they had to remember for a long time. This is in between.
constexpr double FOUNDER_LEARNING_RATE = 1e-3;
constexpr double FOUNDER_ENTROPY = 1e-3;
constexpr double FOUNDER_EYE_SPREAD = 1.05; // About 60 degrees out to each side.

// How many steps go in to one update, which is also how far back gradients reach, and how far ahead
// the returns look before the critic takes over. 16 was best for projects/island_evolution.
constexpr int STRETCH_STEPS = 16;

constexpr float GAMMA = .98f;

constexpr float VALUE_WEIGHT = .5f;
constexpr float MAX_GRAD_NORM = 1;

// See projects/island_evolution/brain.cpp for why this is lower than projects/memory_chase.
constexpr float SCORE_SCALE = .02f;

constexpr float MIN_LOG_STD = -2;
constexpr float MAX_LOG_STD = 1;

static cdn::MutationParams scaled(cdn::MutationParams params, double scale)
{
    const auto scale_rate = [scale](double& rate) { rate = std::min(1.0, rate * scale); };

    scale_rate(params.add_node);
    scale_rate(params.add_connection);
    scale_rate(params.toggle_connection);
    scale_rate(params.node_hyperparams);
    scale_rate(params.add_cell_skip);
    scale_rate(params.toggle_cell_skip);
    scale_rate(params.change_species);
    scale_rate(params.globals);
    scale_rate(params.out_merge);

    return params;
}

Genetics::Genetics() : module_innovations(1), blueprint_innovations(1)
{
    io_spec.inputs = {{"state", {INPUT_SIZE}}};

    // Small starting weights on the actor, so that it starts out with no opinion on what to do.
    io_spec.outputs = {
        {"mean", ACTION_SIZE, "linear", .1f},
        {"log_std", ACTION_SIZE, "linear", .1f},
        {"value", 1, "linear", 1},
    };

    // Without dropout, since it would make the replay at the end of a stretch disagree with what
    // happened during it.
    layer_space.add(cdn::ParamSpec::choice("type", {"dense", "lstm"}));
    layer_space.add(cdn::ParamSpec::integer("units", 2, 32));
    layer_space.add(cdn::ParamSpec::choice("activation", {"tanh", "relu", "elu", "leaky_relu", "linear"}).only_for({"dense"}));
    layer_space.add(cdn::ParamSpec::real("weight_scale", .5, 2));
    layer_space.add(cdn::ParamSpec::choice("merge", {"concat", "sum"}));

    blueprint_space.add(cdn::ParamSpec::choice("merge", {"concat", "sum"}));

    // Small steps, since every child gets a go at this. In log space, .05 of the range is about 18% either way.
    global_space.add(cdn::ParamSpec::real("learning_rate", 1e-4, 3e-3, true).with_mutation_std(.05));
    global_space.add(cdn::ParamSpec::real("entropy_coef", 0, .01).with_mutation_std(.1));

    // From looking almost straight ahead, to seeing all the way around.
    global_space.add(cdn::ParamSpec::real("eye_spread", .2, 2.5).with_mutation_std(.03));

    // The same as projects/island_evolution, which explains them, plus a little of the LSTM cell skips.
    module_mutation.add_node = .08;
    module_mutation.add_connection = .08;
    module_mutation.toggle_connection = .02;
    module_mutation.node_hyperparams = .1;
    module_mutation.add_cell_skip = .02;
    module_mutation.toggle_cell_skip = .01;
    module_mutation.out_merge = .01;
    module_mutation.max_nodes = 4;

    blueprint_mutation.add_node = .05;
    blueprint_mutation.add_connection = .05;
    blueprint_mutation.toggle_connection = .02;
    blueprint_mutation.node_hyperparams = .05;
    blueprint_mutation.change_species = .03;
    blueprint_mutation.globals = .2;
    blueprint_mutation.out_merge = .01;
    blueprint_mutation.max_nodes = 4;
}

cdn::Genome Genetics::new_module(int species, cdn::Rng& rng)
{
    const cdn::GenomeOperators operators(cdn::GenomeKind::Module, layer_space, cdn::HyperparamSpace(), module_mutation, cdn::SpeciationParams());

    cdn::Genome module = operators.create_initial(1, module_innovations, rng);

    // The id goes in to the names of its layers. Keeping it the same as the species is what lets a
    // child find its parent's weights, since a module keeps its species from one generation to the next.
    module.id = species;

    return module;
}

static void set_layer(cdn::HyperparamValues& layer, const std::string& type, int units)
{
    layer.set_choice("type", type);
    layer.set_int("units", units);
    layer.set_float("weight_scale", 1);
    layer.set_choice("merge", "concat");

    if (type == "dense") {
        layer.set_choice("activation", "tanh");
    }
}

cdn::AssembledGenome Genetics::founder(cdn::Rng& rng)
{
    // Every founder is the same species, the same as the first generation of CoDeepNEAT.
    if (next_species == 0) {
        next_species++;
    }

    constexpr int species = 0;

    const cdn::GenomeOperators operators(cdn::GenomeKind::Blueprint, blueprint_space, global_space, blueprint_mutation, cdn::SpeciationParams());

    cdn::AssembledGenome genome;

    genome.blueprint = operators.create_initial(1, blueprint_innovations, rng, {species});
    genome.generation = 0;

    const int skip = blueprint_innovations.connection(0, genome.blueprint.output_id());
    genome.blueprint.conns[skip] = cdn::ConnGene{skip, 0, genome.blueprint.output_id(), true};

    for (auto& node : genome.blueprint.nodes) {
        node.second.hp.set_choice("merge", "concat");
    }

    genome.blueprint.globals.set_float("learning_rate", FOUNDER_LEARNING_RATE);
    genome.blueprint.globals.set_float("entropy_coef", FOUNDER_ENTROPY);
    genome.blueprint.globals.set_float("eye_spread", FOUNDER_EYE_SPREAD);

    // Starts out as input -> memory -> module output. The dense layer goes in between the memory and
    // the output, the same way an add node mutation would put it there, and gets the input as well.
    cdn::Genome module = new_module(species, rng);

    const int memory = module_innovations.initial_node();
    const int input = 0;
    const int output = module.output_id();

    const int through = module_innovations.connection(memory, output);
    const int mix = module_innovations.split_node(through, module);

    module.conns.at(through).enabled = false;
    module.nodes[mix] = cdn::NodeGene{mix, module.nodes.at(memory).hp, -1};

    for (const std::pair<int, int>& link : {std::make_pair(memory, mix), std::make_pair(mix, output), std::make_pair(input, mix)}) {
        const int innovation = module_innovations.connection(link.first, link.second);

        module.conns[innovation] = cdn::ConnGene{innovation, link.first, link.second, true};
    }

    set_layer(module.nodes.at(memory).hp, "lstm", FOUNDER_MEMORY_UNITS);
    set_layer(module.nodes.at(mix).hp, "dense", FOUNDER_MIX_UNITS);

    genome.modules[species] = std::move(module);

    return genome;
}

cdn::AssembledGenome Genetics::offspring(const cdn::AssembledGenome& parent, double rate_scale, cdn::Rng& rng)
{
    cdn::AssembledGenome child = parent;

    child.generation = parent.generation + 1;

    const cdn::GenomeOperators module_operators(cdn::GenomeKind::Module, layer_space, cdn::HyperparamSpace(), scaled(module_mutation, rate_scale), cdn::SpeciationParams());
    const cdn::GenomeOperators blueprint_operators(cdn::GenomeKind::Blueprint, blueprint_space, global_space, scaled(blueprint_mutation, rate_scale), cdn::SpeciationParams());

    for (auto& entry : child.modules) {
        module_operators.mutate(entry.second, module_innovations, rng);
    }

    // On top of the species it already has, a blueprint node can point to one that doesn't exist
    // yet. That is how a new module species comes about, starting out as a single random layer.
    std::vector<int> species_choices;

    for (const auto& entry : child.modules) {
        species_choices.push_back(entry.first);
    }

    const int fresh = next_species;

    species_choices.push_back(fresh);

    blueprint_operators.mutate(child.blueprint, blueprint_innovations, rng, species_choices);

    const auto points_to = [&](int species) {
        return std::any_of(child.blueprint.nodes.begin(), child.blueprint.nodes.end(), [species](const std::pair<const int, cdn::NodeGene>& node) {
            return node.second.species == species;
        });
    };

    if (points_to(fresh)) {
        child.modules[fresh] = new_module(fresh, rng);
        next_species++;
    }

    for (auto entry = child.modules.begin(); entry != child.modules.end();) {
        entry = points_to(entry->first) ? std::next(entry) : child.modules.erase(entry);
    }

    return child;
}

Brain::Brain(cdn::AssembledGenome genome, const cdn::IOSpec& io, uint64_t seed, const std::vector<cdn::NamedTensor>* inherited)
    : genotype(std::move(genome)),
      rng(seed),
      net(std::make_unique<cdn::Network>(cdn::assemble_network(genotype, io), cdn::BuildOptions(), rng.next_u64())),
      mean_head(net->output_index("mean")),
      log_std_head(net->output_index("log_std")),
      value_head(net->output_index("value")),
      rate(static_cast<float>(genotype.globals().f("learning_rate", FOUNDER_LEARNING_RATE))),
      entropy_weight(static_cast<float>(genotype.globals().f("entropy_coef", FOUNDER_ENTROPY))),
      spread(genotype.globals().f("eye_spread", FOUNDER_EYE_SPREAD)),
      parameters(net->num_parameters()),
      layers(0),
      memories(0),
      bite_cap(MAX_LOG_STD)
{
    if (inherited) {
        inherit(*inherited);
    }

    optimizer = cdn::make_optimizer("adam", net->named_parameters(), rate);

    for (const cdn::NodeInfo& node : net->nodes()) {
        if (node.kind == cdn::NodeKind::Layer && node.type != "identity") {
            layers++;
        }

        if (node.kind == cdn::NodeKind::Layer && node.type == "lstm") {
            memories++;
        }
    }

    net->reset_state(1);

    states.reserve(STRETCH_STEPS * INPUT_SIZE);
    actions.reserve(STRETCH_STEPS * ACTION_SIZE);
    rewards.reserve(STRETCH_STEPS);
    bite_rewards.reserve(STRETCH_STEPS);
}

// An LSTM's weights are [4 * units, inputs + units], with a block of rows for each of its four gates,
// and its own last output on the end of the inputs. So when it changes size, each part has to be
// lined up on its own.
static void inherit_memory(const cdn::Tensor& old, cdn::Tensor& now)
{
    const std::vector<float>& from = old.data();
    std::vector<float>& to = now.data();

    if (old.dim() == 1 && now.dim() == 1) {
        const int old_units = old.shape()[0] / 4;
        const int now_units = now.shape()[0] / 4;

        for (int gate = 0; gate < 4; gate++) {
            for (int unit = 0; unit < std::min(old_units, now_units); unit++) {
                to[gate * now_units + unit] = from[gate * old_units + unit];
            }
        }

        return;
    }

    if (old.dim() != 2 || now.dim() != 2 || old.shape()[0] % 4 != 0 || now.shape()[0] % 4 != 0) {
        return;
    }

    const int old_units = old.shape()[0] / 4;
    const int now_units = now.shape()[0] / 4;
    const int old_inputs = old.shape()[1] - old_units;
    const int now_inputs = now.shape()[1] - now_units;

    if (old_inputs < 0 || now_inputs < 0) {
        return;
    }

    const int old_width = old.shape()[1];
    const int now_width = now.shape()[1];

    for (int gate = 0; gate < 4; gate++) {
        for (int unit = 0; unit < std::min(old_units, now_units); unit++) {
            const float* from_row = &from[(gate * old_units + unit) * old_width];
            float* to_row = &to[(gate * now_units + unit) * now_width];

            for (int col = 0; col < now_inputs; col++) {
                to_row[col] = col < old_inputs ? from_row[col] : 0.0f;
            }

            for (int col = 0; col < now_units; col++) {
                to_row[now_inputs + col] = col < old_units ? from_row[old_inputs + col] : 0.0f;
            }
        }
    }
}

void Brain::inherit(const std::vector<cdn::NamedTensor>& inherited)
{
    std::map<std::string, const cdn::Tensor*> by_name;

    for (const cdn::NamedTensor& entry : inherited) {
        by_name[entry.name] = &entry.tensor;
    }

    // The main weights and biases of every LSTM. Their cell skips are left to the general case below.
    std::set<std::string> memory_parameters;

    for (const cdn::NodeInfo& node : net->nodes()) {
        if (node.kind == cdn::NodeKind::Layer && node.type == "lstm") {
            memory_parameters.insert(node.name + ".weight");
            memory_parameters.insert(node.name + ".bias");
        }
    }

    for (cdn::NamedTensor entry : net->named_parameters()) {
        const auto found = by_name.find(entry.name);

        if (found == by_name.end()) {
            continue;
        }

        const cdn::Tensor& old = *found->second;
        cdn::Tensor& now = entry.tensor;

        if (old.shape() == now.shape()) {
            now.copy_from(old);
            continue;
        }

        if (memory_parameters.count(entry.name) != 0) {
            inherit_memory(old, now);
            continue;
        }

        if (old.dim() != now.dim() || now.dim() > 2) {
            continue;
        }

        // Weights are [outputs, inputs], and biases are [outputs].
        const int old_rows = old.shape()[0];
        const int now_rows = now.shape()[0];
        const int old_cols = old.dim() == 2 ? old.shape()[1] : 1;
        const int now_cols = now.dim() == 2 ? now.shape()[1] : 1;

        const std::vector<float>& from = old.data();
        std::vector<float>& to = now.data();

        for (int row = 0; row < std::min(old_rows, now_rows); row++) {
            for (int col = 0; col < now_cols; col++) {
                to[row * now_cols + col] = col < old_cols ? from[row * old_cols + col] : 0.0f;
            }
        }
    }
}

Heritage Brain::heritage() const
{
    return {genotype, net->state_dict()};
}

bool Brain::learns_next() const
{
    return acted && static_cast<int>(rewards.size()) + 1 == STRETCH_STEPS;
}

cdn::Tensor Brain::input(const float* state) const
{
    return cdn::Tensor({1, INPUT_SIZE}, std::vector<float>(state, state + INPUT_SIZE));
}

Action Brain::run(const Outcome& outcome, const State& state, const Learning& learning)
{
    if (acted) {
        rewards.push_back(outcome.score * SCORE_SCALE);
        bite_rewards.push_back(outcome.could_bite ? outcome.bite_score * SCORE_SCALE : 0);
    }

    lambda = learning.lambda;

    // The memory from before this step, which is where the live run is, whatever the learning does.
    const std::vector<cdn::NamedTensor> memory = net->get_state();

    if (static_cast<int>(rewards.size()) == STRETCH_STEPS) {
        float bootstrap;

        {
            cdn::NoGradGuard no_grad;

            bootstrap = net->forward(std::vector<cdn::Tensor>{input(state.data())})[value_head].item();
        }

        learn(bootstrap);

        net->set_state(memory);
    }

    if (states.empty()) {
        stretch_memory = memory;
    }

    // After learning, since that has to go by what the stretch was picked with.
    bite_cap = std::min(MAX_LOG_STD, std::max(MIN_LOG_STD, learning.bite_noise));
    bite_focus = std::min(1.0f, std::max(0.0f, learning.bite_focus));

    std::vector<cdn::Tensor> outputs;

    {
        cdn::NoGradGuard no_grad;

        outputs = net->forward(std::vector<cdn::Tensor>{input(state.data())});
    }

    const std::vector<float>& mean = outputs[mean_head].data();
    const std::vector<float>& log_std = outputs[log_std_head].data();

    Action action;

    for (int i = 0; i < ACTION_SIZE; i++) {
        const float most = i == ACT_ATTACK ? bite_cap : MAX_LOG_STD;
        const float spread_out = std::exp(std::min(most, std::max(MIN_LOG_STD, log_std[i])));

        action[i] = mean[i] + spread_out * static_cast<float>(rng.normal());
        last_mean[i] = mean[i];
        last_spread[i] = spread_out;
    }

    states.insert(states.end(), state.begin(), state.end());
    actions.insert(actions.end(), action.begin(), action.end());

    acted = true;

    return action;
}

void Brain::learn(float bootstrap)
{
    std::vector<float> returns(STRETCH_STEPS);

    float running = bootstrap;

    for (int step = STRETCH_STEPS - 1; step >= 0; step--) {
        running = rewards[step] + GAMMA * running;
        returns[step] = running;
    }

    // The whole stretch again, this time with gradients, from the memory it started with.
    net->set_state(stretch_memory);

    std::vector<cdn::Tensor> means;
    std::vector<cdn::Tensor> log_stds;
    std::vector<cdn::Tensor> values;

    for (int step = 0; step < STRETCH_STEPS; step++) {
        const std::vector<cdn::Tensor> outputs = net->forward(std::vector<cdn::Tensor>{input(&states[static_cast<std::size_t>(step) * INPUT_SIZE])});

        means.push_back(outputs[mean_head]);
        log_stds.push_back(outputs[log_std_head]);
        values.push_back(outputs[value_head]);
    }

    const cdn::Tensor mean = cdn::concat(means, 0);
    cdn::Tensor log_std = cdn::clamp(cdn::concat(log_stds, 0), MIN_LOG_STD, MAX_LOG_STD);

    // Over the cap, the bite's spread gets no gradient at all, so neither the entropy bonus nor
    // anything else can push it any higher.
    if (bite_cap < MAX_LOG_STD) {
        std::vector<float> most(ACTION_SIZE, MAX_LOG_STD);

        most[ACT_ATTACK] = bite_cap;

        log_std = cdn::minimum(log_std, cdn::Tensor({1, ACTION_SIZE}, std::move(most)));
    }
    const cdn::Tensor value = cdn::reshape(cdn::concat(values, 0), {STRETCH_STEPS});

    const cdn::Tensor taken({STRETCH_STEPS, ACTION_SIZE}, actions);

    // Log probability of what it did. The constant in it is left out, since it has no gradient.
    const cdn::Tensor z = (taken - mean) / cdn::exp(log_std);
    const cdn::Tensor log_prob_each = -.5f * cdn::square(z) - log_std;
    const cdn::Tensor log_prob = cdn::sum(log_prob_each, 1);

    const cdn::Tensor target({STRETCH_STEPS}, std::move(returns));

    // How much better each action did than the critic expected. At a lambda of 1, that is everything
    // that came after it, to the end of the stretch, which is fair but noisy, since every reward
    // after an action counts towards it as much as its own does. Lower down, more of it is what the
    // critic thinks of where the action led, and less of it is what came later (generalised
    // advantage estimation), so that what an action got straight away stands out more.
    cdn::Tensor advantage;

    if (lambda >= 1) {
        advantage = target - value.detach();
    } else {
        const std::vector<float>& expected = value.data();

        std::vector<float> better(STRETCH_STEPS);

        float later = 0;

        for (int step = STRETCH_STEPS - 1; step >= 0; step--) {
            const float next = step + 1 < STRETCH_STEPS ? expected[static_cast<std::size_t>(step + 1)] : bootstrap;
            const float surprise = rewards[static_cast<std::size_t>(step)] + GAMMA * next - expected[static_cast<std::size_t>(step)];

            later = surprise + GAMMA * lambda * later;
            better[static_cast<std::size_t>(step)] = later;
        }

        advantage = cdn::Tensor({STRETCH_STEPS}, std::move(better));
    }

    cdn::Tensor policy_loss;

    if (bite_focus <= 0) {
        policy_loss = -cdn::mean(log_prob * advantage);
    } else {
        // Something is only in reach a percent or two of the time, so to the bite, the rest of what
        // happens is noise that swamps the little that it does. Judged by just what it got on the
        // step it bit on (what it took, what biting cost, and what it did to family), it only learns
        // from what it did itself. That is only on steps that it could have bitten something on,
        // since otherwise biting only ever costs a little, every step, which is enough for the
        // optimiser to keep pushing the bite down until it never bites anything, even in reach.
        const cdn::Tensor bite_log_prob = cdn::reshape(cdn::narrow(log_prob_each, 1, ACT_ATTACK, 1), {STRETCH_STEPS});
        const cdn::Tensor bite_now({STRETCH_STEPS}, bite_rewards);
        const cdn::Tensor bite_advantage = advantage * (1 - bite_focus) + bite_now * bite_focus;

        policy_loss = -cdn::mean((log_prob - bite_log_prob) * advantage + bite_log_prob * bite_advantage);
    }
    const cdn::Tensor value_loss = cdn::mean(cdn::square(target - value));
    const cdn::Tensor entropy = cdn::mean(cdn::sum(log_std, 1));

    cdn::Tensor loss = policy_loss + value_loss * VALUE_WEIGHT - entropy * entropy_weight;

    optimizer->zero_grad();
    loss.backward();
    cdn::clip_grad_norm(net->parameters(), MAX_GRAD_NORM);
    optimizer->step();

    states.clear();
    actions.clear();
    rewards.clear();
    bite_rewards.clear();
}

}
