#include "brain.hpp"

#include <algorithm>
#include <cmath>
#include <iterator>
#include <map>
#include <string>
#include <utility>

namespace island_evolution {

// What founders start with. The learning rate is the one thing about learning that evolves, as one
// of the global hyperparameters on the blueprint, and this is where it starts.
constexpr int FOUNDER_UNITS = 8;
// 5e-4 learned about three times slower, and 5e-3 or more often fell apart within a few hours.
constexpr double FOUNDER_LEARNING_RATE = 1.5e-3;
constexpr double FOUNDER_ENTROPY = 1e-3;

// How many steps go in to one update, which is also how far ahead the returns look before the critic takes over.
constexpr int STRETCH_STEPS = 16; // 8 learned slower.

constexpr float GAMMA = .98f;

constexpr float VALUE_WEIGHT = .5f;
constexpr float MAX_GRAD_NORM = 1;

// With GAMMA above, values come out about 50 times bigger than scores. Adam only moves a weight about
// one learning rate per update, and there is only one update per stretch, so a value of -20 takes the
// critic hours just to reach. At .1, as in projects/memory_chase, food per hour after 2 hours was a
// third of what it is at this.
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

    // Dense layers only, so that there is no memory to carry between steps. Without dropout, since
    // it would make the batch at the end of a stretch disagree with what happened during it.
    layer_space.add(cdn::ParamSpec::choice("type", {"dense"}));
    layer_space.add(cdn::ParamSpec::integer("units", 2, 32));
    layer_space.add(cdn::ParamSpec::choice("activation", {"tanh", "relu", "elu", "leaky_relu", "linear"}));
    layer_space.add(cdn::ParamSpec::real("weight_scale", .5, 2));
    layer_space.add(cdn::ParamSpec::choice("merge", {"concat", "sum"}));

    blueprint_space.add(cdn::ParamSpec::choice("merge", {"concat", "sum"}));

    // Small steps, since every child gets a go at this. In log space, .05 of the range is about 18% either way.
    global_space.add(cdn::ParamSpec::real("learning_rate", 1e-4, 3e-3, true).with_mutation_std(.05));
    global_space.add(cdn::ParamSpec::real("entropy_coef", 0, .01).with_mutation_std(.1));

    // Lower than the CoDeepNEAT defaults, since those are per generation of a whole population, and
    // these are per child. With these, about half of all children get some change, and a bit under a
    // third get a change to how their network is wired.
    module_mutation.add_node = .08;
    module_mutation.add_connection = .08;
    module_mutation.toggle_connection = .02;
    module_mutation.node_hyperparams = .1;
    module_mutation.add_cell_skip = 0;
    module_mutation.toggle_cell_skip = 0;
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

    cdn::Genome module = new_module(species, rng);

    for (auto& node : module.nodes) {
        cdn::HyperparamValues& layer = node.second.hp;

        layer.set_choice("type", "dense");
        layer.set_int("units", FOUNDER_UNITS);
        layer.set_choice("activation", "tanh");
        layer.set_float("weight_scale", 1);
        layer.set_choice("merge", "concat");
    }

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
      parameters(net->num_parameters()),
      layers(0)
{
    if (inherited) {
        inherit(*inherited);
    }

    optimizer = cdn::make_optimizer("adam", net->named_parameters(), rate);

    for (const cdn::NodeInfo& node : net->nodes()) {
        if (node.kind == cdn::NodeKind::Layer && node.type != "identity") {
            layers++;
        }
    }

    states.reserve(STRETCH_STEPS * INPUT_SIZE);
    actions.reserve(STRETCH_STEPS * ACTION_SIZE);
    rewards.reserve(STRETCH_STEPS);
}

void Brain::inherit(const std::vector<cdn::NamedTensor>& inherited)
{
    std::map<std::string, const cdn::Tensor*> by_name;

    for (const cdn::NamedTensor& entry : inherited) {
        by_name[entry.name] = &entry.tensor;
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

Action Brain::run(float score, const State& state)
{
    if (acted) {
        rewards.push_back(score * SCORE_SCALE);
    }

    if (static_cast<int>(rewards.size()) == STRETCH_STEPS) {
        learn(state);
    }

    std::vector<cdn::Tensor> outputs;

    {
        cdn::NoGradGuard no_grad;

        outputs = net->forward(std::vector<cdn::Tensor>{cdn::Tensor({1, INPUT_SIZE}, std::vector<float>(state.begin(), state.end()))});
    }

    const std::vector<float>& mean = outputs[mean_head].data();
    const std::vector<float>& log_std = outputs[log_std_head].data();

    Action action;

    for (int i = 0; i < ACTION_SIZE; i++) {
        const float spread = std::exp(std::min(MAX_LOG_STD, std::max(MIN_LOG_STD, log_std[i])));

        action[i] = mean[i] + spread * static_cast<float>(rng.normal());
    }

    states.insert(states.end(), state.begin(), state.end());
    actions.insert(actions.end(), action.begin(), action.end());

    acted = true;

    return action;
}

void Brain::learn(const State& next_state)
{
    float running;

    {
        cdn::NoGradGuard no_grad;

        const std::vector<cdn::Tensor> outputs = net->forward(std::vector<cdn::Tensor>{cdn::Tensor({1, INPUT_SIZE}, std::vector<float>(next_state.begin(), next_state.end()))});

        running = outputs[value_head].item();
    }

    std::vector<float> returns(STRETCH_STEPS);

    for (int step = STRETCH_STEPS - 1; step >= 0; step--) {
        running = rewards[step] + GAMMA * running;
        returns[step] = running;
    }

    const std::vector<cdn::Tensor> outputs = net->forward(std::vector<cdn::Tensor>{cdn::Tensor({STRETCH_STEPS, INPUT_SIZE}, states)});

    const cdn::Tensor& mean = outputs[mean_head];
    const cdn::Tensor log_std = cdn::clamp(outputs[log_std_head], MIN_LOG_STD, MAX_LOG_STD);
    const cdn::Tensor value = cdn::reshape(outputs[value_head], {STRETCH_STEPS});

    const cdn::Tensor taken({STRETCH_STEPS, ACTION_SIZE}, actions);

    // Log probability of what it did. The constant in it is left out, since it has no gradient.
    const cdn::Tensor z = (taken - mean) / cdn::exp(log_std);
    const cdn::Tensor log_prob = cdn::sum(-.5f * cdn::square(z) - log_std, 1);

    const cdn::Tensor target({STRETCH_STEPS}, std::move(returns));
    const cdn::Tensor advantage = target - value.detach();

    const cdn::Tensor policy_loss = -cdn::mean(log_prob * advantage);
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
}

}
