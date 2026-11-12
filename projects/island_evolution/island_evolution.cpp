#include "island_evolution.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <exception>
#include <map>
#include <set>
#include <string>
#include <thread>
#include <utility>

namespace island_evolution {

constexpr double PI = 3.14159265358979323846;

// Where the right thruster sits, relative to the middle of the organism.
constexpr Point RIGHT_LEG = {ORGANISM_SIZE * RIGHT_FOOT.x, ORGANISM_SIZE * (RIGHT_FOOT.y - DRAWING_CENTER_Y)};

// Thrust past this doesn't make the flames any bigger, so that a wild output can't fill the screen.
// Lower than projects/island_food, since with this many agents around, bigger flames hide the bodies.
constexpr double MAX_FLAME_BOOST = 1;

// The bars drawn over each agent, in meters.
constexpr double BAR_WIDTH = 3;
constexpr double BAR_HEIGHT = .5;
constexpr double BAR_GAP = 2; // From the middle of the agent up to the lowest bar.

// One for each core, going by two hardware threads per core. Every step waits on the slowest thread,
// and on the physics in between, which only the main thread can do. So going any higher only made it
// slower: on a 12 core, 24 thread Ryzen, 8 to 12 pools all came out about the same, at 2.7 to 4 times
// as fast as one thread, while 16 was slower, and 24 was slower again. With every hardware thread
// busy, whatever else the computer is doing (the GUI included) knocks one of them off now and then,
// and then the whole step waits on it.
int IslandEvolution::default_pool_count()
{
    return std::max(1, static_cast<int>(std::thread::hardware_concurrency() / 2));
}

IslandEvolution::IslandEvolution(sim::Ui& ui, uint64_t seed, int pool_count)
    : world(BOX_SIZE, ORGANISM_RADIUS, FOOD_RADIUS, RIGHT_LEG, generate_islands(BOX_SIZE, ISLAND_COUNT, WALL_SENSE_RANGE + 1, ISLAND_GAP, ISLAND_SEED), seed),
      rng(seed),
      pools(static_cast<std::size_t>(std::max(1, pool_count))),
      threads(std::max(1, pool_count)),
      graphic(plotato::Color(0, 0, 255), ORGANISM_SIZE),
      food_growth(ui.add_slider({"Food growth (pieces per second)", 0, 20, 2, .1})),
      max_food(ui.add_slider({"Most food at once", 10, 1000, 300, 10})),
      food_energy(ui.add_slider({"Energy per food", 0, 5, 1, .05})),
      metabolism(ui.add_slider({"Energy used per second", 0, .5, .02, .005})),
      thrust_energy(ui.add_slider({"Energy per second per unit of thrust", 0, .05, .002, .0005})),
      start_energy(ui.add_slider({"Starting energy", 0, 20, 5, .5})),
      reproduction_cost(ui.add_slider({"Energy to reproduce", .5, 20, 5, .5})),
      reproduction_time(ui.add_slider({"Seconds to reproduce at full effort", .2, 20, 1, .1})),
      reproduction_decay(ui.add_slider({"Progress lost per second when not trying", 0, 2, .2, .01})),
      max_agents(ui.add_slider({"Most agents at once", 1, 200, 100, 1})),
      mutation_rate(ui.add_slider({"Mutation rate", 0, 5, 1, .1})),
      food_reward(ui.add_entry({"Food reward", 0, 100, 10, 1})),
      birth_reward(ui.add_entry({"Birth reward", 0, 500, 0, 1})),
      closeness_reward(ui.add_slider({"Closeness reward", 0, 5, 1, .1})),
      water_punish(ui.add_slider({"Water punishment", 0, 20, 5, .5})),
      water_friction(ui.add_slider({"Water friction", 0, 10, 2, .1})),
      wall_punish(ui.add_slider({"Wall punishment", 0, 20, 5, .5})),
      effort_punish(ui.add_slider({"Effort punishment", 0, 1, .1, .01})),
      spin_punish(ui.add_slider({"Spin punishment", 0, 1, .2, .01})),
      network_graph(ui.add_live_graph({"Network of the featured agent", "", sim::Page::World, "", true,
                                       [](plotato::Graph& graph) { graph.set_bounds({-.1, 1.1, -1.5, 1.6}); }})),
      population_field(ui.add_field({"Population", sim::Page::World, "%.0f"})),
      births_field(ui.add_field({"Births", sim::Page::World, "%.0f"})),
      deaths_field(ui.add_field({"Deaths", sim::Page::World, "%.0f"})),
      extinction_field(ui.add_field({"Extinctions", sim::Page::World, "%.0f"})),
      generation_field(ui.add_field({"Highest generation alive", sim::Page::World, "%.0f"})),
      record_field(ui.add_field({"Most children by one agent", sim::Page::World, "%.0f"})),
      species_field(ui.add_field({"Module species alive", sim::Page::World, "%.0f"})),
      layers_field(ui.add_field({"Most layers alive", sim::Page::World, "%.0f"})),
      population_graph(ui.add_history_graph({"Population Graph", "Agents alive"})),
      food_graph(ui.add_history_graph({"Food Graph", "Food per agent per hour"})),
      birth_graph(ui.add_history_graph({"Birth Graph", "Births per hour"})),
      energy_graph(ui.add_history_graph({"Energy Graph", "Average energy"})),
      reproducing_graph(ui.add_history_graph({"Reproducing Graph", "Fraction of agents trying to reproduce"})),
      water_graph(ui.add_history_graph({"Water Graph", "Fraction of agents in water"})),
      generation_graph(ui.add_history_graph({"Generation Graph", "Average generation"})),
      layers_graph(ui.add_history_graph({"Layers Graph", "Average hidden layers"})),
      parameters_graph(ui.add_history_graph({"Parameters Graph", "Average parameters"})),
      learning_rate_graph(ui.add_history_graph({"Learning Rate Graph", "Average learning rate"})),
      score_graph(ui.add_history_graph({"Score Graph", "Average score"}))
{
    const Settings settings = read_settings();

    for (int founder = 0; founder < FOUNDERS; founder++) {
        add_founder(settings);
    }

    update_order();
}

double IslandEvolution::time_step() const
{
    return DT;
}

void IslandEvolution::setup_world(plotato::Graph& graph)
{
    graph.equal_aspect_framing = true;
    graph.set_bounds({-BOX_SIZE * 1.1, BOX_SIZE * 1.1, -BOX_SIZE * 1.1, BOX_SIZE * 1.1});
}

Settings IslandEvolution::read_settings() const
{
    Settings settings;

    settings.food_growth = food_growth.value();
    settings.max_food = max_food.value();
    settings.food_energy = food_energy.value();
    settings.metabolism = metabolism.value();
    settings.thrust_energy = thrust_energy.value();
    settings.start_energy = start_energy.value();
    settings.reproduction_cost = reproduction_cost.value();
    settings.reproduction_time = reproduction_time.value();
    settings.reproduction_decay = reproduction_decay.value();
    settings.max_agents = max_agents.value();
    settings.mutation_rate = mutation_rate.value();

    settings.food_reward = food_reward.value();
    settings.birth_reward = birth_reward.value();
    settings.closeness_reward = closeness_reward.value();
    settings.water_punish = water_punish.value();
    settings.water_friction = water_friction.value();
    settings.wall_punish = wall_punish.value();
    settings.effort_punish = effort_punish.value();
    settings.spin_punish = spin_punish.value();

    return settings;
}

void IslandEvolution::add_founder(const Settings& settings)
{
    Agent agent;

    agent.id = next_id++;
    agent.body = world.add_body(world.random_land(), rng.uniform(-PI, PI));
    agent.family = agent.id;
    agent.energy = settings.start_energy;
    agent.brain = std::make_unique<Brain>(genetics.founder(rng), genetics.io(), rng.next_u64(), nullptr);

    place(std::move(agent));
}

void IslandEvolution::place(Agent agent)
{
    // On a tie, the lowest numbered one. That way pool 0, which the main thread runs itself, is never left empty.
    const auto emptiest = std::min_element(pools.begin(), pools.end(), [](const AgentPool& a, const AgentPool& b) { return a.agents.size() < b.agents.size(); });

    emptiest->agents.push_back(std::move(agent));
}

void IslandEvolution::update_order()
{
    order.clear();

    for (AgentPool& pool : pools) {
        for (Agent& agent : pool.agents) {
            order.push_back(&agent);
        }
    }

    std::sort(order.begin(), order.end(), [](const Agent* a, const Agent* b) { return a->id < b->id; });
}

template <typename Work>
void IslandEvolution::run_pools(bool slowest_first, Work work)
{
    for (AgentPool& pool : pools) {
        pool.prepare(slowest_first);
    }

    const int count = static_cast<int>(pools.size());

    threads.run([&](int own) {
        // Starting from the next pool along, so that the helpers spread out instead of all piling on to the same one.
        for (int offset = 0; offset < count; offset++) {
            AgentPool& pool = pools[(own + offset) % count];

            while (Agent* agent = pool.claim()) {
                work(*agent);
            }
        }
    });
}

std::unique_ptr<Brain> IslandEvolution::breed(const Heritage& parent, const Settings& settings)
{
    const uint64_t seed = rng.next_u64();

    // None of the mutations should ever make a network that can't be built, but if one does, it
    // shouldn't take the whole simulation down. The parent's own genome is known to work.
    try {
        return std::make_unique<Brain>(genetics.offspring(parent.genome, settings.mutation_rate, rng), genetics.io(), seed, &parent.weights);
    } catch (const std::exception&) {
        return std::make_unique<Brain>(parent.genome, genetics.io(), seed, &parent.weights);
    }
}

// Food is what children are made of, so the most of it over a lifetime is what makes the best agent.
// Going by children first sounds better, but it never moves on. Some early agent gets lucky with two
// children, and every restart after that comes from its weights, throwing away whatever the restarts
// themselves learned. Food goes up as learning carries on from one restart to the next, so it ratchets.
static bool beats(uint64_t eaten, int children, const Record& record)
{
    return eaten > record.eaten || (eaten == record.eaten && children > record.children);
}

void IslandEvolution::step(double)
{
    const Settings settings = read_settings();

    const double cost = settings.reproduction_cost;

    world.grow_food(settings.food_growth, static_cast<std::size_t>(settings.max_food), DT);

    // Decide what to do, all at once, and then push.
    run_pools(true, [&](Agent& agent) { think(agent, world, settings); });

    for (const Agent* agent : order) {
        Body& body = world.body(agent->body);

        body.left_thrust = agent->left_thrust;
        body.right_thrust = agent->right_thrust;
    }

    world.sim(settings.water_friction, DT);

    // See what came of it.
    run_pools(false, [&](Agent& agent) { score(agent, world, settings); });

    std::vector<Agent> born;

    int births = 0;
    int eaten = 0;
    int in_water = 0;
    int reproducing = 0;
    double total_score = 0;

    const std::size_t alive = order.size();

    for (Agent* agent : order) {
        const Body& body = world.body(agent->body);

        // With the population full, a finished agent waits on 1 until there is room.
        if (agent->progress >= 1 && agent->energy > cost && static_cast<double>(alive + born.size()) < settings.max_agents) {
            agent->energy -= cost;
            agent->progress = 0;
            agent->children++;
            agent->score += settings.birth_reward;

            most_children = std::max(most_children, agent->children);

            Agent child;

            child.id = next_id++;
            child.body = world.add_body(body.pos, body.rotation);
            child.generation = agent->generation + 1;
            child.family = agent->family;
            child.energy = CHILD_SHARE * cost;

            child.brain = breed(agent->brain->heritage(), settings);

            born.push_back(std::move(child));

            births++;
        }

        eaten += body.eaten;
        in_water += agent->wet ? 1 : 0;
        reproducing += agent->effort > 0 ? 1 : 0;
        total_score += agent->score;
    }

    // Out of energy is out of time. The last score never gets learned from, since there is no next step.
    int deaths = 0;

    for (Agent* agent : order) {
        if (agent->energy > 0) {
            continue;
        }

        if (!has_record || beats(agent->eaten, agent->children, record)) {
            record = {agent->brain->heritage(), agent->generation, agent->family, agent->children, agent->eaten};
            has_record = true;
        }

        world.remove_body(agent->body);
        agent->brain.reset();

        deaths++;
    }

    if (deaths > 0) {
        for (AgentPool& pool : pools) {
            pool.agents.erase(std::remove_if(pool.agents.begin(), pool.agents.end(), [](const Agent& agent) { return !agent.brain; }), pool.agents.end());
        }
    }

    for (Agent& child : born) {
        place(std::move(child));
    }

    const bool everybody_gone = std::all_of(pools.begin(), pools.end(), [](const AgentPool& pool) { return pool.agents.empty(); });

    // Everybody is gone, so the best there ever was gets to start over, with a child of its own.
    if (everybody_gone && has_record) {
        extinctions++;

        const Point spot = world.random_land();

        Agent restart;

        restart.id = next_id++;
        restart.body = world.add_body(spot, rng.uniform(-PI, PI));
        restart.generation = record.generation + 1;
        restart.family = record.family;
        restart.energy = settings.start_energy;
        restart.brain = breed(record.heritage, settings);

        place(std::move(restart));
    }

    // The pools moved agents around in memory, if anybody came or went.
    if (deaths > 0 || births > 0 || everybody_gone) {
        update_order();
    }

    total_births += births;
    total_deaths += deaths;
    total_eaten += eaten;

    population_graph.add(static_cast<double>(order.size()));
    birth_graph.add(births * 3600 / DT);

    if (alive > 0) {
        food_graph.add(eaten * 3600 / DT / alive);
        water_graph.add(static_cast<double>(in_water) / alive);
        reproducing_graph.add(static_cast<double>(reproducing) / alive);
        score_graph.add(total_score / alive);
    }

    if (!order.empty()) {
        double energy = 0;
        double generation = 0;
        double layers = 0;
        double parameters = 0;
        double learning_rate = 0;

        for (const Agent* agent : order) {
            energy += agent->energy;
            generation += agent->generation;
            layers += agent->brain->layer_count();
            parameters += static_cast<double>(agent->brain->parameter_count());
            learning_rate += agent->brain->learning_rate();
        }

        const double count = static_cast<double>(order.size());

        energy_graph.add(energy / count);
        generation_graph.add(generation / count);
        layers_graph.add(layers / count);
        parameters_graph.add(parameters / count);
        learning_rate_graph.add(learning_rate / count);
    }
}

const Agent* IslandEvolution::featured() const
{
    const Agent* best = nullptr;

    for (const Agent* agent : order) {
        if (!best || agent->children > best->children || (agent->children == best->children && agent->age > best->age)) {
            best = agent;
        }
    }

    return best;
}

static double flame_boost(double thrust)
{
    return std::min(MAX_FLAME_BOOST, std::max(-MAX_FLAME_BOOST, thrust));
}

// Spread the families out around the colour wheel, a golden angle apart, so that close ids still look different.
static plotato::Color family_color(int family)
{
    const double hue = std::fmod(family * 137.50776, 360.0) / 60;
    const double fraction = hue - std::floor(hue);

    constexpr double value = 200;
    constexpr double low = value * .25;

    const double rising = low + (value - low) * fraction;
    const double falling = value - (value - low) * fraction;

    double red = 0;
    double green = 0;
    double blue = 0;

    switch (static_cast<int>(hue) % 6) {
    case 0: red = value; green = rising; blue = low; break;
    case 1: red = falling; green = value; blue = low; break;
    case 2: red = low; green = value; blue = rising; break;
    case 3: red = low; green = falling; blue = value; break;
    case 4: red = rising; green = low; blue = value; break;
    default: red = value; green = low; blue = falling; break;
    }

    return plotato::Color(static_cast<uint8_t>(red), static_cast<uint8_t>(green), static_cast<uint8_t>(blue));
}

static void draw_bar(plotato::Graph& graph, Point position, double offset, double fraction, plotato::Color color)
{
    plotato::ShapeStyle back_style;
    back_style.fill = plotato::Color(60, 60, 60, 150);
    back_style.outline = plotato::Color(0, 0, 0, 0);

    plotato::ShapeStyle fill_style;
    fill_style.fill = color;
    fill_style.outline = plotato::Color(0, 0, 0, 0);

    const double left = position.x - BAR_WIDTH / 2;
    const double bottom = position.y + BAR_GAP + offset;
    const double top = bottom + BAR_HEIGHT;

    graph.polygon({left, left, left + BAR_WIDTH, left + BAR_WIDTH}, {bottom, top, top, bottom}, back_style);

    const double right = left + BAR_WIDTH * std::min(1.0, std::max(0.0, fraction));

    if (right > left) {
        graph.polygon({left, left, right, right}, {bottom, top, top, bottom}, fill_style);
    }
}

void IslandEvolution::draw(plotato::Graph& graph)
{
    constexpr double box = BOX_SIZE;

    plotato::ShapeStyle water_style;
    water_style.fill = plotato::Color(170, 215, 245);
    water_style.outline = plotato::Color(0, 0, 0, 0); // No outline.

    plotato::ShapeStyle land_style;
    land_style.fill = plotato::Color(235, 215, 160);
    land_style.outline = plotato::Color(0, 0, 0, 0); // No outline.

    plotato::PlotStyle wall_style;
    wall_style.line_color = plotato::Color(0, 0, 255);

    plotato::ShapeStyle food_style;
    food_style.fill = plotato::Color(230, 140, 0);
    food_style.outline = food_style.fill;

    plotato::ShapeStyle featured_style;
    featured_style.fill = plotato::Color(0, 0, 0, 0);
    featured_style.outline = plotato::Color(200, 0, 0);

    graph.polygon({-box, -box, box, box}, {-box, box, box, -box}, water_style);

    for (const Rect& island : world.island_list()) {
        graph.polygon({island.left, island.left, island.right, island.right}, {island.bottom, island.top, island.top, island.bottom}, land_style);
    }

    graph.plot({-box, -box, box, box, -box}, {-box, box, box, -box, -box}, wall_style);

    for (const Point& piece : world.food) {
        graph.circle(piece.x, piece.y, FOOD_RADIUS, food_style);
    }

    const double cost = reproduction_cost.value();

    for (const Agent* agent_pointer : order) {
        const Agent& agent = *agent_pointer;
        const Body& body = world.body(agent.body);

        graphic.set_color(family_color(agent.family));
        graphic.draw(graph, body.pos, body.rotation, flame_boost(agent.left_thrust), flame_boost(agent.right_thrust));

        // Energy, with the line for having enough to reproduce half way along it, and progress on top.
        draw_bar(graph, body.pos, 0, agent.energy / (2 * cost), agent.energy > cost ? plotato::Color(0, 170, 0) : plotato::Color(220, 160, 0));
        draw_bar(graph, body.pos, BAR_HEIGHT * 1.5, agent.progress, plotato::Color(200, 0, 200));
    }

    const Agent* chosen = featured();

    if (chosen) {
        const Body& body = world.body(chosen->body);

        graph.circle(body.pos.x, body.pos.y, ORGANISM_RADIUS * 3, featured_style);

        draw_network(network_graph.graph(), *chosen);
    }

    // Fields
    std::set<int> species;
    int most_layers = 0;
    int highest_generation = 0;

    for (const Agent* agent : order) {
        for (const auto& entry : agent->brain->genome().modules) {
            species.insert(entry.first);
        }

        most_layers = std::max(most_layers, agent->brain->layer_count());
        highest_generation = std::max(highest_generation, agent->generation);
    }

    population_field.set(static_cast<double>(order.size()));
    births_field.set(static_cast<double>(total_births));
    deaths_field.set(static_cast<double>(total_deaths));
    extinction_field.set(static_cast<double>(extinctions));
    generation_field.set(highest_generation);
    record_field.set(most_children);
    species_field.set(static_cast<double>(species.size()));
    layers_field.set(most_layers);
}

// Lay the network out left to right, by how many layers deep each one is. The identity nodes that
// CoDeepNEAT puts at the edges of modules, and where things merge, are left out, with their
// connections drawn straight through them.
void IslandEvolution::draw_network(plotato::Graph& graph, const Agent& agent)
{
    const cdn::NetworkGraph& net = agent.brain->network().graph();

    const auto shown = [&](const cdn::GraphNode& node) { return node.kind != cdn::NodeKind::Layer || node.layer_type != "identity"; };

    // For every node, the shown nodes that feed it, looking through identities. Inputs always come
    // before the nodes that use them, so one pass in order does it.
    std::vector<std::set<int>> sources(net.nodes.size());
    std::vector<int> depth(net.nodes.size(), 0);

    for (const cdn::GraphNode& node : net.nodes) {
        for (int input : node.inputs) {
            if (shown(net.nodes[input])) {
                sources[node.id].insert(input);
            } else {
                sources[node.id].insert(sources[input].begin(), sources[input].end());
            }
        }

        for (int source : sources[node.id]) {
            depth[node.id] = std::max(depth[node.id], depth[source] + (shown(node) ? 1 : 0));
        }
    }

    int deepest = 0;

    for (const cdn::GraphNode& node : net.nodes) {
        if (shown(node) && node.kind != cdn::NodeKind::Output) {
            deepest = std::max(deepest, depth[node.id]);
        }
    }

    // The heads all go in a column of their own at the end.
    std::map<int, std::vector<int>> columns;

    for (const cdn::GraphNode& node : net.nodes) {
        if (shown(node)) {
            columns[node.kind == cdn::NodeKind::Output ? deepest + 1 : depth[node.id]].push_back(node.id);
        }
    }

    std::map<int, Point> where;

    for (const auto& column : columns) {
        const int count = static_cast<int>(column.second.size());

        for (int row = 0; row < count; row++) {
            // Squeezed in to 0 to 1 across, so that it fits the bounds however deep it gets.
            where[column.second[row]] = {column.first / (deepest + 1.0), count == 1 ? 0.0 : 1 - 2.0 * row / (count - 1)};
        }
    }

    plotato::PlotStyle edge_style;
    edge_style.line_color = plotato::Color(120, 120, 120);
    edge_style.line_width = 1;

    for (const auto& entry : where) {
        for (int source : sources[entry.first]) {
            graph.plot({where[source].x, entry.second.x}, {where[source].y, entry.second.y}, edge_style);
        }
    }

    // Markers rather than circles, since this graph isn't drawn to the same scale both ways.
    plotato::MarkerStyle node_style;
    node_style.fill = family_color(agent.family);
    node_style.outline = plotato::Color(0, 0, 0);
    node_style.outline_width = 1;
    node_style.size = 10;

    plotato::FigureTextStyle label_style;
    label_style.font_size = 10;
    label_style.absolute = false;
    label_style.anchor = plotato::Anchor::TOP_CENTER;
    label_style.background_color = plotato::Color(0, 0, 0, 0);
    label_style.outline_color = plotato::Color(0, 0, 0, 0);
    label_style.outline_width = 0;

    for (const auto& entry : where) {
        const cdn::GraphNode& node = net.nodes[entry.first];

        std::string label;

        if (node.kind == cdn::NodeKind::Input) {
            label = "inputs";
        } else if (node.kind == cdn::NodeKind::Output) {
            label = node.name.substr(node.name.find(':') + 1);
        } else {
            label = std::to_string(node.hp.i("units", 32)) + " " + node.hp.s("activation", "relu");
        }

        graph.scatter({entry.second.x}, {entry.second.y}, node_style);
        graph.add_text(entry.second.x, entry.second.y - .12, label, label_style);
    }

    char caption[160];

    std::snprintf(caption, sizeof(caption), "Agent %d, generation %d, %d children, %.0f s old, %zu parameters, learning rate %.2g",
                  agent.id, agent.generation, agent.children, agent.age, agent.brain->parameter_count(), agent.brain->learning_rate());

    plotato::FigureTextStyle caption_style;
    caption_style.font_size = 11;
    caption_style.absolute = false;
    caption_style.anchor = plotato::Anchor::BOTTOM_LEFT;
    caption_style.background_color = plotato::Color(0, 0, 0, 0);
    caption_style.outline_color = plotato::Color(0, 0, 0, 0);
    caption_style.outline_width = 0;

    graph.add_text(0, 1.3, caption, caption_style);
}

}

std::unique_ptr<sim::Project> sim::create_project(sim::Ui& ui, uint64_t seed)
{
    return std::make_unique<island_evolution::IslandEvolution>(ui, seed);
}
