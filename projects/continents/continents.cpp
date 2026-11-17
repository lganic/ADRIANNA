#include "continents.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <exception>
#include <limits>
#include <map>
#include <set>
#include <string>
#include <thread>
#include <utility>

namespace continents {

// Where the right thruster sits on a body of radius 1, relative to its middle.
constexpr Point RIGHT_LEG = {ORGANISM_SIZE * RIGHT_FOOT.x, ORGANISM_SIZE * (RIGHT_FOOT.y - DRAWING_CENTER_Y)};

// Box2D only gets told about a change in size once it adds up to this fraction of the radius.
constexpr double RESIZE_STEP = .02;

constexpr double CONTACT_CELL = 8; // Meters across each bucket of the grid used for fighting.

// Thrust past this doesn't make the flames any bigger, so that a wild output can't fill the screen.
constexpr double MAX_FLAME_BOOST = 1;

// The bars drawn over each agent, in meters.
constexpr double BAR_WIDTH = 3;
constexpr double BAR_HEIGHT = .5;
constexpr double BAR_GAP = 1; // From the top of the agent up to the lowest bar.

// The colour of each channel, for the rings drawn around agents that are calling.
const plotato::Color CHANNEL_COLORS[CHANNELS] = {
    plotato::Color(220, 40, 40),
    plotato::Color(30, 160, 30),
    plotato::Color(40, 80, 230),
    plotato::Color(190, 40, 190),
    plotato::Color(240, 150, 0),
};

int Continents::default_pool_count()
{
    // See projects/island_evolution for why it is cores rather than hardware threads.
    return std::max(1, static_cast<int>(std::thread::hardware_concurrency() / 2));
}

Continents::Continents(sim::Ui& ui, uint64_t seed, int pool_count)
    : terrain(HALF_WIDTH, HALF_HEIGHT, MAP_SEED),
      world(terrain, FOOD_RADIUS, RIGHT_LEG, seed),
      rng(seed),
      spots(rng.next_u64()),
      pools(static_cast<std::size_t>(std::max(1, pool_count))),
      census(HALF_WIDTH, HALF_HEIGHT),
      contacts(HALF_WIDTH, HALF_HEIGHT, CONTACT_CELL),
      threads(std::max(1, pool_count)),
      graphic(plotato::Color(0, 0, 255), ORGANISM_SIZE),
      food_growth(ui.add_slider({"Food growth (pieces per second)", 0, 50, 10, .5})),
      max_food(ui.add_slider({"Most plants at once", 100, 10000, 3000, 50})),
      meat_lifetime(ui.add_slider({"Seconds before meat rots away", 5, 600, 120, 5})),
      food_energy(ui.add_slider({"Energy per food", 0, 5, 1, .05})),
      metabolism(ui.add_slider({"Energy used per second, per unit of mass", 0, .5, .02, .005})),
      thrust_energy(ui.add_slider({"Energy per second per unit of thrust", 0, .05, .002, .0005})),
      cold_energy(ui.add_slider({"Extra energy per second at the poles", 0, 1, .1, .01})),
      start_energy(ui.add_slider({"Starting energy", 0, 20, 5, .5})),
      reproduction_cost(ui.add_slider({"Energy to reproduce", .5, 20, 5, .5})),
      reproduction_time(ui.add_slider({"Seconds to reproduce at full effort", .2, 20, 1, .1})),
      reproduction_decay(ui.add_slider({"Progress lost per second when not trying", 0, 2, .2, .01})),
      growth_rate(ui.add_slider({"Mass gained per second with a full bar", 0, .2, .02, .005})),
      attack_damage(ui.add_slider({"Bite damage per second, per unit of mass", 0, 50, 5, .5})),
      attack_energy(ui.add_slider({"Energy per second spent biting, per unit of mass", 0, 1, .02, .005})),
      bite_drain(ui.add_slider({"Energy a bite takes, per unit of damage", 0, 5, 1, .05})),
      healing(ui.add_slider({"Fraction of health healed per second", 0, .5, .05, .01})),
      hold_strength(ui.add_slider({"Holding strength, per unit of mass", 0, 500, 100, 5})),
      hold_energy(ui.add_slider({"Energy per second spent holding on, per unit of mass", 0, 1, .01, .005})),
      call_energy(ui.add_slider({"Energy per second for calling on a channel", 0, .2, .002, .001})),
      marking_drift(ui.add_slider({"How far markings drift with each child", 0, .1, .02, .005})),
      max_agents(ui.add_slider({"Most agents at once", 1, 2000, 500, 10})),
      mutation_rate(ui.add_slider({"Mutation rate", 0, 5, 1, .1})),
      food_reward(ui.add_entry({"Food reward", 0, 100, 10, 1})),
      closeness_reward(ui.add_slider({"Closeness reward", 0, 5, 1, .1})),
      water_punish(ui.add_slider({"Water punishment", 0, 20, 5, .5})),
      water_friction(ui.add_slider({"Water friction while settling", 0, 10, 2, .1})),
      settled_friction(ui.add_slider({"Water friction once every landmass is settled", 0, 10, 8, .1})),
      wall_punish(ui.add_slider({"Wall punishment", 0, 20, 5, .5})),
      effort_punish(ui.add_slider({"Effort punishment", 0, 1, .1, .01})),
      spin_punish(ui.add_slider({"Spin punishment", 0, 1, .2, .01})),
      energy_punish(ui.add_slider({"Punishment per unit of energy spent", 0, 50, 10, .5})),
      hurt_punish(ui.add_slider({"Punishment per unit of damage taken", 0, 10, 1, .1})),
      attack_reward(ui.add_slider({"Reward per unit of damage done", 0, 10, 0, .1})),
      kin_share(ui.add_slider({"Share of family's food reward", 0, 1, 0, .05})),
      kin_harm(ui.add_slider({"Share of the harm done to family that the biter feels", 0, 3, 0, .05})),
      lambda(ui.add_slider({"How much later rewards count when learning (lambda)", 0, 1, 1, .05})),
      bite_noise(ui.add_slider({"Most randomness in biting (log of its spread)", -2, 1, 1, .1})),
      bite_focus(ui.add_slider({"How much a bite is judged by just what it got straight away", 0, 1, 0, .05})),
      network_graph(ui.add_live_graph({"Network of the featured agent", "", sim::Page::World, "", true,
                                       [](plotato::Graph& graph) { graph.set_bounds({-.1, 1.1, -1.5, 1.6}); }})),
      population_field(ui.add_field({"Population", sim::Page::World, "%.0f"})),
      births_field(ui.add_field({"Births", sim::Page::World, "%.0f"})),
      kills_field(ui.add_field({"Killed", sim::Page::World, "%.0f"})),
      starved_field(ui.add_field({"Starved", sim::Page::World, "%.0f"})),
      extinction_field(ui.add_field({"Extinctions", sim::Page::World, "%.0f"})),
      generation_field(ui.add_field({"Highest generation alive", sim::Page::World, "%.0f"})),
      record_field(ui.add_field({"Most children by one agent", sim::Page::World, "%.0f"})),
      landmass_field(ui.add_field({"Landmasses with agents on them", sim::Page::World, "%.0f"})),
      friction_field(ui.add_field({"Water friction now", sim::Page::World, "%.1f"})),
      biggest_field(ui.add_field({"Biggest radius alive", sim::Page::World, "%.2f"})),
      species_field(ui.add_field({"Module species alive", sim::Page::World, "%.0f"})),
      layers_field(ui.add_field({"Most layers alive", sim::Page::World, "%.0f"})),
      population_graph(ui.add_history_graph({"Population Graph", "Agents alive"})),
      plant_graph(ui.add_history_graph({"Plant Graph", "Plants eaten per agent per hour"})),
      meat_graph(ui.add_history_graph({"Meat Graph", "Meat eaten per agent per hour"})),
      hunting_graph(ui.add_history_graph({"Hunting Graph", "Fraction of energy eaten that came from meat or biting"})),
      birth_graph(ui.add_history_graph({"Birth Graph", "Births per hour"})),
      kill_graph(ui.add_history_graph({"Kill Graph", "Agents killed per hour"})),
      starve_graph(ui.add_history_graph({"Starve Graph", "Agents starved per hour"})),
      size_graph(ui.add_history_graph({"Size Graph", "Average radius"})),
      energy_graph(ui.add_history_graph({"Energy Graph", "Average fraction of the energy bar"})),
      attacking_graph(ui.add_history_graph({"Attacking Graph", "Fraction of agents biting"})),
      holding_graph(ui.add_history_graph({"Holding Graph", "Fraction of agents holding on to another"})),
      calling_graph(ui.add_history_graph({"Calling Graph", "Average call, over every channel"})),
      bitten_kin_graph(ui.add_history_graph({"Bitten Kinship Graph", "Kinship between biter and bitten, by damage"})),
      neighbour_kin_graph(ui.add_history_graph({"Neighbour Kinship Graph", "Kinship with the nearest other agent"})),
      water_graph(ui.add_history_graph({"Water Graph", "Fraction of agents in water"})),
      cold_graph(ui.add_history_graph({"Cold Graph", "Fraction of agents where it is more than half cold"})),
      reproducing_graph(ui.add_history_graph({"Reproducing Graph", "Fraction of agents trying to reproduce"})),
      generation_graph(ui.add_history_graph({"Generation Graph", "Average generation"})),
      layers_graph(ui.add_history_graph({"Layers Graph", "Average hidden layers"})),
      memory_graph(ui.add_history_graph({"Memory Graph", "Average LSTM layers"})),
      parameters_graph(ui.add_history_graph({"Parameters Graph", "Average parameters"})),
      learning_rate_graph(ui.add_history_graph({"Learning Rate Graph", "Average learning rate"})),
      eye_graph(ui.add_history_graph({"Eye Graph", "Average degrees from the middle eye to the outer ones"})),
      score_graph(ui.add_history_graph({"Score Graph", "Average score"}))
{
    const Settings settings = read_settings();

    // The plants start out fully grown. Otherwise the founders, which can't do anything sensible
    // yet, have to wait minutes for there to be anything to find.
    world.grow_food(settings.max_food, static_cast<std::size_t>(settings.max_food), 1);

    for (int founder = 0; founder < FOUNDERS; founder++) {
        add_founder(settings);
    }

    update_order();
}

double Continents::time_step() const
{
    return DT;
}

void Continents::setup_world(plotato::Graph& graph)
{
    graph.equal_aspect_framing = true;
    graph.set_bounds({-HALF_WIDTH * 1.05, HALF_WIDTH * 1.05, -HALF_HEIGHT * 1.05, HALF_HEIGHT * 1.05});
}

Settings Continents::read_settings() const
{
    Settings settings;

    settings.food_growth = food_growth.value();
    settings.max_food = max_food.value();
    settings.meat_lifetime = meat_lifetime.value();
    settings.food_energy = food_energy.value();
    settings.metabolism = metabolism.value();
    settings.thrust_energy = thrust_energy.value();
    settings.cold_energy = cold_energy.value();
    settings.start_energy = start_energy.value();
    settings.reproduction_cost = reproduction_cost.value();
    settings.reproduction_time = reproduction_time.value();
    settings.reproduction_decay = reproduction_decay.value();
    settings.growth_rate = growth_rate.value();
    settings.attack_damage = attack_damage.value();
    settings.attack_energy = attack_energy.value();
    settings.bite_drain = bite_drain.value();
    settings.healing = healing.value();
    settings.hold_strength = hold_strength.value();
    settings.hold_energy = hold_energy.value();
    settings.call_energy = call_energy.value();
    settings.marking_drift = marking_drift.value();
    settings.max_agents = max_agents.value();
    settings.mutation_rate = mutation_rate.value();

    settings.food_reward = food_reward.value();
    settings.closeness_reward = closeness_reward.value();
    settings.water_punish = water_punish.value();
    settings.water_friction = settled ? settled_friction.value() : water_friction.value();
    settings.wall_punish = wall_punish.value();
    settings.effort_punish = effort_punish.value();
    settings.spin_punish = spin_punish.value();
    settings.energy_punish = energy_punish.value();
    settings.hurt_punish = hurt_punish.value();
    settings.attack_reward = attack_reward.value();
    settings.kin_share = kin_share.value();
    settings.kin_harm = kin_harm.value();
    settings.lambda = lambda.value();
    settings.bite_noise = bite_noise.value();
    settings.bite_focus = bite_focus.value();

    return settings;
}

void Continents::add_founder(const Settings& settings)
{
    Agent agent;

    agent.id = next_id++;
    agent.mass = 1;
    agent.body = world.add_body(terrain.random_land(spots), rng.uniform(-PI, PI), agent.radius());
    agent.family = agent.id;
    agent.energy = settings.start_energy;
    agent.health = agent.max_health();

    // Founders are unrelated, so they all look different.
    for (double& marking : agent.markings) {
        marking = rng.uniform(0, 1);
    }

    agent.brain = std::make_unique<Brain>(genetics.founder(rng), genetics.io(), rng.next_u64(), nullptr);

    place(std::move(agent));
}

void Continents::place(Agent agent)
{
    // On a tie, the lowest numbered one. That way pool 0, which the main thread runs itself, is never left empty.
    const auto emptiest = std::min_element(pools.begin(), pools.end(), [](const AgentPool& a, const AgentPool& b) { return a.agents.size() < b.agents.size(); });

    emptiest->agents.push_back(std::move(agent));
}

void Continents::update_order()
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
void Continents::run_pools(bool slowest_first, Work work)
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

std::unique_ptr<Brain> Continents::breed(const Heritage& parent, const Settings& settings)
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

Markings Continents::drift(const Markings& parent, double amount)
{
    Markings child = parent;

    for (double& marking : child) {
        marking += amount * rng.normal();

        // Bounced back off of the ends, so that they don't pile up there.
        marking = std::abs(marking);
        marking = marking > 1 ? 2 - marking : marking;
        marking = clamp(marking, 0, 1);
    }

    return child;
}

void Continents::take_census()
{
    census.agents.resize(order.size());

    for (std::size_t slot = 0; slot < order.size(); slot++) {
        Agent& agent = *order[slot];
        const Body& body = world.body(agent.body);

        agent.slot = static_cast<int>(slot);

        Seen& seen = census.agents[slot];

        seen.pos = body.pos;
        seen.rotation = body.rotation;
        seen.radius = body.radius;
        seen.markings = agent.markings;
        seen.calls = agent.calls;
        seen.eaten = 0;
    }

    census.grid.build(static_cast<int>(order.size()), [this](int slot) { return census.agents[slot].pos; });
}

Agent* Continents::in_reach(const Agent& agent, const Grid& grid, double reach, double angle)
{
    const Body& self = world.body(agent.body);

    const double reach_distance = reach * self.radius;
    const double search = self.radius + std::sqrt(MAX_MASS) + reach_distance;

    Agent* best = nullptr;
    double best_gap = std::numeric_limits<double>::infinity();

    grid.visit(self.pos, search, [&](int slot) {
        Agent* other = order[slot];

        if (other == &agent) {
            return;
        }

        const Body& body = world.body(other->body);

        const Point offset = {body.pos.x - self.pos.x, body.pos.y - self.pos.y};
        const double gap = std::hypot(offset.x, offset.y) - self.radius - body.radius;

        if (gap > reach_distance || std::abs(bearing(self.rotation, offset)) > angle) {
            return;
        }

        if (gap < best_gap) {
            best = other;
            best_gap = gap;
        }
    });

    return best;
}

void Continents::update_holds()
{
    // The census was taken at the start of this step, and nothing has moved since.
    for (Agent* agent : order) {
        const Body& body = world.body(agent->body);

        if (!agent->hold) {
            if (body.holding >= 0) {
                world.release(agent->body);
            }

            continue;
        }

        if (body.holding >= 0) {
            continue;
        }

        if (Agent* target = in_reach(*agent, census.grid, HOLD_REACH, HOLD_ANGLE)) {
            world.hold(agent->body, target->body);
        }
    }
}

void Continents::fight(const Settings& settings)
{
    contacts.build(static_cast<int>(order.size()), [this](int slot) { return world.body(order[slot]->body).pos; });

    for (Agent* agent : order) {
        agent->hurt = 0;
        agent->dealt = 0;
        agent->fed = 0;
        agent->family_hurt = 0;
        agent->family_taken = 0;
        agent->could_bite = false;
    }

    step_damage = 0;
    step_family_damage = 0;

    // In order of id, so that it comes out the same every time, although the order makes no
    // difference anyway, since nobody dies until everybody has had their bite.
    for (Agent* attacker : order) {
        // Whether or not it bites, since learning to bite needs to know when it could have.
        Agent* victim = in_reach(*attacker, contacts, BITE_REACH, BITE_ANGLE);

        attacker->could_bite = victim != nullptr;

        if (attacker->attack <= 0 || !victim) {
            continue;
        }

        const double damage = DT * settings.attack_damage * attacker->attack * attacker->mass;

        victim->health -= damage;
        victim->hurt += damage;
        victim->last_attacker = attacker->id;
        attacker->dealt += damage;

        // Eating while biting. The biter only gets the same share of it as it would out of meat,
        // so that nothing comes out ahead by raising children to feed on.
        const double taken = std::min(std::max(0.0, victim->energy), settings.bite_drain * damage);

        victim->energy -= taken;
        attacker->fed += CORPSE_SHARE * taken;

        const double related = kinship(attacker->markings, victim->markings);

        attacker->family_hurt += related * damage;
        attacker->family_taken += related * taken;

        step_damage += damage;
        step_family_damage += related * damage;
        total_damage += damage;
        total_family_damage += related * damage;
    }
}

// Food is what children are made of, so the most of it over a lifetime is what makes the best agent.
// See projects/island_evolution for why it isn't children.
static bool beats(double eaten, int children, const Record& record)
{
    return eaten > record.eaten || (eaten == record.eaten && children > record.children);
}

void Continents::step(double)
{
    const Settings settings = read_settings();

    const double cost = settings.reproduction_cost;

    world.grow_food(settings.food_growth, static_cast<std::size_t>(settings.max_food), DT);
    world.rot_meat(settings.meat_lifetime, DT);
    world.index_food();

    take_census();

    // Decide what to do, all at once, and then do it.
    run_pools(true, [&](Agent& agent) { think(agent, world, census, settings); });

    for (const Agent* agent : order) {
        Body& body = world.body(agent->body);

        body.left_thrust = agent->left_thrust;
        body.right_thrust = agent->right_thrust;
    }

    update_holds();

    world.sim(settings.water_friction, DT);
    world.break_holds(settings.hold_strength);

    fight(settings);

    for (const Agent* agent : order) {
        const Body& body = world.body(agent->body);

        census.agents[agent->slot].eaten = body.eaten_plants + body.eaten_meat + (settings.food_energy > 0 ? agent->fed / settings.food_energy : 0);
    }

    // See what came of it.
    run_pools(false, [&](Agent& agent) { score(agent, world, census, settings); });

    std::vector<Agent> born;

    int births = 0;
    int plants = 0;
    int meat = 0;
    double bitten = 0; // Energy, where plants and meat are pieces.
    int in_water = 0;
    int in_cold = 0;
    int reproducing = 0;
    int attacking = 0;
    int holding = 0;
    double calling = 0;
    double total_score = 0;
    double total_radius = 0;
    double total_fill = 0;
    double neighbour_kinship = 0;
    int with_neighbour = 0;

    const std::size_t alive = order.size();

    for (Agent* agent : order) {
        const Body& body = world.body(agent->body);

        // With the population full, a finished agent waits on 1 until there is room.
        if (agent->progress >= 1 && agent->energy > cost && static_cast<double>(alive + born.size()) < settings.max_agents) {
            agent->energy -= cost;
            agent->progress = 0;
            agent->children++;

            most_children = std::max(most_children, agent->children);

            Agent child;

            child.id = next_id++;
            child.mass = BIRTH_MASS;

            // Just behind the parent, facing the other way, so that neither of them starts out
            // looking at, or biting, the other.
            const double radius = child.radius();
            const Point forward = heading(body.rotation);
            const double gap = body.radius + radius + .2;

            const Point spot = {
                clamp(body.pos.x - forward.x * gap, -HALF_WIDTH + radius, HALF_WIDTH - radius),
                clamp(body.pos.y - forward.y * gap, -HALF_HEIGHT + radius, HALF_HEIGHT - radius),
            };

            child.body = world.add_body(spot, body.rotation + PI, radius);
            child.generation = agent->generation + 1;
            child.family = agent->family;
            child.energy = std::max(.1, CHILD_SHARE * cost - BODY_ENERGY * BIRTH_MASS);
            child.health = child.max_health();
            child.markings = drift(agent->markings, settings.marking_drift);

            child.brain = breed(agent->brain->heritage(), settings);

            born.push_back(std::move(child));

            births++;
        }

        plants += body.eaten_plants;
        meat += body.eaten_meat;
        bitten += agent->fed;
        in_water += agent->wet ? 1 : 0;
        in_cold += agent->cold > .5 ? 1 : 0;
        reproducing += agent->effort > 0 ? 1 : 0;
        attacking += agent->attack > .1 ? 1 : 0;
        holding += body.holding >= 0 ? 1 : 0;
        total_score += agent->score;
        total_radius += body.radius;
        total_fill += clamp(agent->energy / agent->capacity(cost), 0, 1);

        for (double call : agent->calls) {
            calling += call / CHANNELS;
        }

        // How related the closest other agent is, to see whether families stick together.
        double closest = EYE_RANGE;
        const Agent* neighbour = nullptr;

        contacts.visit(body.pos, EYE_RANGE, [&](int slot) {
            const Agent* other = order[slot];

            if (other == agent) {
                return;
            }

            const Point& pos = world.body(other->body).pos;
            const double distance = std::hypot(pos.x - body.pos.x, pos.y - body.pos.y);

            if (distance < closest) {
                closest = distance;
                neighbour = other;
            }
        });

        if (neighbour) {
            neighbour_kinship += kinship(agent->markings, neighbour->markings);
            with_neighbour++;
        }
    }

    // Out of energy, or out of health, is out of time. The last score never gets learned from, since
    // there is no next step. Whatever was left in it gets dropped as meat.
    int kills = 0;
    int starved = 0;

    for (Agent* agent : order) {
        // Bled dry by a bite counts as killed too, not starved.
        const bool killed = agent->health <= 0 || (agent->energy <= 0 && agent->hurt > 0);

        if (!killed && agent->energy > 0) {
            continue;
        }

        if (killed) {
            kills++;

            if (agent->age >= GROWN_AGE) {
                total_grown_kills++;
            }

            const auto killer = std::lower_bound(order.begin(), order.end(), agent->last_attacker, [](const Agent* a, int id) { return a->id < id; });

            if (killer != order.end() && (*killer)->id == agent->last_attacker) {
                (*killer)->kills++;
                total_kill_kinship += kinship(agent->markings, (*killer)->markings);
            }
        } else {
            starved++;
        }

        if (!has_record || beats(agent->eaten, agent->children, record)) {
            record = {agent->brain->heritage(), agent->markings, agent->generation, agent->family, agent->children, agent->eaten};
            has_record = true;
        }

        const Body& body = world.body(agent->body);

        const double left = CORPSE_SHARE * (std::max(0.0, agent->energy) + BODY_ENERGY * agent->mass);
        const int pieces = settings.food_energy > 0 ? std::min(MOST_MEAT, static_cast<int>(left / settings.food_energy)) : 0;

        world.drop_meat(body.pos, body.radius + 2, pieces);
        world.remove_body(agent->body);

        agent->brain.reset();
    }

    if (kills + starved > 0) {
        for (AgentPool& pool : pools) {
            pool.agents.erase(std::remove_if(pool.agents.begin(), pool.agents.end(), [](const Agent& agent) { return !agent.brain; }), pool.agents.end());
        }
    }

    for (Agent& child : born) {
        place(std::move(child));
    }

    const bool everybody_gone = std::all_of(pools.begin(), pools.end(), [](const AgentPool& pool) { return pool.agents.empty(); });

    // Everybody is gone, so the best there ever was gets to start over, with children of its own.
    if (everybody_gone && has_record) {
        extinctions++;
        settled = false;

        for (int restart = 0; restart < RESTARTERS; restart++) {
            Agent agent;

            agent.id = next_id++;
            agent.mass = 1;
            agent.body = world.add_body(terrain.random_land(spots), rng.uniform(-PI, PI), agent.radius());
            agent.generation = record.generation + 1;
            agent.family = record.family;
            agent.energy = settings.start_energy;
            agent.health = agent.max_health();
            agent.markings = drift(record.markings, settings.marking_drift);
            agent.brain = breed(record.heritage, settings);

            place(std::move(agent));
        }
    }

    // The pools moved agents around in memory, if anybody came or went.
    if (kills + starved > 0 || births > 0 || everybody_gone) {
        update_order();
    }

    if (!settled) {
        std::vector<int> settlers(static_cast<std::size_t>(terrain.landmass_count()), 0);

        // Founders don't count, since they get spread out over all of the land to start with.
        for (const Agent* agent : order) {
            const int landmass = terrain.landmass(world.body(agent->body).pos);

            if (landmass >= 0 && agent->generation > 0) {
                settlers[static_cast<std::size_t>(landmass)]++;
            }
        }

        settled = std::all_of(settlers.begin(), settlers.end(), [](int count) { return count >= SETTLERS; });
    }

    for (const Agent* agent : order) {
        const double radius = agent->radius();
        const Body& body = world.body(agent->body);

        if (std::abs(radius - body.radius) > RESIZE_STEP * body.radius) {
            world.set_radius(agent->body, radius);
        }
    }

    total_births += static_cast<uint64_t>(births);
    total_kills += static_cast<uint64_t>(kills);
    total_starved += static_cast<uint64_t>(starved);
    total_plants += static_cast<uint64_t>(plants);
    total_meat += static_cast<uint64_t>(meat);
    total_bitten += bitten;

    constexpr double PER_HOUR = 3600 / DT;

    population_graph.add(static_cast<double>(order.size()));
    birth_graph.add(births * PER_HOUR);
    kill_graph.add(kills * PER_HOUR);
    starve_graph.add(starved * PER_HOUR);

    if (step_damage > 0) {
        bitten_kin_graph.add(step_family_damage / step_damage);
    }

    if (with_neighbour > 0) {
        neighbour_kin_graph.add(neighbour_kinship / with_neighbour);
    }

    if (alive > 0) {
        const double count = static_cast<double>(alive);

        plant_graph.add(plants * PER_HOUR / count);
        meat_graph.add(meat * PER_HOUR / count);

        const double from_food = (plants + meat) * settings.food_energy;

        if (from_food + bitten > 0) {
            hunting_graph.add((meat * settings.food_energy + bitten) / (from_food + bitten));
        }
        water_graph.add(in_water / count);
        cold_graph.add(in_cold / count);
        reproducing_graph.add(reproducing / count);
        attacking_graph.add(attacking / count);
        holding_graph.add(holding / count);
        calling_graph.add(calling / count);
        size_graph.add(total_radius / count);
        energy_graph.add(total_fill / count);
        score_graph.add(total_score / count);
    }

    if (!order.empty()) {
        double generation = 0;
        double layers = 0;
        double memories = 0;
        double parameters = 0;
        double learning_rate = 0;
        double eyes = 0;

        for (const Agent* agent : order) {
            generation += agent->generation;
            layers += agent->brain->layer_count();
            memories += agent->brain->memory_count();
            parameters += static_cast<double>(agent->brain->parameter_count());
            learning_rate += agent->brain->learning_rate();
            eyes += agent->brain->eye_spread();
        }

        const double count = static_cast<double>(order.size());

        generation_graph.add(generation / count);
        layers_graph.add(layers / count);
        memory_graph.add(memories / count);
        parameters_graph.add(parameters / count);
        learning_rate_graph.add(learning_rate / count);
        eye_graph.add(eyes / count * 180 / PI);
    }
}

const Agent* Continents::featured() const
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

static plotato::Color marking_color(const Markings& markings, uint8_t alpha = 255)
{
    const auto channel = [](double value) { return static_cast<uint8_t>(30 + 200 * clamp(value, 0, 1)); };

    return plotato::Color(channel(markings[0]), channel(markings[1]), channel(markings[2]), alpha);
}

static plotato::Color blend(plotato::Color a, plotato::Color b, double t)
{
    const auto mix = [t](uint8_t x, uint8_t y) { return static_cast<uint8_t>(x + (y - x) * clamp(t, 0, 1)); };

    return plotato::Color(mix(a.r, b.r), mix(a.g, b.g), mix(a.b, b.b), mix(a.a, b.a));
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
    const double bottom = position.y + offset;
    const double top = bottom + BAR_HEIGHT;

    graph.polygon({left, left, left + BAR_WIDTH, left + BAR_WIDTH}, {bottom, top, top, bottom}, back_style);

    const double right = left + BAR_WIDTH * clamp(fraction, 0, 1);

    if (right > left) {
        graph.polygon({left, left, right, right}, {bottom, top, top, bottom}, fill_style);
    }
}

void Continents::draw(plotato::Graph& graph)
{
    const double width = HALF_WIDTH;
    const double height = HALF_HEIGHT;

    plotato::ShapeStyle water_style;
    water_style.fill = plotato::Color(170, 215, 245);
    water_style.outline = plotato::Color(0, 0, 0, 0); // No outline.

    plotato::PlotStyle wall_style;
    wall_style.line_color = plotato::Color(0, 0, 255);

    graph.polygon({-width, -width, width, width}, {-height, height, height, -height}, water_style);

    // The sea gets paler towards the poles, in bands.
    constexpr double BAND = 25;

    for (double y = 0; y < height; y += BAND) {
        const double cold = terrain.cold(y + BAND / 2);

        if (cold <= 0) {
            continue;
        }

        plotato::ShapeStyle band_style;
        band_style.fill = plotato::Color(255, 255, 255, static_cast<uint8_t>(140 * cold));
        band_style.outline = plotato::Color(0, 0, 0, 0);

        const double top = std::min(height, y + BAND);

        graph.polygon({-width, -width, width, width}, {y, top, top, y}, band_style);
        graph.polygon({-width, -width, width, width}, {-top, -y, -y, -top}, band_style);
    }

    const plotato::Color warm_land(226, 206, 150);
    const plotato::Color frozen_land(246, 248, 252);

    for (const LandRun& run : terrain.runs()) {
        plotato::ShapeStyle land_style;
        land_style.fill = blend(warm_land, frozen_land, std::pow(run.cold, 1.5));
        land_style.outline = plotato::Color(0, 0, 0, 0);

        graph.polygon({run.left, run.left, run.right, run.right}, {run.bottom, run.top, run.top, run.bottom}, land_style);
    }

    graph.plot({-width, -width, width, width, -width}, {-height, height, height, -height, -height}, wall_style);

    // Markers, rather than circles, so that food can still be seen with the whole map on screen.
    std::vector<double> plant_x;
    std::vector<double> plant_y;
    std::vector<double> meat_x;
    std::vector<double> meat_y;

    for (const Food& piece : world.food) {
        (piece.meat ? meat_x : plant_x).push_back(piece.pos.x);
        (piece.meat ? meat_y : plant_y).push_back(piece.pos.y);
    }

    plotato::MarkerStyle plant_style;
    plant_style.fill = plotato::Color(60, 150, 40);
    plant_style.outline = plant_style.fill;
    plant_style.outline_width = 0;
    plant_style.size = 3;

    plotato::MarkerStyle meat_style;
    meat_style.fill = plotato::Color(150, 20, 20);
    meat_style.outline = meat_style.fill;
    meat_style.outline_width = 0;
    meat_style.size = 3;

    if (!plant_x.empty()) {
        graph.scatter(plant_x, plant_y, plant_style);
    }

    if (!meat_x.empty()) {
        graph.scatter(meat_x, meat_y, meat_style);
    }

    plotato::PlotStyle hold_style;
    hold_style.line_color = plotato::Color(40, 40, 40);
    hold_style.line_width = 2;

    plotato::ShapeStyle bite_style;
    bite_style.outline = plotato::Color(0, 0, 0, 0);

    const double cost = reproduction_cost.value();

    for (const Agent* agent_pointer : order) {
        const Agent& agent = *agent_pointer;
        const Body& body = world.body(agent.body);

        const plotato::Color color = marking_color(agent.markings);
        const Point forward = heading(body.rotation);

        // A dot underneath, which is all that can be seen of it with the whole map on screen.
        plotato::MarkerStyle dot_style;
        dot_style.fill = color;
        dot_style.outline = plotato::Color(0, 0, 0);
        dot_style.outline_width = 1;
        dot_style.size = 5;

        graph.scatter({body.pos.x}, {body.pos.y}, dot_style);

        graphic.set_color(color);
        graphic.set_size(ORGANISM_SIZE * body.radius);
        graphic.draw(graph, body.pos, body.rotation, flame_boost(agent.left_thrust), flame_boost(agent.right_thrust));

        // Jaws, when it is biting.
        if (agent.attack > .2) {
            const Point mouth = {body.pos.x + forward.x * body.radius, body.pos.y + forward.y * body.radius};
            const Point side = {-forward.y * .5 * body.radius, forward.x * .5 * body.radius};
            const double reach = .6 * body.radius;

            bite_style.fill = plotato::Color(220, 0, 0, static_cast<uint8_t>(255 * agent.attack));

            graph.polygon({mouth.x + side.x, mouth.x + forward.x * reach, mouth.x - side.x}, {mouth.y + side.y, mouth.y + forward.y * reach, mouth.y - side.y}, bite_style);
        }

        if (body.holding >= 0) {
            const Body& target = world.body(body.holding);

            graph.plot({body.pos.x + forward.x * body.radius, target.pos.x}, {body.pos.y + forward.y * body.radius, target.pos.y}, hold_style);
        }

        // A ring for whichever channel it is calling loudest on, bigger the louder it is.
        const auto loudest = std::max_element(agent.calls.begin(), agent.calls.end());

        if (*loudest > .3) {
            plotato::ShapeStyle ring_style;
            ring_style.fill = plotato::Color(0, 0, 0, 0);
            ring_style.outline = CHANNEL_COLORS[loudest - agent.calls.begin()];
            ring_style.outline_width = 1.5;

            graph.circle(body.pos.x, body.pos.y, body.radius * (1.5 + 2 * *loudest), ring_style);
        }

        // Energy, with the line for having enough to reproduce partway along it, and health on top.
        const double fill = agent.energy / agent.capacity(cost);
        const double bars = body.radius + BAR_GAP;

        draw_bar(graph, body.pos, bars, fill, agent.energy > cost ? plotato::Color(0, 170, 0) : plotato::Color(220, 160, 0));
        draw_bar(graph, body.pos, bars + BAR_HEIGHT * 1.5, agent.health / agent.max_health(), plotato::Color(210, 40, 40));
    }

    const Agent* chosen = featured();

    if (chosen) {
        const Body& body = world.body(chosen->body);

        plotato::ShapeStyle featured_style;
        featured_style.fill = plotato::Color(0, 0, 0, 0);
        featured_style.outline = plotato::Color(200, 0, 0);

        graph.circle(body.pos.x, body.pos.y, body.radius * 3, featured_style);

        draw_network(network_graph.graph(), *chosen);
    }

    // Fields
    std::set<int> species;
    std::set<int> landmasses;
    int most_layers = 0;
    int highest_generation = 0;
    double biggest = 0;

    for (const Agent* agent : order) {
        for (const auto& entry : agent->brain->genome().modules) {
            species.insert(entry.first);
        }

        const Body& body = world.body(agent->body);
        const int landmass = terrain.landmass(body.pos);

        if (landmass >= 0) {
            landmasses.insert(landmass);
        }

        most_layers = std::max(most_layers, agent->brain->layer_count());
        highest_generation = std::max(highest_generation, agent->generation);
        biggest = std::max(biggest, body.radius);
    }

    population_field.set(static_cast<double>(order.size()));
    births_field.set(static_cast<double>(total_births));
    kills_field.set(static_cast<double>(total_kills));
    starved_field.set(static_cast<double>(total_starved));
    extinction_field.set(static_cast<double>(extinctions));
    generation_field.set(highest_generation);
    record_field.set(most_children);
    landmass_field.set(static_cast<double>(landmasses.size()));
    friction_field.set(settled ? settled_friction.value() : water_friction.value());
    biggest_field.set(biggest);
    species_field.set(static_cast<double>(species.size()));
    layers_field.set(most_layers);
}

// Lay the network out left to right, by how many layers deep each one is. The identity nodes that
// CoDeepNEAT puts at the edges of modules, and where things merge, are left out, with their
// connections drawn straight through them. The same as projects/island_evolution.
void Continents::draw_network(plotato::Graph& graph, const Agent& agent)
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
    node_style.fill = marking_color(agent.markings);
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
        } else if (node.layer_type == "lstm") {
            label = std::to_string(node.hp.i("units", 32)) + " lstm";
        } else {
            label = std::to_string(node.hp.i("units", 32)) + " " + node.hp.s("activation", "relu");
        }

        graph.scatter({entry.second.x}, {entry.second.y}, node_style);
        graph.add_text(entry.second.x, entry.second.y - .12, label, label_style);
    }

    char caption[200];

    std::snprintf(caption, sizeof(caption), "Agent %d, generation %d, %d children, %d kills, %.0f s old, radius %.2f, %zu parameters, learning rate %.2g",
                  agent.id, agent.generation, agent.children, agent.kills, agent.age, agent.radius(), agent.brain->parameter_count(), agent.brain->learning_rate());

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
    return std::make_unique<continents::Continents>(ui, seed);
}
