#ifndef CONTINENTS_CONTINENTS_HPP
#define CONTINENTS_CONTINENTS_HPP

#include "agent_pool.hpp"
#include "brain.hpp"
#include "grid.hpp"
#include "organism_graphic.hpp"
#include "physics.hpp"
#include "pool_threads.hpp"
#include "terrain.hpp"

#include <sim/project.hpp>

#include <codeepneat/rng.hpp>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <random>
#include <vector>

namespace continents {

constexpr double HALF_WIDTH = 500;
constexpr double HALF_HEIGHT = 350;

// The map is the same on every run, whatever the seed, so that runs can be compared.
constexpr uint64_t MAP_SEED = 2026;

constexpr double FOOD_RADIUS = 1;

// Makes the drawing fit a collider of radius 1 exactly. Bigger agents get drawn bigger to match.
constexpr double ORGANISM_SIZE = 1 / DRAWING_RADIUS;

constexpr double DT = 0.2;

constexpr int FOUNDERS = 60; // How many agents there are to start with, spread over all the land.
constexpr int RESTARTERS = 10; // How many children the best agent so far gets, if everybody dies.

// The sea starts out calm, so that the founders can spread out to every landmass, and gets rough
// once each one has at least this many agents on it at the same time, not counting founders, which
// mostly cuts them off from each other, so that the population on each one can go its own way. It
// goes calm again if everybody dies.
constexpr int SETTLERS = 5;

// Reproducing slows an agent down by this much at full effort.
constexpr double REPRODUCTION_SLOWDOWN = .95;

// How much of the energy that goes in to reproducing ends up in the child, its body included. The rest is lost.
constexpr double CHILD_SHARE = .8;

// How much of what an agent had in it, energy and body, is left as meat when it dies. Under 1, so
// that nobody can make energy out of nothing by raising children and eating them.
constexpr double CORPSE_SHARE = .7;
constexpr int MOST_MEAT = 200; // Pieces from one body, however big it was.

// Seconds old an agent has to be for killing it to count as more than killing a newborn.
constexpr double GROWN_AGE = 60;

// The best agent so far, as it was when it died. Going by food eaten, and then by children for a tie.
struct Record {
    Heritage heritage;
    Markings markings{};

    int generation = 0;
    int family = 0;
    int children = 0;
    double eaten = 0;
};

// projects/island_evolution, grown in to a whole world. Six continents, kept apart by the sea, and a
// strip of land at each pole, where it is cold enough that just staying alive costs extra energy.
// The sea is calm until every landmass has been settled, and rough after (see SETTLERS).
//
// On top of eating, moving, and reproducing, every agent can:
// * Bite whatever is right in front of it, harder the bigger it is, and feed on the energy it
//   takes. An agent that runs out of health dies the same as one that runs out of energy, and
//   either way, what was left of its energy and its body gets scattered around as meat, which
//   anybody can eat.
// * Take hold of another agent in front of it, which joins them up so that it can drag it along
//   or push it around, until it lets go, or gets pulled on harder than it is strong enough for.
// * Call out on any of five channels, which everybody nearby hears, added up, along with which way
//   it is coming from.
//
// Agents grow while their energy bar is over half full, and shrink while it is under. Bigger agents
// bite harder, take more to kill, and hold more energy, but need more food to keep going.
//
// Each one can see in five directions, out to EYE_RANGE, and for any other agent it sees, it can
// tell how big it is next to itself, and how much it looks like family. Everybody has markings,
// which get passed down and drift a little with each child, so looking alike means being related.
// That is the only way an agent has of telling friend from foe. What to do about it, it has to
// learn, or evolve.
//
// The agents are split up between a fixed number of pools, each with a thread of its own, the same
// as projects/island_evolution. Who is in which pool makes no difference to what happens, so a run
// comes out the same with any number of them.
class Continents : public sim::Project {
public:
    // How many pools the GUI gets: about one for each core.
    static int default_pool_count();

    Continents(sim::Ui& ui, uint64_t seed, int pool_count = default_pool_count());

    double time_step() const override;
    void setup_world(plotato::Graph& graph) override;

    void step(double t) override;
    void draw(plotato::Graph& graph) override;

    Terrain terrain;
    World world;

    std::size_t population() const { return order.size(); }

    // Every agent, in order of id, which is also the order they were born in.
    template <typename Visit>
    void for_each_agent(Visit visit) const
    {
        for (const Agent* agent : order) {
            visit(*agent);
        }
    }

    bool has_record = false;
    Record record;

    bool settled = false; // Whether every landmass has been settled, which makes the sea rough.

    // Counts over the whole run.
    uint64_t total_births = 0;
    uint64_t total_kills = 0;
    uint64_t total_starved = 0;
    uint64_t total_plants = 0;
    uint64_t total_meat = 0;
    double total_bitten = 0; // Energy taken by biting.
    double total_damage = 0; // Done by biting.
    double total_family_damage = 0; // The same, times how related biter and bitten were.
    uint64_t total_grown_kills = 0; // Of agents at least GROWN_AGE old, so not just newborns.
    double total_kill_kinship = 0; // Between each agent killed and its killer, added up.
    uint64_t extinctions = 0;
    int most_children = 0; // By any one agent.

private:
    Settings read_settings() const;

    void add_founder(const Settings& settings);

    // Hand an agent to the pool with the fewest in it.
    void place(Agent agent);

    // Has to happen after anything that adds or removes agents, since that moves them around.
    void update_order();

    // Do work(agent) for every agent, spread over all the threads. See projects/island_evolution.
    template <typename Work>
    void run_pools(bool slowest_first, Work work);

    // What everybody looks like right now, for the agents to see each other by.
    void take_census();

    // Take hold of, or let go of, whatever each agent wants to.
    void update_holds();

    // Everybody who is biting bites whatever is in front of it.
    void fight(const Settings& settings);

    // The nearest other agent in front of this one, close enough to reach, or null.
    Agent* in_reach(const Agent& agent, const Grid& grid, double reach, double angle);

    // A brain for a child of this parent: the parent's genome put through mutation, and its weights.
    std::unique_ptr<Brain> breed(const Heritage& parent, const Settings& settings);

    Markings drift(const Markings& parent, double amount);

    // The living agent with the most children, and then the oldest, or null if there isn't one.
    const Agent* featured() const;

    void draw_network(plotato::Graph& graph, const Agent& agent);

    cdn::Rng rng;
    std::mt19937_64 spots; // For picking spots on the map.

    Genetics genetics;

    int next_id = 0;

    std::vector<AgentPool> pools;

    // Every agent in every pool, in order of id. Anything where the order makes a difference to what
    // happens goes by this, so that it doesn't depend on which pool an agent is in.
    std::vector<Agent*> order;

    Census census;

    Grid contacts; // Where everybody is right after the physics, for fighting.

    // Taken from fight() for the graphs: the damage done on this step, and how much it was done to
    // family, weighted by kinship.
    double step_damage = 0;
    double step_family_damage = 0;

    PoolThreads threads; // After the pools, so that the threads stop before the pools go away.

    OrganismGraphic graphic;

    sim::Control food_growth;
    sim::Control max_food;
    sim::Control meat_lifetime;
    sim::Control food_energy;
    sim::Control metabolism;
    sim::Control thrust_energy;
    sim::Control cold_energy;
    sim::Control start_energy;
    sim::Control reproduction_cost;
    sim::Control reproduction_time;
    sim::Control reproduction_decay;
    sim::Control growth_rate;
    sim::Control attack_damage;
    sim::Control attack_energy;
    sim::Control bite_drain;
    sim::Control healing;
    sim::Control hold_strength;
    sim::Control hold_energy;
    sim::Control call_energy;
    sim::Control marking_drift;
    sim::Control max_agents;
    sim::Control mutation_rate;

    sim::Control food_reward;
    sim::Control closeness_reward;
    sim::Control water_punish;
    sim::Control water_friction;
    sim::Control settled_friction;
    sim::Control wall_punish;
    sim::Control effort_punish;
    sim::Control spin_punish;
    sim::Control energy_punish;
    sim::Control hurt_punish;
    sim::Control attack_reward;
    sim::Control kin_share;
    sim::Control kin_harm;
    sim::Control lambda;
    sim::Control bite_noise;
    sim::Control bite_focus;

    sim::LiveGraph network_graph;

    sim::Field population_field;
    sim::Field births_field;
    sim::Field kills_field;
    sim::Field starved_field;
    sim::Field extinction_field;
    sim::Field generation_field;
    sim::Field record_field;
    sim::Field landmass_field;
    sim::Field friction_field;
    sim::Field biggest_field;
    sim::Field species_field;
    sim::Field layers_field;

    sim::HistoryGraph population_graph;
    sim::HistoryGraph plant_graph;
    sim::HistoryGraph meat_graph;
    sim::HistoryGraph hunting_graph;
    sim::HistoryGraph birth_graph;
    sim::HistoryGraph kill_graph;
    sim::HistoryGraph starve_graph;
    sim::HistoryGraph size_graph;
    sim::HistoryGraph energy_graph;
    sim::HistoryGraph attacking_graph;
    sim::HistoryGraph holding_graph;
    sim::HistoryGraph calling_graph;
    sim::HistoryGraph bitten_kin_graph;
    sim::HistoryGraph neighbour_kin_graph;
    sim::HistoryGraph water_graph;
    sim::HistoryGraph cold_graph;
    sim::HistoryGraph reproducing_graph;
    sim::HistoryGraph generation_graph;
    sim::HistoryGraph layers_graph;
    sim::HistoryGraph memory_graph;
    sim::HistoryGraph parameters_graph;
    sim::HistoryGraph learning_rate_graph;
    sim::HistoryGraph eye_graph;
    sim::HistoryGraph score_graph;
};

}

#endif // CONTINENTS_CONTINENTS_HPP
