#ifndef ISLAND_EVOLUTION_HPP
#define ISLAND_EVOLUTION_HPP

#include "agent_pool.hpp"
#include "brain.hpp"
#include "organism_graphic.hpp"
#include "physics.hpp"
#include "pool_threads.hpp"

#include <sim/project.hpp>

#include <codeepneat/rng.hpp>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace island_evolution {

constexpr double BOX_SIZE = 150;

// The map is the same on every run, whatever the seed, so that runs can be compared.
constexpr int ISLAND_COUNT = 40;
constexpr uint64_t ISLAND_SEED = 2026;
constexpr double ISLAND_GAP = 6; // The least water there is between two islands.

// The same body as projects/island_food, which explains why it is this big.
constexpr double ORGANISM_RADIUS = 1;
constexpr double FOOD_RADIUS = 1;

// Makes the drawing fit the collider exactly.
constexpr double ORGANISM_SIZE = ORGANISM_RADIUS / DRAWING_RADIUS;

constexpr double DT = 0.2;

constexpr double WALL_SENSE_RANGE = 15; // Islands are kept further than this from the walls, so no island can see one.
constexpr double FOOD_SCALE = 10;
constexpr double VEL_SCALE = 25;
constexpr double SPIN_SCALE = 5;

constexpr int FOUNDERS = 20; // How many agents there are to start with.

// Reproducing slows an agent down by this much at full effort.
constexpr double REPRODUCTION_SLOWDOWN = .95;

// How much of the energy that goes in to reproducing ends up in the child. The rest is lost.
constexpr double CHILD_SHARE = .8;

// The best agent so far, as it was when it died. Going by food eaten, and then by children for a tie.
struct Record {
    Heritage heritage;

    int generation = 0;
    int family = 0;
    int children = 0;
    uint64_t eaten = 0;
};

// projects/island_food, but with many more islands and a whole population of agents, which live off
// of an energy bar. Food is the only way to get energy, and once it runs out, an agent dies. With
// enough of it, an agent can put some of it in to having a child, which gets its parent's network
// with a few mutations, the way CoDeepNEAT does them, as well as what its parent had learned. Every
// agent keeps on learning for itself as it goes, with its own actor-critic.
//
// Reproducing takes commitment. An agent has a third output for how hard it is trying, which builds
// up progress towards a child, but slows it down by up to 95% while it does. Letting up loses the
// progress again, slowly. If everybody dies, the best agent so far gets a child of its own to start over.
//
// The agents are split up between a fixed number of pools, each with a thread of its own, and a new
// agent goes to whichever pool has the fewest. See AgentPool for what gets done where. Who is in
// which pool makes no difference to what happens, so a run comes out the same with any number of them.
class IslandEvolution : public sim::Project {
public:
    // How many pools the GUI gets: about one for each core.
    static int default_pool_count();

    IslandEvolution(sim::Ui& ui, uint64_t seed, int pool_count = default_pool_count());

    double time_step() const override;
    void setup_world(plotato::Graph& graph) override;

    void step(double t) override;
    void draw(plotato::Graph& graph) override;

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

    // Counts over the whole run.
    uint64_t total_births = 0;
    uint64_t total_deaths = 0;
    uint64_t total_eaten = 0;
    uint64_t extinctions = 0;
    int most_children = 0; // By any one agent.

private:
    Settings read_settings() const;

    void add_founder(const Settings& settings);

    // Hand an agent to the pool with the fewest in it.
    void place(Agent agent);

    // Has to happen after anything that adds or removes agents, since that moves them around.
    void update_order();

    // Do work(agent) for every agent, spread over all the threads. Each thread starts on its own
    // pool, and then helps with whatever is left in the others. See AgentPool.
    template <typename Work>
    void run_pools(bool slowest_first, Work work);

    // A brain for a child of this parent: the parent's genome put through mutation, and its weights.
    std::unique_ptr<Brain> breed(const Heritage& parent, const Settings& settings);

    // The living agent with the most children, and then the oldest, or null if there isn't one.
    const Agent* featured() const;

    void draw_network(plotato::Graph& graph, const Agent& agent);

    cdn::Rng rng;

    Genetics genetics;

    int next_id = 0;

    std::vector<AgentPool> pools;

    // Every agent in every pool, in order of id. Anything where the order makes a difference to what
    // happens goes by this, so that it doesn't depend on which pool an agent is in.
    std::vector<Agent*> order;

    PoolThreads threads; // After the pools, so that the threads stop before the pools go away.

    OrganismGraphic graphic;

    sim::Control food_growth;
    sim::Control max_food;
    sim::Control food_energy;
    sim::Control metabolism;
    sim::Control thrust_energy;
    sim::Control start_energy;
    sim::Control reproduction_cost;
    sim::Control reproduction_time;
    sim::Control reproduction_decay;
    sim::Control max_agents;
    sim::Control mutation_rate;

    sim::Control food_reward;
    sim::Control birth_reward;
    sim::Control closeness_reward;
    sim::Control water_punish;
    sim::Control water_friction;
    sim::Control wall_punish;
    sim::Control effort_punish;
    sim::Control spin_punish;

    sim::LiveGraph network_graph;

    sim::Field population_field;
    sim::Field births_field;
    sim::Field deaths_field;
    sim::Field extinction_field;
    sim::Field generation_field;
    sim::Field record_field;
    sim::Field species_field;
    sim::Field layers_field;

    sim::HistoryGraph population_graph;
    sim::HistoryGraph food_graph;
    sim::HistoryGraph birth_graph;
    sim::HistoryGraph energy_graph;
    sim::HistoryGraph reproducing_graph;
    sim::HistoryGraph water_graph;
    sim::HistoryGraph generation_graph;
    sim::HistoryGraph layers_graph;
    sim::HistoryGraph parameters_graph;
    sim::HistoryGraph learning_rate_graph;
    sim::HistoryGraph score_graph;
};

}

#endif // ISLAND_EVOLUTION_HPP
