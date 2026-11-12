#ifndef ISLAND_EVOLUTION_AGENT_POOL_HPP
#define ISLAND_EVOLUTION_AGENT_POOL_HPP

#include "brain.hpp"
#include "physics.hpp"

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace island_evolution {

// One agent: a body, a brain, and an energy bar.
struct Agent {
    int id = 0; // Goes up by one for every agent there has ever been.
    int body = 0; // Handle in the World.
    int generation = 0;
    int family = 0; // Id of the founder it is descended from. Only used for the colour.

    std::unique_ptr<Brain> brain;

    Action action{}; // What it asked for on the last step.

    double left_thrust = 0; // What it got, after being slowed down by reproducing.
    double right_thrust = 0;

    double energy = 0;
    double progress = 0; // Towards having a child, from 0 to 1.
    double effort = 0; // How hard it is trying to reproduce, from -1 to 1.

    double score = 0; // For the last step, which the brain gets on the next one.
    bool wet = false; // Whether it was in the water at the end of the last step.

    double age = 0; // Seconds.
    int children = 0;
    uint64_t eaten = 0;
};

// The controls, as they were at the start of a step. Every pool works from the same copy, so that a
// slider moving part way through a step can't treat some agents differently from the rest.
struct Settings {
    double food_growth = 0;
    double max_food = 0;
    double food_energy = 0;
    double metabolism = 0;
    double thrust_energy = 0;
    double start_energy = 0;
    double reproduction_cost = 0;
    double reproduction_time = 0;
    double reproduction_decay = 0;
    double max_agents = 0;
    double mutation_rate = 0;

    double food_reward = 0;
    double birth_reward = 0;
    double closeness_reward = 0;
    double water_punish = 0;
    double water_friction = 0;
    double wall_punish = 0;
    double effort_punish = 0;
    double spin_punish = 0;
};

// Everything in a step that only has to do with one agent at a time: sensing, running and training
// its brain, and scoring what came of it, which together are nearly all of the work. These only
// ever read the world, so that every thread can run them at once while the world sits still. The
// world only changes in between, on the main thread, which is also where anything that involves
// more than one agent happens, like eating, being born, and dying.

// Decide what to do, and leave the thrust for it on the agent, for the physics to use.
void think(Agent& agent, const World& world, const Settings& settings);

// Score the agent for the step that the world just took, and fill up or use up its energy.
void score(Agent& agent, const World& world, const Settings& settings);

// The agents that one thread looks after, which stay in it for as long as they live.
//
// For each step, the pool hands its agents out one at a time, to its own thread first. A thread
// that runs out of agents of its own goes on to help with the other pools, by claiming whatever is
// left in them. Without that, every step waits on whichever pool has the most agents learning on
// that step, since learning costs many times what deciding does. Who runs an agent makes no
// difference to what happens to it.
//
// Lined up with the cache, so that the claims on two pools never share a line.
class alignas(64) AgentPool {
public:
    // Get ready to hand out every agent again. The slow ones go first, so that whatever is left over
    // at the end for other threads to help with is quick.
    void prepare(bool slowest_first);

    // The next agent that nobody has started on yet, or null once they have all been taken. Safe to
    // call from any thread, but only between prepare() and anything that adds or removes agents.
    Agent* claim();

    std::vector<Agent> agents;

private:
    std::vector<Agent*> queue;
    std::atomic<std::size_t> next{0};
};

}

#endif // ISLAND_EVOLUTION_AGENT_POOL_HPP
