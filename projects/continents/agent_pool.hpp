#ifndef CONTINENTS_AGENT_POOL_HPP
#define CONTINENTS_AGENT_POOL_HPP

#include "brain.hpp"
#include "grid.hpp"
#include "physics.hpp"

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <vector>

namespace continents {

// How an agent looks to others, as three numbers from 0 to 1, which is also the colour it gets
// drawn in. A child gets its parent's markings, nudged a little at random, so close family look
// alike, and lines that split a long time ago, like the ones on different continents, drift apart.
// An agent can't see anyone else's markings directly, only how much they look like its own.
constexpr int MARKINGS = 3;

using Markings = std::array<double, MARKINGS>;
using Calls = std::array<double, CHANNELS>;

// The smallest and biggest an agent can be, as mass, which is radius squared.
constexpr double MIN_MASS = .25;
constexpr double MAX_MASS = 9;
constexpr double BIRTH_MASS = .5;

constexpr double HEALTH_PER_MASS = 10;

// The energy that goes in to each unit of mass when growing. Shrinking only gets some of it back.
constexpr double BODY_ENERGY = 2;
constexpr double SHRINK_RETURN = .8;

// How far out the eyes see agents and food, and how far they see where the coast or a wall is.
constexpr double EYE_RANGE = 40;
constexpr double SHORE_RANGE = 20;

// Calls get quieter with the square of distance, past HEAR_SCALE, and can't be heard at all past HEAR_RANGE.
constexpr double HEAR_RANGE = 80;
constexpr double HEAR_SCALE = 10;

// How far apart two sets of markings are before they only look about 60% alike. With the default
// drift, siblings come out about 95% alike, and lines that split 25 generations ago about 25%.
constexpr double KIN_WIDTH = .15;

// How close family has to be for its eating to count towards the kin share of the score.
constexpr double KIN_SHARE_RANGE = 30;

// Biting and holding on reach this far past the front of an agent, times its radius, and only at
// whatever is within this many radians of straight ahead.
constexpr double BITE_REACH = .5;
constexpr double BITE_ANGLE = .8;
constexpr double HOLD_REACH = .5;
constexpr double HOLD_ANGLE = .8;

// How similar two sets of markings are, from 1 for the same down towards 0.
double kinship(const Markings& a, const Markings& b);

// One agent: a body, a brain, an energy bar, and a health bar.
struct Agent {
    int id = 0; // Goes up by one for every agent there has ever been.
    int body = 0; // Handle in the World.
    int slot = 0; // Where it is in the census for this step.
    int generation = 0;
    int family = 0; // Id of the founder it is descended from.

    std::unique_ptr<Brain> brain;

    Action action{}; // What it asked for on the last step.

    double left_thrust = 0; // What it got, after being slowed down by reproducing.
    double right_thrust = 0;

    double energy = 0;
    double progress = 0; // Towards having a child, from 0 to 1.
    double effort = 0; // How hard it is trying to reproduce, from -1 to 1.

    double mass = 1;
    double health = 0;

    Markings markings{};
    Calls calls{}; // How loud it is calling on each channel, from 0 to 1.

    double attack = 0; // How hard it is biting, from 0 to 1.
    bool hold = false; // Whether it wants to hold on to whatever is in front of it.

    double hurt = 0; // Damage taken on the last step.
    double dealt = 0; // Damage done on the last step.
    double fed = 0; // Energy it got out of whoever it bit on the last step.
    double family_hurt = 0; // Damage it did on the last step, times how related each one it bit was.
    double family_taken = 0; // Energy it took on the last step, the same way.
    bool could_bite = false; // Whether anything was in bite reach on the last step.
    int last_attacker = -1; // Id of whoever bit it last.

    double score = 0; // For the last step, which the brain gets on the next one.
    double bite_score = 0; // The part of the score that came from biting.
    bool wet = false; // Whether it was in the water at the end of the last step.
    double cold = 0;
    double spent = 0; // Energy spent on the last step, other than just being alive.

    double age = 0; // Seconds.
    int children = 0;
    int kills = 0;
    double eaten = 0; // Food over its life, with energy out of bites counted as the food it is worth.

    double radius() const;
    double capacity(double reproduction_cost) const; // Most energy it can hold.
    double max_health() const { return HEALTH_PER_MASS * mass; }
};

// The controls, as they were at the start of a step. Every pool works from the same copy, so that a
// slider moving part way through a step can't treat some agents differently from the rest.
struct Settings {
    double food_growth = 0;
    double max_food = 0;
    double meat_lifetime = 0;
    double food_energy = 0;
    double metabolism = 0;
    double thrust_energy = 0;
    double cold_energy = 0;
    double start_energy = 0;
    double reproduction_cost = 0;
    double reproduction_time = 0;
    double reproduction_decay = 0;
    double growth_rate = 0;
    double attack_damage = 0;
    double attack_energy = 0;
    double bite_drain = 0;
    double healing = 0;
    double hold_strength = 0;
    double hold_energy = 0;
    double call_energy = 0;
    double marking_drift = 0;
    double max_agents = 0;
    double mutation_rate = 0;

    double food_reward = 0;
    double closeness_reward = 0;
    double water_punish = 0;
    double water_friction = 0;
    double wall_punish = 0;
    double effort_punish = 0;
    double spin_punish = 0;
    double energy_punish = 0;
    double hurt_punish = 0;
    double attack_reward = 0;
    double kin_share = 0;
    double kin_harm = 0;
    double lambda = 1; // How much the rewards after an action count towards it, in learning. See Brain::learn().
    double bite_noise = 1; // The most the log of the spread of the bite output can be. See Brain::learn().
    double bite_focus = 0; // How much the bite is judged by just what it got straight away. See Brain::learn().
};

// What every agent looks like to the others, as of the start of a step. Agents can only see each
// other through this, which never changes while the pools are working, so it doesn't matter what
// order they run in.
struct Seen {
    Point pos;
    double rotation = 0;
    double radius = 1;

    Markings markings{};
    Calls calls{}; // From the step before, since everybody calls and listens at the same time.

    double eaten = 0; // Food eaten on this step, bites included, filled in once the physics is done.
};

struct Census {
    Census(double half_width, double half_height) : grid(half_width, half_height, EYE_RANGE / 2) {}

    std::vector<Seen> agents; // In the same order as the agents themselves, by id.
    Grid grid;
};

// Everything in a step that only has to do with one agent at a time: sensing, running and training
// its brain, and scoring what came of it, which together are nearly all of the work. These only
// ever read the world and the census, so that every thread can run them at once while the world
// sits still. The world only changes in between, on the main thread, which is also where anything
// that involves more than one agent happens, like eating, fighting, holding on, being born, and dying.

// Decide what to do, and leave it on the agent, for the main thread to use.
void think(Agent& agent, const World& world, const Census& census, const Settings& settings);

// Score the agent for the step that the world just took, and fill up or use up its energy, grow or
// shrink it, and heal it.
void score(Agent& agent, const World& world, const Census& census, const Settings& settings);

// The agents that one thread looks after, which stay in it for as long as they live. See
// projects/island_evolution/agent_pool.hpp for how the work gets shared out.
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

#endif // CONTINENTS_AGENT_POOL_HPP
