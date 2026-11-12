#include "agent_pool.hpp"

#include "island_evolution.hpp"

#include <algorithm>
#include <cmath>

namespace island_evolution {

// Shrinks a vector down to length 1 if it is any longer, which keeps it pointing the same way.
static Point limit_length(Point vector)
{
    const double length = std::hypot(vector.x, vector.y);

    return length > 1 ? Point{vector.x / length, vector.y / length} : vector;
}

// See projects/island_food for why the walls are only seen from close up.
static double wall_closeness(double distance)
{
    return std::max(0.0, 1 - distance / WALL_SENSE_RANGE);
}

static double clamp(double value, double low, double high)
{
    return std::isnan(value) ? 0 : std::min(high, std::max(low, value));
}

static State sense(const World& world, const Agent& agent, double cost)
{
    // All of this is from the agent's point of view, since the thrusters only push along the way it
    // is facing. The first ten are the same as projects/island_food, which explains them.
    // * How close the wall is in front, behind, to the left, and to the right
    // * How far the food is to the right, and ahead
    // * Velocity to the right, and ahead
    // * Spin
    // * Whether it is in the water
    // * How full its energy bar is
    // * Whether it has enough energy to reproduce
    // * How far along it is with reproducing
    // * A constant
    const Body& body = world.body(agent.body);

    const RaycastDistances distances = world.raycast(agent.body);

    const Point target = world.nearest_food(body.pos);

    // With no food anywhere, there is nowhere in particular to go.
    const Point to_food = std::isinf(target.x) ? Point{0, 0} : world.to_body(agent.body, {target.x - body.pos.x, target.y - body.pos.y});
    const Point velocity = world.to_body(agent.body, body.vel);

    const Point food_input = limit_length({to_food.x / FOOD_SCALE, to_food.y / FOOD_SCALE});
    const Point velocity_input = limit_length({velocity.x / VEL_SCALE, velocity.y / VEL_SCALE});

    return {
        static_cast<float>(wall_closeness(distances.front)),
        static_cast<float>(wall_closeness(distances.back)),
        static_cast<float>(wall_closeness(distances.left)),
        static_cast<float>(wall_closeness(distances.right)),
        static_cast<float>(food_input.x),
        static_cast<float>(food_input.y),
        static_cast<float>(velocity_input.x),
        static_cast<float>(velocity_input.y),
        static_cast<float>(clamp(body.spin / SPIN_SCALE, -1, 1)),
        world.in_water(body.pos) ? 1.0f : 0.0f,
        static_cast<float>(clamp(agent.energy / (2 * cost), 0, 1)),
        agent.energy > cost ? 1.0f : 0.0f,
        static_cast<float>(agent.progress),
        1.0f,
    };
}

void think(Agent& agent, const World& world, const Settings& settings)
{
    const double cost = settings.reproduction_cost;
    const double rate = 1 / settings.reproduction_time;

    agent.action = agent.brain->run(static_cast<float>(agent.score), sense(world, agent, cost));

    // Effort is whatever the third output asks for, up to all or nothing in either direction.
    // Pushing on, with enough energy for a child, builds up progress at dt * effort. Anything
    // else loses it, and letting up altogether still loses it slowly.
    agent.effort = clamp(agent.action[2], -1, 1);

    if (agent.effort > 0) {
        if (agent.energy > cost) {
            agent.progress += DT * agent.effort * rate;
        }
    } else {
        agent.progress += DT * (agent.effort * rate - settings.reproduction_decay);
    }

    agent.progress = clamp(agent.progress, 0, 1);

    const double slowdown = 1 - REPRODUCTION_SLOWDOWN * std::max(0.0, agent.effort);

    agent.left_thrust = clamp(agent.action[0], -1e6, 1e6) * slowdown;
    agent.right_thrust = clamp(agent.action[1], -1e6, 1e6) * slowdown;
}

void score(Agent& agent, const World& world, const Settings& settings)
{
    const double max_energy = 2 * settings.reproduction_cost; // Room for two children's worth.

    const Body& body = world.body(agent.body);

    const Point target = world.nearest_food(body.pos);
    const double dist = std::hypot(target.x - body.pos.x, target.y - body.pos.y);

    const double thrust = std::abs(agent.left_thrust) + std::abs(agent.right_thrust);

    agent.wet = world.in_water(body.pos);

    agent.score = body.eaten * settings.food_reward - settings.closeness_reward * std::min(1.0, dist / FOOD_SCALE);

    if (agent.wet) {
        agent.score -= settings.water_punish;
    }

    if (body.hit_wall) {
        agent.score -= settings.wall_punish;
    }

    // Going by what it asked for, rather than what it got. Otherwise reproducing would be a way
    // to get out of paying for thrust, since it cuts the thrust down without changing the outputs.
    const double asked = std::abs(clamp(agent.action[0], -1e6, 1e6)) + std::abs(clamp(agent.action[1], -1e6, 1e6));

    agent.score -= asked * settings.effort_punish;
    agent.score -= std::abs(body.spin) * settings.spin_punish;

    agent.energy += body.eaten * settings.food_energy;
    agent.energy -= (settings.metabolism + thrust * settings.thrust_energy) * DT;
    agent.energy = std::min(agent.energy, max_energy);

    agent.age += DT;
    agent.eaten += body.eaten;
}

// Learning is by far the slowest thing an agent does, and then the bigger the network, the slower.
static bool slower(const Agent* a, const Agent* b)
{
    const bool a_learns = a->brain->learns_next();
    const bool b_learns = b->brain->learns_next();

    if (a_learns != b_learns) {
        return a_learns;
    }

    return a->brain->parameter_count() > b->brain->parameter_count();
}

void AgentPool::prepare(bool slowest_first)
{
    queue.clear();

    for (Agent& agent : agents) {
        queue.push_back(&agent);
    }

    if (slowest_first) {
        std::stable_sort(queue.begin(), queue.end(), slower);
    }

    next.store(0, std::memory_order_relaxed);
}

Agent* AgentPool::claim()
{
    // Looking first saves fighting over the counter of a pool that is already done.
    if (next.load(std::memory_order_relaxed) >= queue.size()) {
        return nullptr;
    }

    const std::size_t index = next.fetch_add(1, std::memory_order_relaxed);

    return index < queue.size() ? queue[index] : nullptr;
}

}
