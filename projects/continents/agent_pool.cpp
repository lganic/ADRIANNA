#include "agent_pool.hpp"

#include "continents.hpp"

#include <algorithm>
#include <cmath>

namespace continents {

constexpr double VEL_SCALE = 25;
constexpr double SPIN_SCALE = 5;
constexpr double FOOD_SCALE = 10; // Food further than this gets the whole of the closeness punishment.

// Pain reads 1 at a tenth of its health gone in one step.
constexpr double PAIN_SCALE = .1;

double kinship(const Markings& a, const Markings& b)
{
    double distance = 0;

    for (int i = 0; i < MARKINGS; i++) {
        distance += (a[i] - b[i]) * (a[i] - b[i]);
    }

    return std::exp(-distance / (2 * KIN_WIDTH * KIN_WIDTH));
}

double Agent::radius() const
{
    return std::sqrt(mass);
}

double Agent::capacity(double reproduction_cost) const
{
    // Room for two children's worth at the size of a founder, and more for bigger agents.
    return 2 * reproduction_cost * mass;
}

// Which eye sees something at this bearing, or -1 if none of them do. The eyes go from left to
// right, each one looking spread / 2 further round than the last, and each sees halfway to the next.
static int eye_for(double bearing, double spread)
{
    const double step = spread / 2;
    const double edge = spread + step / 2;

    if (std::abs(bearing) > edge || step <= 0) {
        return -1;
    }

    const int eye = static_cast<int>(std::lround((spread - bearing) / step));

    return std::min(EYES - 1, std::max(0, eye));
}

namespace {

// What one eye can see. Everything starts out as nothing at all.
struct Eye {
    double food = std::numeric_limits<double>::infinity();
    double agent = std::numeric_limits<double>::infinity();
    double kin = 0;
    double size = 0;
};

}

// How far away the closest food is that a body can see, or infinity if it can't see any.
static double nearest_visible_food(const World& world, const Body& body, double spread)
{
    double nearest = std::numeric_limits<double>::infinity();

    world.visit_food(body.pos, EYE_RANGE, [&](int index) {
        const Point offset = {world.food[index].pos.x - body.pos.x, world.food[index].pos.y - body.pos.y};
        const double distance = std::hypot(offset.x, offset.y);

        if (distance < nearest && eye_for(bearing(body.rotation, offset), spread) >= 0) {
            nearest = distance;
        }
    });

    return nearest;
}

static State sense(const Agent& agent, const World& world, const Census& census, const Settings& settings)
{
    // All of this is from the agent's point of view, since the thrusters only push along the way it
    // is facing.
    //
    // About itself:
    // * Velocity to the right, and ahead
    // * Spin
    // * Whether it is in the water
    // * How full its energy bar is, and its health bar
    // * How big it is
    // * How cold it is
    // * Whether it has enough energy to reproduce, and how far along it is with it
    // * Whether it is holding on to anything, and whether anything is holding on to it
    // * How much it got hurt on the last step
    // * A constant
    //
    // Then for each eye, from left to right:
    // * How close the nearest food is
    // * How close the nearest agent is, how much it looks like family, and how big it is next to this one
    // * How close the coast, or a wall, is, straight out along the middle of the eye
    //
    // Then for each channel:
    // * How loud everybody calling on it is, added up
    // * Which way the calls are coming from, on average, weighted by how loud each one is. The
    //   further this is from length 1, the more the calls come from all over.
    const Body& body = world.body(agent.body);
    const Seen& self = census.agents[agent.slot];

    const double spread = agent.brain->eye_spread();

    std::array<Eye, EYES> eyes;

    world.visit_food(body.pos, EYE_RANGE, [&](int index) {
        const Point offset = {world.food[index].pos.x - body.pos.x, world.food[index].pos.y - body.pos.y};
        const double distance = std::hypot(offset.x, offset.y);

        if (distance > EYE_RANGE) {
            return;
        }

        const int eye = eye_for(bearing(body.rotation, offset), spread);

        if (eye >= 0 && distance < eyes[eye].food) {
            eyes[eye].food = distance;
        }
    });

    std::array<double, CHANNELS> loudness{};
    std::array<Point, CHANNELS> direction{};

    // Far enough to hear, which is further than the eyes can see.
    census.grid.visit(body.pos, HEAR_RANGE, [&](int index) {
        if (index == agent.slot) {
            return;
        }

        const Seen& other = census.agents[index];

        const Point offset = {other.pos.x - body.pos.x, other.pos.y - body.pos.y};
        const double distance = std::hypot(offset.x, offset.y);

        if (distance > HEAR_RANGE) {
            return;
        }

        const double falloff = 1 / (1 + (distance / HEAR_SCALE) * (distance / HEAR_SCALE));
        const Point unit = distance > 1e-9 ? Point{offset.x / distance, offset.y / distance} : Point{0, 0};

        for (int channel = 0; channel < CHANNELS; channel++) {
            const double heard = other.calls[channel] * falloff;

            loudness[channel] += heard;
            direction[channel].x += heard * unit.x;
            direction[channel].y += heard * unit.y;
        }

        // Seen from the edge of the other body, rather than its middle.
        const double gap = std::max(0.0, distance - other.radius);

        if (gap > EYE_RANGE) {
            return;
        }

        const int eye = eye_for(bearing(body.rotation, offset), spread);

        if (eye >= 0 && gap < eyes[eye].agent) {
            eyes[eye].agent = gap;
            eyes[eye].kin = kinship(agent.markings, other.markings);
            eyes[eye].size = clamp(std::log2(other.radius / self.radius) / 2, -1, 1);
        }
    });

    State state;

    std::size_t at = 0;

    const auto put = [&](double value) { state[at++] = static_cast<float>(value); };

    const Point velocity = limit_length(to_body(body.rotation, {body.vel.x / VEL_SCALE, body.vel.y / VEL_SCALE}));
    const double cost = settings.reproduction_cost;

    put(velocity.x);
    put(velocity.y);
    put(clamp(body.spin / SPIN_SCALE, -1, 1));
    put(world.terrain.is_land(body.pos) ? 0 : 1);
    put(clamp(agent.energy / agent.capacity(cost), 0, 1));
    put(clamp(agent.health / agent.max_health(), 0, 1));
    put((agent.radius() - 1) / 2);
    put(world.terrain.cold(body.pos.y));
    put(agent.energy > cost ? 1 : 0);
    put(agent.progress);
    put(body.holding >= 0 ? 1 : 0);
    put(body.held_by > 0 ? 1 : 0);
    put(clamp(agent.hurt / (PAIN_SCALE * agent.max_health()), 0, 1));
    put(1);

    for (int eye = 0; eye < EYES; eye++) {
        const Eye& seen = eyes[eye];

        const double angle = body.rotation + spread * (1 - eye / 2.0);
        const double shore = world.terrain.boundary_distance(body.pos, heading(angle), SHORE_RANGE);

        put(std::isinf(seen.food) ? 0 : 1 - seen.food / EYE_RANGE);
        put(std::isinf(seen.agent) ? 0 : 1 - seen.agent / EYE_RANGE);
        put(seen.kin);
        put(seen.size);
        put(1 - shore / SHORE_RANGE);
    }

    for (int channel = 0; channel < CHANNELS; channel++) {
        const double total = loudness[channel];
        const Point average = total > 1e-9 ? limit_length(to_body(body.rotation, {direction[channel].x / total, direction[channel].y / total})) : Point{0, 0};

        put(1 - std::exp(-total));
        put(average.x);
        put(average.y);
    }

    return state;
}

void think(Agent& agent, const World& world, const Census& census, const Settings& settings)
{
    const double cost = settings.reproduction_cost;
    const double rate = 1 / settings.reproduction_time;

    Learning learning;

    learning.lambda = static_cast<float>(settings.lambda);
    learning.bite_noise = static_cast<float>(settings.bite_noise);
    learning.bite_focus = static_cast<float>(settings.bite_focus);

    Outcome outcome;

    outcome.score = static_cast<float>(agent.score);
    outcome.bite_score = static_cast<float>(agent.bite_score);
    outcome.could_bite = agent.could_bite;

    agent.action = agent.brain->run(outcome, sense(agent, world, census, settings), learning);

    // Effort is whatever the third output asks for, up to all or nothing in either direction.
    // Pushing on, with enough energy for a child, builds up progress at dt * effort. Anything
    // else loses it, and letting up altogether still loses it slowly.
    agent.effort = clamp(agent.action[ACT_REPRODUCE], -1, 1);

    if (agent.effort > 0) {
        if (agent.energy > cost) {
            agent.progress += DT * agent.effort * rate;
        }
    } else {
        agent.progress += DT * (agent.effort * rate - settings.reproduction_decay);
    }

    agent.progress = clamp(agent.progress, 0, 1);

    const double slowdown = 1 - REPRODUCTION_SLOWDOWN * std::max(0.0, agent.effort);

    agent.left_thrust = clamp(agent.action[ACT_LEFT], -1e6, 1e6) * slowdown;
    agent.right_thrust = clamp(agent.action[ACT_RIGHT], -1e6, 1e6) * slowdown;

    agent.attack = clamp(agent.action[ACT_ATTACK], 0, 1);
    agent.hold = agent.action[ACT_HOLD] > 0;

    for (int channel = 0; channel < CHANNELS; channel++) {
        agent.calls[channel] = clamp(agent.action[ACT_CALL + channel], 0, 1);
    }
}

void score(Agent& agent, const World& world, const Census& census, const Settings& settings)
{
    const double cost = settings.reproduction_cost;

    const Body& body = world.body(agent.body);

    const int eaten = body.eaten_plants + body.eaten_meat;
    const double radius = agent.radius();

    agent.wet = !world.terrain.is_land(body.pos);
    agent.cold = world.terrain.cold(body.pos.y);

    double calling = 0;

    for (double call : agent.calls) {
        calling += call;
    }

    const double thrust = std::abs(agent.left_thrust) + std::abs(agent.right_thrust);
    const bool holding = body.holding >= 0;

    // Everything it spends energy on other than just being alive. Thrust goes with radius, since
    // that is how its pushing scales. Cold goes with radius too, since heat is lost through the
    // skin, so big agents lose less of it for their mass.
    agent.spent = DT * (settings.thrust_energy * thrust * radius
                        + settings.cold_energy * agent.cold * radius
                        + settings.attack_energy * agent.attack * agent.mass
                        + (holding ? settings.hold_energy * agent.mass : 0)
                        + settings.call_energy * calling);

    agent.energy += eaten * settings.food_energy + agent.fed - agent.spent - DT * settings.metabolism * agent.mass;

    // Growing while over half full, and shrinking while under, faster the further from half it is.
    // Growing costs energy, and shrinking gets most of it back.
    const double fill = agent.energy / agent.capacity(cost);
    const double change = clamp(DT * settings.growth_rate * (2 * fill - 1), MIN_MASS - agent.mass, MAX_MASS - agent.mass);

    if (change > 0) {
        agent.energy -= BODY_ENERGY * change;
    } else {
        agent.energy -= BODY_ENERGY * SHRINK_RETURN * change;
    }

    agent.mass += change;
    agent.energy = std::min(agent.energy, agent.capacity(cost));

    agent.health = std::min(agent.max_health(), agent.health + DT * settings.healing * agent.max_health());

    // Family eating nearby, the closer the family the more it counts.
    double family_ate = 0;

    if (settings.kin_share > 0) {
        census.grid.visit(body.pos, KIN_SHARE_RANGE, [&](int index) {
            const Seen& other = census.agents[index];

            if (index == agent.slot || other.eaten == 0) {
                return;
            }

            if (std::hypot(other.pos.x - body.pos.x, other.pos.y - body.pos.y) <= KIN_SHARE_RANGE) {
                family_ate += kinship(agent.markings, other.markings) * other.eaten;
            }
        });
    }

    const double asked = std::abs(clamp(agent.action[ACT_LEFT], -1e6, 1e6)) + std::abs(clamp(agent.action[ACT_RIGHT], -1e6, 1e6));

    // Where it ended up after moving, so that this is about what it just did.
    const double food_distance = nearest_visible_food(world, body, agent.brain->eye_spread());

    // Energy out of a bite is worth the same as the same energy out of food.
    const double bitten_food = settings.food_energy > 0 ? agent.fed / settings.food_energy : 0;

    // What its bites cost family, as much as they were related: the damage, punished the same as
    // being hurt, and the energy taken, worth what it would have been as food. So biting a sibling
    // costs more than it feeds, and biting a stranger costs next to nothing.
    const double family_food = settings.food_energy > 0 ? agent.family_taken / settings.food_energy : 0;

    // What came of biting, on its own: what it took, what biting cost, and what it did to family.
    agent.bite_score = bitten_food * settings.food_reward
                       + settings.attack_reward * agent.dealt
                       - settings.energy_punish * DT * settings.attack_energy * agent.attack * agent.mass
                       - settings.kin_harm * (settings.hurt_punish * agent.family_hurt + settings.food_reward * family_food);

    agent.score = (eaten + bitten_food) * settings.food_reward
                  + settings.kin_share * settings.food_reward * family_ate
                  + settings.attack_reward * agent.dealt
                  - settings.closeness_reward * std::min(1.0, food_distance / FOOD_SCALE)
                  - settings.energy_punish * agent.spent
                  - settings.hurt_punish * agent.hurt
                  - settings.kin_harm * (settings.hurt_punish * agent.family_hurt + settings.food_reward * family_food)
                  - asked * settings.effort_punish // What it asked for, not what it got. See projects/island_evolution.
                  - std::abs(body.spin) * settings.spin_punish;

    if (agent.wet) {
        agent.score -= settings.water_punish;
    }

    if (body.hit_wall) {
        agent.score -= settings.wall_punish;
    }

    agent.age += DT;
    agent.eaten += eaten + bitten_food;
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
