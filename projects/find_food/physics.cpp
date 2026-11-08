#include "physics.hpp"

#include <box2d/box2d.h>

#include <algorithm>
#include <cmath>

namespace find_food {

// See the notes on these in projects/chase_target/physics.cpp. In short, steps set how accurate
// bounces are, substeps set how hard the walls are, and the thrust has to stay well under what
// the walls can take.
constexpr int PHYSICS_STEPS = 4;
constexpr int PHYSICS_SUBSTEPS = 4;
constexpr double MAX_THRUST = 1000;

constexpr double DRAG = .01;

constexpr double WALL_BOUNCE = .5;

// Vision is a fat ray, so that it can't slip between food. This is how wide it is, as a radius.
constexpr float VISION_WIDTH = .3f;

// New food won't turn up any closer to the agent than this, on top of their sizes.
constexpr double FOOD_SPAWN_GAP = .3;

constexpr uint64_t WALL_CATEGORY = 1;
constexpr uint64_t FOOD_CATEGORY = 2;
constexpr uint64_t AGENT_CATEGORY = 4;

static void create_walls(b2WorldId world, float box_size)
{
    const b2BodyDef body_def = b2DefaultBodyDef();
    const b2BodyId walls = b2CreateBody(world, &body_def);

    b2ShapeDef shape_def = b2DefaultShapeDef();

    shape_def.material.friction = 0;
    shape_def.material.restitution = WALL_BOUNCE;
    shape_def.filter.categoryBits = WALL_CATEGORY;

    // Solid instead of just lines, so that anything shoved past the surface gets pushed back out.
    const float half_thickness = box_size;
    const float center = box_size + half_thickness;
    const float half_length = box_size + 2 * half_thickness; // Long enough to fill in the corners.

    const b2Polygon sides[] = {
        b2MakeOffsetBox(half_length, half_thickness, {0, center}, b2Rot_identity),
        b2MakeOffsetBox(half_length, half_thickness, {0, -center}, b2Rot_identity),
        b2MakeOffsetBox(half_thickness, half_length, {-center, 0}, b2Rot_identity),
        b2MakeOffsetBox(half_thickness, half_length, {center, 0}, b2Rot_identity),
    };

    for (const b2Polygon& side : sides) {
        b2CreatePolygonShape(walls, &shape_def, &side);
    }
}

Arena::Arena(double box_size, double agent_radius, double food_radius, uint64_t seed)
    : rng(seed), box_size(box_size), agent_radius(agent_radius), food_radius(food_radius)
{
    b2WorldDef world_def = b2DefaultWorldDef();

    world_def.gravity = {0, 0};
    world_def.restitutionThreshold = 0; // Bounce at any speed. By default slow hits just stop dead.
    world_def.hitEventThreshold = 0;
    world_def.enableSleep = false;

    world = b2CreateWorld(&world_def);

    create_walls(world, static_cast<float>(box_size));

    b2BodyDef body_def = b2DefaultBodyDef();

    body_def.type = b2_dynamicBody;

    agent = b2CreateBody(world, &body_def);

    const float circle_radius = static_cast<float>(agent_radius);
    const b2Circle circle = {{0, 0}, circle_radius};

    b2ShapeDef shape_def = b2DefaultShapeDef();

    shape_def.density = 1 / (B2_PI * circle_radius * circle_radius); // Mass = 1.
    shape_def.enableHitEvents = true;
    shape_def.enableSensorEvents = true;
    shape_def.filter.categoryBits = AGENT_CATEGORY;

    b2CreateCircleShape(agent, &shape_def, &circle);
}

Arena::~Arena()
{
    b2DestroyWorld(world);
}

// Move a piece of food to a random spot, away from the agent so that it can't just get eaten again.
void Arena::place_food(std::size_t index)
{
    const double limit = box_size - food_radius;
    const double closest = agent_radius + food_radius + FOOD_SPAWN_GAP;

    std::uniform_real_distribution<double> spot(-limit, limit);

    Point position;

    do {
        position = {spot(rng), spot(rng)};
    } while (std::hypot(position.x - pos.x, position.y - pos.y) < closest);

    food[index] = position;

    b2Body_SetTransform(food_bodies[index], {static_cast<float>(position.x), static_cast<float>(position.y)}, b2Rot_identity);
}

void Arena::set_food_count(std::size_t count)
{
    while (food_bodies.size() > count) {
        b2DestroyBody(food_bodies.back());

        food_bodies.pop_back();
        food.pop_back();
    }

    while (food_bodies.size() < count) {

        const b2BodyDef body_def = b2DefaultBodyDef();
        const b2BodyId body = b2CreateBody(world, &body_def);

        const b2Circle circle = {{0, 0}, static_cast<float>(food_radius)};

        b2ShapeDef shape_def = b2DefaultShapeDef();

        shape_def.isSensor = true;
        shape_def.enableSensorEvents = true;
        shape_def.filter.categoryBits = FOOD_CATEGORY;
        shape_def.filter.maskBits = AGENT_CATEGORY; // Sensors see other sensors too, otherwise food eats food.

        b2CreateCircleShape(body, &shape_def, &circle);

        food_bodies.push_back(body);
        food.emplace_back();

        place_food(food.size() - 1);
    }
}

// Written like this, instead of with std::clamp, so that a NaN turns into no thrust.
static double clamp_thrust(double force)
{
    return std::min(MAX_THRUST, std::max(0.0, force));
}

SimResult Arena::sim(double up_force, double down_force, double left_force, double right_force, double dt)
{
    const double thrust_x = clamp_thrust(right_force) - clamp_thrust(left_force);
    const double thrust_y = clamp_thrust(up_force) - clamp_thrust(down_force);

    const float step_time = static_cast<float>(dt / PHYSICS_STEPS);

    SimResult result;

    for (int i = 0; i < PHYSICS_STEPS; i++) {

        const b2Vec2 velocity = b2Body_GetLinearVelocity(agent);

        // Drag always opposes velocity. Box2D clears forces after a step, so this needs doing each time.
        const double force_x = thrust_x - DRAG * velocity.x * std::abs(velocity.x);
        const double force_y = thrust_y - DRAG * velocity.y * std::abs(velocity.y);

        b2Body_ApplyForceToCenter(agent, {static_cast<float>(force_x), static_cast<float>(force_y)}, true);

        b2World_Step(world, step_time, PHYSICS_SUBSTEPS);

        if (b2World_GetContactEvents(world).hitCount > 0) {
            result.hit_wall = true;
        }

        const b2Pos position = b2Body_GetPosition(agent);

        pos = {position.x, position.y};

        const b2SensorEvents sensors = b2World_GetSensorEvents(world);

        for (int event = 0; event < sensors.beginCount; event++) {

            const b2BodyId eaten = b2Shape_GetBody(sensors.beginEvents[event].sensorShapeId);

            for (std::size_t index = 0; index < food_bodies.size(); index++) {
                if (B2_ID_EQUALS(food_bodies[index], eaten)) {
                    place_food(index);
                    result.eaten++;
                }
            }
        }
    }

    const b2Vec2 velocity = b2Body_GetLinearVelocity(agent);

    vel = {velocity.x, velocity.y};

    return result;
}

double Arena::wall_distance(double x, double y) const
{
    // Nothing inside the box can be further away than this.
    const double reach = 2 * box_size;

    const b2Pos origin = {static_cast<float>(pos.x), static_cast<float>(pos.y)};
    const b2Vec2 translation = {static_cast<float>(x * reach), static_cast<float>(y * reach)};

    const b2QueryFilter filter = {AGENT_CATEGORY, WALL_CATEGORY};

    const b2RayResult result = b2World_CastRayClosest(world, origin, translation, filter);

    return result.hit ? result.fraction * reach : reach;
}

// Keeps the closest hit, by clipping the cast to each one that it finds.
static float keep_closest(b2ShapeId, b2Pos, b2Vec2, float fraction, void* context)
{
    *static_cast<float*>(context) = fraction;

    return fraction;
}

double Arena::look(double x, double y, double range) const
{
    const b2Vec2 center = {0, 0};
    const b2ShapeProxy proxy = b2MakeProxy(&center, 1, VISION_WIDTH);

    // Starting out in front of the agent, otherwise anything close would overlap the cast in every
    // direction, which says that there is food nearby, but not where.
    const b2Pos origin = {static_cast<float>(pos.x + x * VISION_WIDTH), static_cast<float>(pos.y + y * VISION_WIDTH)};

    const double reach = range - VISION_WIDTH;
    const b2Vec2 translation = {static_cast<float>(x * reach), static_cast<float>(y * reach)};

    const b2QueryFilter filter = {AGENT_CATEGORY, FOOD_CATEGORY};

    float closest = 1; // Which stays the end of the cast if it doesn't find anything.

    b2World_CastShape(world, origin, &proxy, translation, filter, keep_closest, &closest);

    return 1 - closest;
}

}
