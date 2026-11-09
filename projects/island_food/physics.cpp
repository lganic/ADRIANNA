
// Utility wrapper around box2d physics.

#include "physics.hpp"

#include <box2d/box2d.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <utility>

namespace island_food {

// See the notes on these in projects/chase_target/physics.cpp. In short, steps set how accurate
// bounces are, substeps set how hard the walls are, and the thrust has to stay well under what
// the walls can take.
constexpr int PHYSICS_STEPS = 4;
constexpr int PHYSICS_SUBSTEPS = 4;

// For one thruster. Both of them pushing the same way comes to the same 1000 as chase_target.
constexpr double MAX_THRUST = 500;

// Newtons per unit of what the organism asks for. Turning is thrust / radius, so with a body 5x the
// size of the one in thruster_chase, this keeps it turning just the same.
constexpr double THRUST_GAIN = 5;

constexpr double DRAG = .01;

// How quickly spin dies away on land, per second. See the note on this in projects/thruster_chase/physics.cpp
constexpr float SPIN_DAMPING = 3;

constexpr double WALL_BOUNCE = .5;

// New food won't turn up any closer to the organism than this, on top of their sizes.
constexpr double FOOD_SPAWN_GAP = .3;

static void create_walls(b2WorldId world, float box_size)
{
    const b2BodyDef body_def = b2DefaultBodyDef();
    const b2BodyId walls = b2CreateBody(world, &body_def);

    b2ShapeDef shape_def = b2DefaultShapeDef();

    shape_def.material.friction = 0;
    shape_def.material.restitution = WALL_BOUNCE;

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

World::World(Point initial_pos, double box_size, double radius, double food_radius, Point right_leg, std::vector<Rect> islands, uint64_t seed)
    : pos(initial_pos), rng(seed), box_size(box_size), radius(radius), food_radius(food_radius), right_leg(right_leg), islands(std::move(islands))
{
    b2WorldDef world_def = b2DefaultWorldDef();

    world_def.gravity = {0, 0};
    world_def.restitutionThreshold = 0; // Bounce at any speed. By default slow hits just stop dead.
    world_def.hitEventThreshold = 0; // Every hit counts.
    world_def.enableSleep = false;

    world = b2CreateWorld(&world_def);

    create_walls(world, static_cast<float>(box_size));

    b2BodyDef body_def = b2DefaultBodyDef();

    body_def.type = b2_dynamicBody;
    body_def.position = {static_cast<float>(pos.x), static_cast<float>(pos.y)};
    body_def.angularDamping = SPIN_DAMPING;

    body = b2CreateBody(world, &body_def);

    const float circle_radius = static_cast<float>(radius);
    const b2Circle circle = {{0, 0}, circle_radius};

    b2ShapeDef shape_def = b2DefaultShapeDef();

    shape_def.density = 1 / (B2_PI * circle_radius * circle_radius); // Mass = 1.
    shape_def.enableHitEvents = true;

    b2CreateCircleShape(body, &shape_def, &circle);
}

World::~World()
{
    b2DestroyWorld(world);
}

// Move a piece of food to a random spot on land, away from the organism so that it can't just get eaten again.
void World::place_food(std::size_t index)
{
    // Going by area, so that the food is just as dense on every island.
    std::vector<double> areas;

    for (const Rect& island : islands) {
        areas.push_back((island.right - island.left) * (island.top - island.bottom));
    }

    std::discrete_distribution<std::size_t> pick_island(areas.begin(), areas.end());

    const double closest = radius + food_radius + FOOD_SPAWN_GAP;

    Point position;

    do {
        const Rect& island = islands[pick_island(rng)];

        // Pulled in from the edges, so that all of the food is on land and not just the middle of it.
        std::uniform_real_distribution<double> spot_x(island.left + food_radius, island.right - food_radius);
        std::uniform_real_distribution<double> spot_y(island.bottom + food_radius, island.top - food_radius);

        position = {spot_x(rng), spot_y(rng)};

    } while (std::hypot(position.x - pos.x, position.y - pos.y) < closest);

    food[index] = position;
}

void World::set_food_count(std::size_t count)
{
    while (food.size() > count) {
        food.pop_back();
    }

    while (food.size() < count) {
        food.emplace_back();

        place_food(food.size() - 1);
    }
}

// The check is needed since the thrust goes both ways, so std::max can't turn a NaN in to no thrust.
static double clamp_thrust(double force)
{
    if (std::isnan(force)) {
        return 0;
    }

    return std::min(MAX_THRUST, std::max(-MAX_THRUST, force));
}

SimResult World::sim(double left_thrust, double right_thrust, double water_friction, double dt)
{
    const float left = static_cast<float>(clamp_thrust(left_thrust * THRUST_GAIN));
    const float right = static_cast<float>(clamp_thrust(right_thrust * THRUST_GAIN));

    const b2Vec2 right_point = {static_cast<float>(right_leg.x), static_cast<float>(right_leg.y)};
    const b2Vec2 left_point = {-right_point.x, right_point.y};

    const float step_time = static_cast<float>(dt / PHYSICS_STEPS);

    const double eat_distance = radius + food_radius;

    SimResult result;

    for (int i = 0; i < PHYSICS_STEPS; i++) {

        // The damping built in to Box2D is what this should be, since it can't go unstable however
        // high the slider gets. A force strong enough to matter could reverse the organism in a single step.
        const float friction = in_water() ? static_cast<float>(water_friction) : 0;

        b2Body_SetLinearDamping(body, friction);
        b2Body_SetAngularDamping(body, SPIN_DAMPING + friction);

        // Pushing from the legs is what turns it, when one of them is pushing harder than the
        // other. Box2D clears forces after a step, and the body has turned, so this needs doing each time.
        const b2Vec2 forward = b2Body_GetWorldVector(body, {0, 1});

        b2Body_ApplyForce(body, b2MulSV(left, forward), b2Body_GetWorldPoint(body, left_point), true);
        b2Body_ApplyForce(body, b2MulSV(right, forward), b2Body_GetWorldPoint(body, right_point), true);

        const b2Vec2 velocity = b2Body_GetLinearVelocity(body);

        // Drag always opposes velocity. This has to go in as a force, since the damping built in to
        // Box2D is linear, and this is not.
        const double drag_x = -DRAG * velocity.x * std::abs(velocity.x);
        const double drag_y = -DRAG * velocity.y * std::abs(velocity.y);

        b2Body_ApplyForceToCenter(body, {static_cast<float>(drag_x), static_cast<float>(drag_y)}, true);

        b2World_Step(world, step_time, PHYSICS_SUBSTEPS);

        // Going by real impacts, instead of by touching, since Box2D counts things as touching from
        // 2cm away. That would punish the organism for just being near a wall.
        if (b2World_GetContactEvents(world).hitCount > 0) {
            result.hit_wall = true;
        }

        const b2Pos position = b2Body_GetPosition(body);

        pos = {position.x, position.y};

        for (std::size_t index = 0; index < food.size(); index++) {
            if (std::hypot(food[index].x - pos.x, food[index].y - pos.y) < eat_distance) {
                place_food(index);
                result.eaten++;
            }
        }
    }

    const b2Vec2 velocity = b2Body_GetLinearVelocity(body);

    vel = {velocity.x, velocity.y};

    // Measured from pointing up, and +y is forward, so this is the same as the angle of the body.
    rotation = b2Rot_GetAngle(b2Body_GetRotation(body));
    spin = b2Body_GetAngularVelocity(body);

    return result;
}

Point World::to_body(Point world_vector) const
{
    const b2Vec2 local = b2Body_GetLocalVector(body, {static_cast<float>(world_vector.x), static_cast<float>(world_vector.y)});

    return {local.x, local.y};
}

double World::raycast_distance(float x, float y) const
{
    // The rays turn with the body, so the longest they can be is corner to corner across the box.
    const float reach = static_cast<float>(2 * std::sqrt(2.0) * box_size);

    const b2Pos origin = {static_cast<float>(pos.x), static_cast<float>(pos.y)};
    const b2Vec2 translation = b2MulSV(reach, b2Body_GetWorldVector(body, {x, y}));

    const b2RayResult result = b2World_CastRayClosest(world, origin, translation, b2DefaultQueryFilter());

    return result.hit ? result.fraction * reach : reach;
}

RaycastDistances World::raycast() const
{
    return {raycast_distance(0, 1), raycast_distance(0, -1), raycast_distance(-1, 0), raycast_distance(1, 0)};
}

bool World::in_water() const
{
    for (const Rect& island : islands) {
        if (pos.x >= island.left && pos.x <= island.right && pos.y >= island.bottom && pos.y <= island.top) {
            return false;
        }
    }

    return true;
}

Point World::nearest_food() const
{
    Point closest;
    double closest_distance = std::numeric_limits<double>::infinity();

    for (const Point& piece : food) {

        const double distance = std::hypot(piece.x - pos.x, piece.y - pos.y);

        if (distance < closest_distance) {
            closest = piece;
            closest_distance = distance;
        }
    }

    return closest;
}

}
