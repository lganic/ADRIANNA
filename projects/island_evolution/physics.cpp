
// Utility wrapper around box2d physics.

#include "physics.hpp"

#include <box2d/box2d.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <utility>

namespace island_evolution {

// See the notes on these in projects/chase_target/physics.cpp. In short, steps set how accurate
// bounces are, substeps set how hard the walls are, and the thrust has to stay well under what
// the walls can take.
constexpr int PHYSICS_STEPS = 4;
constexpr int PHYSICS_SUBSTEPS = 4;

// The same body and thrust as projects/island_food, which explains these.
constexpr double MAX_THRUST = 500;
constexpr double THRUST_GAIN = 5;
constexpr double DRAG = .01;
constexpr float SPIN_DAMPING = 3;
constexpr double WALL_BOUNCE = .5;

// New food won't turn up any closer to a body than this, on top of their sizes. It gives up on that
// after this many tries, since with enough bodies around there might not be anywhere left.
constexpr double FOOD_SPAWN_GAP = .3;
constexpr int FOOD_SPAWN_TRIES = 20;

constexpr double MIN_ISLAND_SIZE = 10;
constexpr double MAX_ISLAND_SIZE = 32;
constexpr int ISLAND_TRIES = 200; // Per island.

// The bodies are in their own category, which only collides with the walls, so they pass through each other.
constexpr uint64_t WALL_CATEGORY = B2_DEFAULT_CATEGORY_BITS;
constexpr uint64_t BODY_CATEGORY = 2;

static bool overlaps(const Rect& a, const Rect& b, double gap)
{
    return a.left < b.right + gap && b.left < a.right + gap && a.bottom < b.top + gap && b.bottom < a.top + gap;
}

std::vector<Rect> generate_islands(double box_size, int count, double wall_gap, double island_gap, uint64_t seed)
{
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<double> size(MIN_ISLAND_SIZE, MAX_ISLAND_SIZE);

    std::vector<Rect> islands;

    for (int island = 0; island < count; island++) {
        for (int attempt = 0; attempt < ISLAND_TRIES; attempt++) {
            const double width = size(rng);
            const double height = size(rng);

            const double reach = box_size - wall_gap;

            std::uniform_real_distribution<double> left(-reach, reach - width);
            std::uniform_real_distribution<double> bottom(-reach, reach - height);

            Rect candidate;
            candidate.left = left(rng);
            candidate.right = candidate.left + width;
            candidate.bottom = bottom(rng);
            candidate.top = candidate.bottom + height;

            const bool clear = std::none_of(islands.begin(), islands.end(), [&](const Rect& other) { return overlaps(candidate, other, island_gap); });

            if (clear) {
                islands.push_back(candidate);
                break;
            }
        }
    }

    return islands;
}

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

static std::discrete_distribution<std::size_t> area_distribution(const std::vector<Rect>& islands)
{
    std::vector<double> areas;

    for (const Rect& island : islands) {
        areas.push_back((island.right - island.left) * (island.top - island.bottom));
    }

    return std::discrete_distribution<std::size_t>(areas.begin(), areas.end());
}

World::World(double box_size, double radius, double food_radius, Point right_leg, std::vector<Rect> islands, uint64_t seed)
    : rng(seed), box_size(box_size), radius(radius), food_radius(food_radius), right_leg(right_leg), islands(std::move(islands)),
      pick_island(area_distribution(this->islands))
{
    b2WorldDef world_def = b2DefaultWorldDef();

    world_def.gravity = {0, 0};
    world_def.restitutionThreshold = 0; // Bounce at any speed. By default slow hits just stop dead.
    world_def.hitEventThreshold = 0; // Every hit counts.
    world_def.enableSleep = false;

    world = b2CreateWorld(&world_def);

    create_walls(world, static_cast<float>(box_size));
}

World::~World()
{
    b2DestroyWorld(world);
}

int World::add_body(Point position, double rotation)
{
    const int handle = next_handle++;

    b2BodyDef body_def = b2DefaultBodyDef();

    body_def.type = b2_dynamicBody;
    body_def.position = {static_cast<float>(position.x), static_cast<float>(position.y)};
    body_def.rotation = b2MakeRot(static_cast<float>(rotation));
    body_def.angularDamping = SPIN_DAMPING;

    // Off by one, so that the walls, which have none, can't be mistaken for the first body.
    body_def.userData = reinterpret_cast<void*>(static_cast<intptr_t>(handle) + 1);

    Body body;

    body.id = b2CreateBody(world, &body_def);
    body.pos = position;
    body.rotation = rotation;

    const float circle_radius = static_cast<float>(radius);
    const b2Circle circle = {{0, 0}, circle_radius};

    b2ShapeDef shape_def = b2DefaultShapeDef();

    shape_def.density = 1 / (B2_PI * circle_radius * circle_radius); // Mass = 1.
    shape_def.enableHitEvents = true;
    shape_def.filter.categoryBits = BODY_CATEGORY;
    shape_def.filter.maskBits = WALL_CATEGORY;

    b2CreateCircleShape(body.id, &shape_def, &circle);

    bodies.emplace(handle, body);

    return handle;
}

void World::remove_body(int handle)
{
    const auto found = bodies.find(handle);

    if (found == bodies.end()) {
        return;
    }

    b2DestroyBody(found->second.id);

    bodies.erase(found);
}

Point World::random_land()
{
    const Rect& island = islands[pick_island(rng)];

    // Pulled in from the edges, so that all of the food is on land and not just the middle of it.
    std::uniform_real_distribution<double> spot_x(island.left + food_radius, island.right - food_radius);
    std::uniform_real_distribution<double> spot_y(island.bottom + food_radius, island.top - food_radius);

    return {spot_x(rng), spot_y(rng)};
}

// Put a new piece of food on land, away from the bodies so that it can't just get eaten straight off.
void World::place_food()
{
    const double closest = radius + food_radius + FOOD_SPAWN_GAP;

    Point position;

    for (int attempt = 0; attempt < FOOD_SPAWN_TRIES; attempt++) {
        position = random_land();

        const bool clear = std::none_of(bodies.begin(), bodies.end(), [&](const std::pair<const int, Body>& entry) {
            return std::hypot(position.x - entry.second.pos.x, position.y - entry.second.pos.y) < closest;
        });

        if (clear) {
            break;
        }
    }

    food.push_back(position);
}

void World::grow_food(double per_second, std::size_t max_food, double dt)
{
    while (food.size() > max_food) {
        food.pop_back();
    }

    food_due += per_second * dt;

    while (food_due >= 1 && food.size() < max_food) {
        place_food();
        food_due -= 1;
    }

    // Growth doesn't save up while there is no room for it, or else eating a lot would bring a sudden flood of it.
    if (food.size() >= max_food) {
        food_due = 0;
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

void World::sim(double water_friction, double dt)
{
    const b2Vec2 right_point = {static_cast<float>(right_leg.x), static_cast<float>(right_leg.y)};
    const b2Vec2 left_point = {-right_point.x, right_point.y};

    const float step_time = static_cast<float>(dt / PHYSICS_STEPS);

    const double eat_distance = radius + food_radius;

    for (auto& entry : bodies) {
        entry.second.hit_wall = false;
        entry.second.eaten = 0;
    }

    for (int i = 0; i < PHYSICS_STEPS; i++) {

        // This all runs on the main thread, while the agent pools wait for it, so it is kept down to
        // as few calls in to Box2D as it can be. Each of those checks the handle it is given, which
        // adds up over every body on every physics step.
        for (auto& entry : bodies) {
            Body& body = entry.second;

            const float left = static_cast<float>(clamp_thrust(body.left_thrust * THRUST_GAIN));
            const float right = static_cast<float>(clamp_thrust(body.right_thrust * THRUST_GAIN));

            // See projects/island_food/physics.cpp for why this is damping, and why it can change at
            // any time. Box2D keeps it, so it only has to be told when it changes.
            const float friction = in_water(body.pos) ? static_cast<float>(water_friction) : 0;

            if (friction != body.friction) {
                b2Body_SetLinearDamping(body.id, friction);
                b2Body_SetAngularDamping(body.id, SPIN_DAMPING + friction);

                body.friction = friction;
            }

            // The same as b2Body_GetWorldVector and b2Body_GetWorldPoint, from one call instead of three.
            const b2WorldTransform transform = b2Body_GetTransform(body.id);
            const b2Vec2 forward = b2RotateVector(transform.q, {0, 1});

            b2Body_ApplyForce(body.id, b2MulSV(left, forward), b2TransformWorldPoint(transform, left_point), true);
            b2Body_ApplyForce(body.id, b2MulSV(right, forward), b2TransformWorldPoint(transform, right_point), true);

            const b2Vec2 velocity = b2Body_GetLinearVelocity(body.id);

            const double drag_x = -DRAG * velocity.x * std::abs(velocity.x);
            const double drag_y = -DRAG * velocity.y * std::abs(velocity.y);

            b2Body_ApplyForceToCenter(body.id, {static_cast<float>(drag_x), static_cast<float>(drag_y)}, true);
        }

        b2World_Step(world, step_time, PHYSICS_SUBSTEPS);

        // Real impacts, instead of touching. See projects/island_food/physics.cpp
        const b2ContactEvents events = b2World_GetContactEvents(world);

        for (int hit = 0; hit < events.hitCount; hit++) {
            for (const b2ShapeId shape : {events.hitEvents[hit].shapeIdA, events.hitEvents[hit].shapeIdB}) {

                const intptr_t tag = reinterpret_cast<intptr_t>(b2Body_GetUserData(b2Shape_GetBody(shape)));

                if (tag != 0) {
                    const auto found = bodies.find(static_cast<int>(tag - 1));

                    if (found != bodies.end()) {
                        found->second.hit_wall = true;
                    }
                }
            }
        }

        for (auto& entry : bodies) {
            Body& body = entry.second;

            const b2Pos position = b2Body_GetPosition(body.id);

            body.pos = {position.x, position.y};

            // Going backwards, since eaten food gets swapped out for the last piece.
            for (std::size_t index = food.size(); index-- > 0;) {
                const double dx = food[index].x - body.pos.x;
                const double dy = food[index].y - body.pos.y;

                // Nearly all of it is too far away in one direction or the other, which is quicker
                // to check than the distance. The distance can't be any shorter, so this never changes the answer.
                if (std::abs(dx) >= eat_distance || std::abs(dy) >= eat_distance) {
                    continue;
                }

                if (std::hypot(dx, dy) < eat_distance) {
                    food[index] = food.back();
                    food.pop_back();

                    body.eaten++;
                }
            }
        }
    }

    for (auto& entry : bodies) {
        Body& body = entry.second;

        const b2Vec2 velocity = b2Body_GetLinearVelocity(body.id);

        body.vel = {velocity.x, velocity.y};

        // Measured from pointing up, and +y is forward, so this is the same as the angle of the body.
        body.rotation = b2Rot_GetAngle(b2Body_GetRotation(body.id));
        body.spin = b2Body_GetAngularVelocity(body.id);
    }
}

Point World::to_body(int handle, Point world_vector) const
{
    const b2Vec2 local = b2Body_GetLocalVector(body(handle).id, {static_cast<float>(world_vector.x), static_cast<float>(world_vector.y)});

    return {local.x, local.y};
}

double World::raycast_distance(const Body& body, float x, float y) const
{
    // The rays turn with the body, so the longest they can be is corner to corner across the box.
    const float reach = static_cast<float>(2 * std::sqrt(2.0) * box_size);

    const b2Pos origin = {static_cast<float>(body.pos.x), static_cast<float>(body.pos.y)};
    const b2Vec2 translation = b2MulSV(reach, b2Body_GetWorldVector(body.id, {x, y}));

    // Only the walls, and not the other bodies.
    b2QueryFilter filter = b2DefaultQueryFilter();
    filter.categoryBits = BODY_CATEGORY;
    filter.maskBits = WALL_CATEGORY;

    const b2RayResult result = b2World_CastRayClosest(world, origin, translation, filter);

    return result.hit ? result.fraction * reach : reach;
}

RaycastDistances World::raycast(int handle) const
{
    const Body& target = body(handle);

    return {raycast_distance(target, 0, 1), raycast_distance(target, 0, -1), raycast_distance(target, -1, 0), raycast_distance(target, 1, 0)};
}

bool World::in_water(Point position) const
{
    for (const Rect& island : islands) {
        if (position.x >= island.left && position.x <= island.right && position.y >= island.bottom && position.y <= island.top) {
            return false;
        }
    }

    return true;
}

Point World::nearest_food(Point position) const
{
    const double far = std::numeric_limits<double>::infinity();

    Point closest = {far, far};
    double closest_distance = far;

    for (const Point& piece : food) {

        const double distance = std::hypot(piece.x - position.x, piece.y - position.y);

        if (distance < closest_distance) {
            closest = piece;
            closest_distance = distance;
        }
    }

    return closest;
}

}
