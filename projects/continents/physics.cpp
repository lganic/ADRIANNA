// Utility wrapper around box2d physics.

#include "physics.hpp"

#include <box2d/box2d.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <utility>

namespace continents {

// See the notes on these in projects/chase_target/physics.cpp. In short, steps set how accurate
// bounces are, substeps set how hard the walls are, and the thrust has to stay well under what
// the walls can take.
constexpr int PHYSICS_STEPS = 4;
constexpr int PHYSICS_SUBSTEPS = 4;

// The same body and thrust as projects/island_food, which explains these, at a radius of 1. Thrust,
// its limit, and drag all go up with the radius.
constexpr double MAX_THRUST = 500;
constexpr double THRUST_GAIN = 5;
constexpr double DRAG = .01;
constexpr float SPIN_DAMPING = 3;
constexpr double WALL_BOUNCE = .5;
constexpr double BODY_BOUNCE = .1; // Between bodies. Low, so that they can lean on each other.

// Mass is radius squared.
constexpr float DENSITY = static_cast<float>(1 / B2_PI);

constexpr double FOOD_CELL = 10; // Meters across each bucket of the food grid.

// The shortest that the bar between a body and whatever it has hold of can be.
constexpr double SHORTEST_HOLD = .05;

constexpr uint64_t WALL_CATEGORY = B2_DEFAULT_CATEGORY_BITS;
constexpr uint64_t BODY_CATEGORY = 2;

static void create_walls(b2WorldId world, float half_width, float half_height)
{
    const b2BodyDef body_def = b2DefaultBodyDef();
    const b2BodyId walls = b2CreateBody(world, &body_def);

    b2ShapeDef shape_def = b2DefaultShapeDef();

    shape_def.material.friction = 0;
    shape_def.material.restitution = WALL_BOUNCE;
    shape_def.filter.categoryBits = WALL_CATEGORY;

    // Solid instead of just lines, so that anything shoved past the surface gets pushed back out.
    const float thickness = 50;

    const b2Polygon sides[] = {
        b2MakeOffsetBox(half_width + 2 * thickness, thickness, {0, half_height + thickness}, b2Rot_identity),
        b2MakeOffsetBox(half_width + 2 * thickness, thickness, {0, -half_height - thickness}, b2Rot_identity),
        b2MakeOffsetBox(thickness, half_height + 2 * thickness, {-half_width - thickness, 0}, b2Rot_identity),
        b2MakeOffsetBox(thickness, half_height + 2 * thickness, {half_width + thickness, 0}, b2Rot_identity),
    };

    for (const b2Polygon& side : sides) {
        b2CreatePolygonShape(walls, &shape_def, &side);
    }
}

World::World(const Terrain& terrain, double food_radius, Point right_leg, uint64_t seed)
    : terrain(terrain), rng(seed), food_radius(food_radius), right_leg(right_leg), food_grid(terrain.half_width(), terrain.half_height(), FOOD_CELL)
{
    b2WorldDef world_def = b2DefaultWorldDef();

    world_def.gravity = {0, 0};
    world_def.restitutionThreshold = 0; // Bounce at any speed. By default slow hits just stop dead.
    world_def.hitEventThreshold = 0; // Every hit counts.
    world_def.enableSleep = false;

    world = b2CreateWorld(&world_def);

    create_walls(world, static_cast<float>(terrain.half_width()), static_cast<float>(terrain.half_height()));

    index_food();
}

World::~World()
{
    b2DestroyWorld(world);
}

int World::add_body(Point position, double rotation, double radius)
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
    body.radius = radius;

    const b2Circle circle = {{0, 0}, static_cast<float>(radius)};

    b2ShapeDef shape_def = b2DefaultShapeDef();

    shape_def.density = DENSITY;
    shape_def.material.friction = 0;
    shape_def.material.restitution = BODY_BOUNCE;
    shape_def.enableHitEvents = true;
    shape_def.filter.categoryBits = BODY_CATEGORY;
    shape_def.filter.maskBits = WALL_CATEGORY | BODY_CATEGORY;

    body.shape = b2CreateCircleShape(body.id, &shape_def, &circle);

    bodies.emplace(handle, body);

    return handle;
}

void World::remove_body(int handle)
{
    const auto found = bodies.find(handle);

    if (found == bodies.end()) {
        return;
    }

    release(handle);

    for (auto& entry : bodies) {
        if (entry.second.holding == handle) {
            release(entry.first);
        }
    }

    b2DestroyBody(found->second.id);

    bodies.erase(found);
}

void World::set_radius(int handle, double radius)
{
    Body& body = bodies.at(handle);

    const float r = static_cast<float>(radius);
    const b2Circle circle = {{0, 0}, r};

    b2Shape_SetCircle(body.shape, &circle);

    // Changing the shape doesn't change the mass by itself.
    const float mass = DENSITY * static_cast<float>(B2_PI) * r * r;

    b2Body_SetMassData(body.id, {mass, {0, 0}, .5f * mass * r * r});

    body.radius = radius;
}

void World::place_plant()
{
    food.push_back({terrain.random_land(rng), 0, false, false});
    plants++;
}

void World::grow_food(double per_second, std::size_t max_plants, double dt)
{
    // Newest first, which is the end of the list.
    for (std::size_t index = food.size(); index-- > 0 && plants > max_plants;) {
        if (!food[index].meat) {
            food.erase(food.begin() + static_cast<std::ptrdiff_t>(index));
            plants--;
        }
    }

    food_due += per_second * dt;

    while (food_due >= 1 && plants < max_plants) {
        place_plant();
        food_due -= 1;
    }

    // Growth doesn't save up while there is no room for it, or else eating a lot would bring a sudden flood of it.
    if (plants >= max_plants) {
        food_due = 0;
    }
}

void World::rot_meat(double lifetime, double dt)
{
    for (Food& piece : food) {
        if (piece.meat) {
            piece.age += dt;
        }
    }

    food.erase(std::remove_if(food.begin(), food.end(), [lifetime](const Food& piece) { return piece.meat && piece.age > lifetime; }), food.end());
}

void World::drop_meat(Point center, double spread, int pieces)
{
    std::uniform_real_distribution<double> unit(0, 1);

    const double edge_x = terrain.half_width() - food_radius;
    const double edge_y = terrain.half_height() - food_radius;

    for (int piece = 0; piece < pieces; piece++) {
        // The square root spreads them out evenly over the circle, instead of bunching them in the middle.
        const double angle = 2 * PI * unit(rng);
        const double distance = spread * std::sqrt(unit(rng));

        const Point spot = {
            clamp(center.x + distance * std::cos(angle), -edge_x, edge_x),
            clamp(center.y + distance * std::sin(angle), -edge_y, edge_y),
        };

        food.push_back({spot, 0, true, false});
    }
}

void World::index_food()
{
    food_grid.build(static_cast<int>(food.size()), [this](int index) { return food[index].pos; });
}

void World::hold(int holder, int target)
{
    Body& a = bodies.at(holder);
    const Body& b = bodies.at(target);

    if (a.holding >= 0 || holder == target) {
        return;
    }

    // From the front of the holder, to the spot on the target closest to it.
    const Point forward = heading(a.rotation);
    const Point mouth = {a.pos.x + forward.x * a.radius, a.pos.y + forward.y * a.radius};

    const double dx = mouth.x - b.pos.x;
    const double dy = mouth.y - b.pos.y;
    const double gap = std::hypot(dx, dy);

    const Point toward = gap > 1e-6 ? Point{dx / gap, dy / gap} : Point{-forward.x, -forward.y};
    const Point grip = to_body(b.rotation, {toward.x * b.radius, toward.y * b.radius});

    b2DistanceJointDef def = b2DefaultDistanceJointDef();

    def.base.bodyIdA = a.id;
    def.base.bodyIdB = b.id;
    def.base.localFrameA.p = {0, static_cast<float>(a.radius)};
    def.base.localFrameB.p = {static_cast<float>(grip.x), static_cast<float>(grip.y)};
    def.base.collideConnected = true;

    // Rigid, so that it can push as well as pull.
    def.length = static_cast<float>(std::max(SHORTEST_HOLD, gap - b.radius));
    def.enableSpring = false;

    a.hold = b2CreateDistanceJoint(world, &def);
    a.holding = target;

    bodies.at(target).held_by++;
}

void World::release(int holder)
{
    Body& a = bodies.at(holder);

    if (a.holding < 0) {
        return;
    }

    if (b2Joint_IsValid(a.hold)) {
        b2DestroyJoint(a.hold);
    }

    const auto target = bodies.find(a.holding);

    if (target != bodies.end()) {
        target->second.held_by = std::max(0, target->second.held_by - 1);
    }

    a.holding = -1;
}

void World::break_holds(double strength)
{
    for (auto& entry : bodies) {
        const Body& body = entry.second;

        if (body.holding < 0) {
            continue;
        }

        const b2Vec2 force = b2Joint_GetConstraintForce(body.hold);

        if (std::hypot(force.x, force.y) > strength * body.radius * body.radius) {
            release(entry.first);
        }
    }
}

// The check is needed since the thrust goes both ways, so std::max can't turn a NaN in to no thrust.
static double clamp_thrust(double force, double limit)
{
    if (std::isnan(force)) {
        return 0;
    }

    return std::min(limit, std::max(-limit, force));
}

void World::sim(double water_friction, double dt)
{
    const float step_time = static_cast<float>(dt / PHYSICS_STEPS);

    for (auto& entry : bodies) {
        entry.second.hit_wall = false;
        entry.second.eaten_plants = 0;
        entry.second.eaten_meat = 0;
    }

    for (int i = 0; i < PHYSICS_STEPS; i++) {

        // This all runs on the main thread, while the agent pools wait for it, so it is kept down to
        // as few calls in to Box2D as it can be. See projects/island_evolution/physics.cpp
        for (auto& entry : bodies) {
            Body& body = entry.second;

            const double size = body.radius;
            const double limit = MAX_THRUST * size;

            const float left = static_cast<float>(clamp_thrust(body.left_thrust * THRUST_GAIN * size, limit));
            const float right = static_cast<float>(clamp_thrust(body.right_thrust * THRUST_GAIN * size, limit));

            const float friction = terrain.is_land(body.pos) ? 0 : static_cast<float>(water_friction);

            if (friction != body.friction) {
                b2Body_SetLinearDamping(body.id, friction);
                b2Body_SetAngularDamping(body.id, SPIN_DAMPING + friction);

                body.friction = friction;
            }

            const b2Vec2 right_point = {static_cast<float>(right_leg.x * size), static_cast<float>(right_leg.y * size)};
            const b2Vec2 left_point = {-right_point.x, right_point.y};

            const b2WorldTransform transform = b2Body_GetTransform(body.id);
            const b2Vec2 forward = b2RotateVector(transform.q, {0, 1});

            b2Body_ApplyForce(body.id, b2MulSV(left, forward), b2TransformWorldPoint(transform, left_point), true);
            b2Body_ApplyForce(body.id, b2MulSV(right, forward), b2TransformWorldPoint(transform, right_point), true);

            const b2Vec2 velocity = b2Body_GetLinearVelocity(body.id);

            const double drag_x = -DRAG * size * velocity.x * std::abs(velocity.x);
            const double drag_y = -DRAG * size * velocity.y * std::abs(velocity.y);

            b2Body_ApplyForceToCenter(body.id, {static_cast<float>(drag_x), static_cast<float>(drag_y)}, true);
        }

        b2World_Step(world, step_time, PHYSICS_SUBSTEPS);

        // Real impacts, instead of touching. See projects/island_food/physics.cpp. Only the walls
        // count, since bumping in to each other is part of what the bodies are for here.
        const b2ContactEvents events = b2World_GetContactEvents(world);

        for (int hit = 0; hit < events.hitCount; hit++) {
            const intptr_t a = reinterpret_cast<intptr_t>(b2Body_GetUserData(b2Shape_GetBody(events.hitEvents[hit].shapeIdA)));
            const intptr_t b = reinterpret_cast<intptr_t>(b2Body_GetUserData(b2Shape_GetBody(events.hitEvents[hit].shapeIdB)));

            if ((a == 0) == (b == 0)) {
                continue;
            }

            const auto found = bodies.find(static_cast<int>((a == 0 ? b : a) - 1));

            if (found != bodies.end()) {
                found->second.hit_wall = true;
            }
        }

        for (auto& entry : bodies) {
            Body& body = entry.second;

            const b2Pos position = b2Body_GetPosition(body.id);

            body.pos = {position.x, position.y};

            const double eat_distance = body.radius + food_radius;

            food_grid.visit(body.pos, eat_distance, [&](int index) {
                Food& piece = food[index];

                if (piece.eaten) {
                    return;
                }

                const double dx = piece.pos.x - body.pos.x;
                const double dy = piece.pos.y - body.pos.y;

                if (dx * dx + dy * dy < eat_distance * eat_distance) {
                    piece.eaten = true;

                    if (piece.meat) {
                        body.eaten_meat++;
                    } else {
                        body.eaten_plants++;
                    }
                }
            });
        }
    }

    for (auto& entry : bodies) {
        Body& body = entry.second;

        const b2Vec2 velocity = b2Body_GetLinearVelocity(body.id);

        body.vel = {velocity.x, velocity.y};

        // Measured from pointing up, and +y is forward, so this is the same as the angle of the body.
        body.rotation = b2Rot_GetAngle(b2Body_GetRotation(body.id));
        body.spin = b2Body_GetAngularVelocity(body.id);

        plants -= static_cast<std::size_t>(body.eaten_plants);
    }

    food.erase(std::remove_if(food.begin(), food.end(), [](const Food& piece) { return piece.eaten; }), food.end());

    // The food moved around in the list, so the grid is out of date until index_food() runs again.
    index_food();
}

}
