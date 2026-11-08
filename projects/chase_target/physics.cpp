#include "physics.hpp"

#include <box2d/box2d.h>

#include <algorithm>
#include <cmath>

namespace chase_target {

// Box2D steps per call to sim(). It only looks for collisions once per step, and a bounce comes out
// about one step late, so this sets how accurate bounces are. Each one costs about 1us.
constexpr int PHYSICS_STEPS = 4;

// Solver passes inside each of those steps. These don't help the bounces at all, but they are what
// makes the walls hard. Box2D contacts are springs, which get stiffer with the square of the total
// passes per second (up to 240 of them, where it levels off). Each one costs about 0.13us.
constexpr int PHYSICS_SUBSTEPS = 4;

// The most that one thruster can do. A steady force gets the organism through a wall when it is
// enough to compress that spring by the organism's radius. With the numbers above that is about
// 3000, so anything that changes those needs to keep this comfortably under whatever that becomes.
constexpr double MAX_THRUST = 1000;

constexpr double DRAG = .01;

constexpr double WALL_BOUNCE = .5;

static void create_walls(b2WorldId world, float box_size)
{
    const b2BodyDef body_def = b2DefaultBodyDef();
    const b2BodyId walls = b2CreateBody(world, &body_def);

    b2ShapeDef shape_def = b2DefaultShapeDef();

    shape_def.material.friction = 0;
    shape_def.material.restitution = WALL_BOUNCE;

    // These are solid instead of just lines, so that if something ever does get shoved past the
    // surface, it gets pushed back out instead of being lost forever.
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

PhysicsObject::PhysicsObject(Point initial_pos, Point initial_vel, double box_size, double radius)
    : pos(initial_pos), vel(initial_vel), box_size(box_size)
{
    b2WorldDef world_def = b2DefaultWorldDef();

    world_def.gravity = {0, 0};
    world_def.restitutionThreshold = 0; // Bounce at any speed. By default slow hits just stop dead.
    world_def.hitEventThreshold = 0; // Every hit counts. Letting slow ones off left more organisms stuck on the walls.
    world_def.enableSleep = false;

    world = b2CreateWorld(&world_def);

    create_walls(world, static_cast<float>(box_size));

    b2BodyDef body_def = b2DefaultBodyDef();

    body_def.type = b2_dynamicBody;
    body_def.position = {static_cast<float>(pos.x), static_cast<float>(pos.y)};
    body_def.linearVelocity = {static_cast<float>(vel.x), static_cast<float>(vel.y)};

    body = b2CreateBody(world, &body_def);

    const float circle_radius = static_cast<float>(radius);
    const b2Circle circle = {{0, 0}, circle_radius};

    b2ShapeDef shape_def = b2DefaultShapeDef();

    shape_def.density = 1 / (B2_PI * circle_radius * circle_radius); // Mass = 1.
    shape_def.enableHitEvents = true;

    b2CreateCircleShape(body, &shape_def, &circle);
}

PhysicsObject::~PhysicsObject()
{
    b2DestroyWorld(world);
}

// Written like this, instead of with std::clamp, so that a NaN turns into no thrust.
static double clamp_thrust(double force)
{
    return std::min(MAX_THRUST, std::max(0.0, force));
}

bool PhysicsObject::sim(double up_force, double down_force, double left_force, double right_force, double dt)
{
    // Clamp forces
    const double thrust_x = clamp_thrust(right_force) - clamp_thrust(left_force);
    const double thrust_y = clamp_thrust(up_force) - clamp_thrust(down_force);

    const float step_time = static_cast<float>(dt / PHYSICS_STEPS);

    bool hit_wall = false;

    for (int i = 0; i < PHYSICS_STEPS; i++) {

        const b2Vec2 velocity = b2Body_GetLinearVelocity(body);

        // Drag always opposes velocity. This has to go in as a force, since the damping built in to
        // Box2D is linear, and this is not. Box2D clears forces after a step, so this needs doing each time.
        const double force_x = thrust_x - DRAG * velocity.x * std::abs(velocity.x);
        const double force_y = thrust_y - DRAG * velocity.y * std::abs(velocity.y);

        b2Body_ApplyForceToCenter(body, {static_cast<float>(force_x), static_cast<float>(force_y)}, true);

        b2World_Step(world, step_time, PHYSICS_SUBSTEPS);

        // Going by real impacts, instead of by touching, since Box2D counts things as touching from
        // 2cm away. That would punish the organism for just being near a wall.
        if (b2World_GetContactEvents(world).hitCount > 0) {
            hit_wall = true;
        }
    }

    const b2Pos position = b2Body_GetPosition(body);
    const b2Vec2 velocity = b2Body_GetLinearVelocity(body);

    pos = {position.x, position.y};
    vel = {velocity.x, velocity.y};

    return hit_wall;
}

double PhysicsObject::raycast_distance(float x, float y) const
{
    // Nothing inside the box can be further away than this.
    const double reach = 2 * box_size;

    const b2Pos origin = {static_cast<float>(pos.x), static_cast<float>(pos.y)};
    const b2Vec2 translation = {static_cast<float>(x * reach), static_cast<float>(y * reach)};

    const b2RayResult result = b2World_CastRayClosest(world, origin, translation, b2DefaultQueryFilter());

    return result.hit ? result.fraction * reach : reach;
}

RaycastDistances PhysicsObject::raycast() const
{
    return {raycast_distance(0, 1), raycast_distance(0, -1), raycast_distance(-1, 0), raycast_distance(1, 0)};
}

}
