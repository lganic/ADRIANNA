
// Utility wrapper around box2d physics.

#include "physics.hpp"

#include <box2d/box2d.h>

#include <algorithm>
#include <cmath>

namespace thruster_chase {

// See the notes on these in projects/chase_target/physics.cpp. In short, steps set how accurate
// bounces are, substeps set how hard the walls are, and the thrust has to stay well under what
// the walls can take.
constexpr int PHYSICS_STEPS = 4;
constexpr int PHYSICS_SUBSTEPS = 4;

// For one thruster. Both of them pushing the same way comes to the same 1000 as chase_target.
constexpr double MAX_THRUST = 500;

constexpr double DRAG = .01;

// How quickly spin dies away, per second. This one can use the damping built in to Box2D, since the
// body is so small that any drag strong enough to matter would be unstable if it went in as a torque.
constexpr float SPIN_DAMPING = 3;

constexpr double WALL_BOUNCE = .5;

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

PhysicsObject::PhysicsObject(Point initial_pos, Point initial_vel, double box_size, double radius, Point right_leg)
    : pos(initial_pos), vel(initial_vel), box_size(box_size), right_leg(right_leg)
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
    body_def.linearVelocity = {static_cast<float>(vel.x), static_cast<float>(vel.y)};
    body_def.angularDamping = SPIN_DAMPING;

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

// The check is needed since the thrust goes both ways now, so std::max can't turn a NaN in to no thrust.
static double clamp_thrust(double force)
{
    if (std::isnan(force)) {
        return 0;
    }

    return std::min(MAX_THRUST, std::max(-MAX_THRUST, force));
}

bool PhysicsObject::sim(double left_thrust, double right_thrust, double dt)
{
    const float left = static_cast<float>(clamp_thrust(left_thrust));
    const float right = static_cast<float>(clamp_thrust(right_thrust));

    const b2Vec2 right_point = {static_cast<float>(right_leg.x), static_cast<float>(right_leg.y)};
    const b2Vec2 left_point = {-right_point.x, right_point.y};

    const float step_time = static_cast<float>(dt / PHYSICS_STEPS);

    bool hit_wall = false;

    for (int i = 0; i < PHYSICS_STEPS; i++) {

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
            hit_wall = true;
        }
    }

    const b2Pos position = b2Body_GetPosition(body);
    const b2Vec2 velocity = b2Body_GetLinearVelocity(body);

    pos = {position.x, position.y};
    vel = {velocity.x, velocity.y};

    // Measured from pointing up, and +y is forward, so this is the same as the angle of the body.
    rotation = b2Rot_GetAngle(b2Body_GetRotation(body));
    spin = b2Body_GetAngularVelocity(body);

    return hit_wall;
}

Point PhysicsObject::to_body(Point world_vector) const
{
    const b2Vec2 local = b2Body_GetLocalVector(body, {static_cast<float>(world_vector.x), static_cast<float>(world_vector.y)});

    return {local.x, local.y};
}

double PhysicsObject::raycast_distance(float x, float y) const
{
    // The rays turn with the body, so the longest they can be is corner to corner across the box.
    const float reach = static_cast<float>(2 * std::sqrt(2.0) * box_size);

    const b2Pos origin = {static_cast<float>(pos.x), static_cast<float>(pos.y)};
    const b2Vec2 translation = b2MulSV(reach, b2Body_GetWorldVector(body, {x, y}));

    const b2RayResult result = b2World_CastRayClosest(world, origin, translation, b2DefaultQueryFilter());

    return result.hit ? result.fraction * reach : reach;
}

RaycastDistances PhysicsObject::raycast() const
{
    return {raycast_distance(0, 1), raycast_distance(0, -1), raycast_distance(-1, 0), raycast_distance(1, 0)};
}

}
