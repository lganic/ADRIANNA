#include <sim/physics.hpp>

#include <algorithm>
#include <cmath>

namespace sim {

PhysicsObject::PhysicsObject(Point initial_pos, Point initial_vel, double box_size)
    : pos(initial_pos), vel(initial_vel), box_size(box_size) {}

// Fold one axis back into the box. Return true if it had to bounce.
static bool apply_clamping(double& position, double& velocity, double box_size)
{
    bool hit = false;

    while (std::abs(position) > box_size) {

        if (position > box_size) {
            position = 2 * box_size - position;
        }
        else {
            position = -2 * box_size - position;
        }

        // Invert with half dampening
        velocity = -.5 * velocity;

        hit = true;
    }

    return hit;
}

bool PhysicsObject::sim(double up_force, double down_force, double left_force, double right_force, double dt)
{
    // Clamp forces
    up_force = std::max(0.0, up_force);
    down_force = std::max(0.0, down_force);
    left_force = std::max(0.0, left_force);
    right_force = std::max(0.0, right_force);

    // Euler integration with mass = 1, because I am lazy.
    double force_x = right_force - left_force;
    double force_y = up_force - down_force;

    // drag (always opposes velocity)
    force_x += -.01 * vel.x * std::abs(vel.x);
    force_y += -.01 * vel.y * std::abs(vel.y);

    vel.x += dt * force_x;
    vel.y += dt * force_y;

    pos.x += dt * vel.x;
    pos.y += dt * vel.y;

    // Both axes always need clamping, so no short circuiting these.
    const bool hit_x = apply_clamping(pos.x, vel.x, box_size);
    const bool hit_y = apply_clamping(pos.y, vel.y, box_size);

    return hit_x || hit_y;
}

}
