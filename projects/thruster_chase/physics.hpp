#ifndef THRUSTER_CHASE_PHYSICS_HPP
#define THRUSTER_CHASE_PHYSICS_HPP

#include <box2d/id.h>

namespace thruster_chase {

struct Point {
    double x = 0;
    double y = 0;
};

// Distance from the center of the object to whatever is in each direction, as the object sees it.
struct RaycastDistances {
    double front = 0;
    double back = 0;
    double left = 0;
    double right = 0;
};

// A circle in a walled box, as a Box2D world, with a thruster on each of its two legs. Forward is +y
// on the body, and it starts out pointing up. Not thread safe, the same as the World which owns it.
class PhysicsObject {
public:
    // right_leg is where the right thruster sits on the body. The left one is the mirror image of it.
    PhysicsObject(Point initial_pos, Point initial_vel, double box_size, double radius, Point right_leg);
    ~PhysicsObject();

    PhysicsObject(const PhysicsObject&) = delete;
    PhysicsObject& operator=(const PhysicsObject&) = delete;

    // Simulate the object. Positive thrust pushes forward, and negative pushes backward. Return true if it hit the box.
    bool sim(double left_thrust, double right_thrust, double dt);

    RaycastDistances raycast() const;

    // Turn a direction in the world in to the same direction as the object sees it.
    Point to_body(Point world_vector) const;

    Point pos;
    Point vel;

    double rotation = 0; // Radians, counter clockwise from pointing up.
    double spin = 0; // Radians per second, counter clockwise.

private:
    double raycast_distance(float x, float y) const;

    b2WorldId world;
    b2BodyId body;

    double box_size;
    Point right_leg;
};

}

#endif // THRUSTER_CHASE_PHYSICS_HPP
