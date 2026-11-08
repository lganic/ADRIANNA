#ifndef CHASE_TARGET_PHYSICS_HPP
#define CHASE_TARGET_PHYSICS_HPP

#include <box2d/id.h>

namespace chase_target {

struct Point {
    double x = 0;
    double y = 0;
};

// Distance from the center of the object to whatever is in each direction.
struct RaycastDistances {
    double top = 0;
    double bottom = 0;
    double left = 0;
    double right = 0;
};

// A circle in a walled box, as a Box2D world. Not thread safe, the same as the World which owns it.
class PhysicsObject {
public:
    PhysicsObject(Point initial_pos, Point initial_vel, double box_size, double radius);
    ~PhysicsObject();

    PhysicsObject(const PhysicsObject&) = delete;
    PhysicsObject& operator=(const PhysicsObject&) = delete;

    // Simulate the object. Return true if it hit the box.
    bool sim(double up_force, double down_force, double left_force, double right_force, double dt);

    RaycastDistances raycast() const;

    Point pos;
    Point vel;

private:
    double raycast_distance(float x, float y) const;

    b2WorldId world;
    b2BodyId body;

    double box_size;
};

}

#endif // CHASE_TARGET_PHYSICS_HPP
