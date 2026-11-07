#ifndef SIM_PHYSICS_HPP
#define SIM_PHYSICS_HPP

namespace sim {

struct Point {
    double x = 0;
    double y = 0;
};

class PhysicsObject {
public:
    PhysicsObject(Point initial_pos, Point initial_vel, double box_size);

    // Simulate the object. Return true if it hit the box.
    bool sim(double up_force, double down_force, double left_force, double right_force, double dt);

    Point pos;
    Point vel;

private:
    double box_size;
};

}

#endif // SIM_PHYSICS_HPP
