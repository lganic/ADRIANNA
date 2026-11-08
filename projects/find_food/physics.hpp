#ifndef FIND_FOOD_PHYSICS_HPP
#define FIND_FOOD_PHYSICS_HPP

#include <box2d/id.h>

#include <cstddef>
#include <cstdint>
#include <random>
#include <vector>

namespace find_food {

struct Point {
    double x = 0;
    double y = 0;
};

// What happened during a call to Arena::sim()
struct SimResult {
    bool touching_wall = false; // Hit one, or was resting against one, at any point during the call.
    int eaten = 0;
};

// An agent in a walled box with food lying around, as a Box2D world.
//
// Each piece of food is a single sensor shape. The agent goes straight through it, Box2D reports the
// overlap (which is what eating is), and shape casts can still see it, which is what vision is.
class Arena {
public:
    Arena(double box_size, double agent_radius, double food_radius, uint64_t seed);
    ~Arena();

    Arena(const Arena&) = delete;
    Arena& operator=(const Arena&) = delete;

    // Add or remove food until there is this much.
    void set_food_count(std::size_t count);

    // Simulate the agent. Anything eaten comes back somewhere else.
    SimResult sim(double up_force, double down_force, double left_force, double right_force, double dt);

    // How far it is to the wall along a direction, which has to be a unit vector.
    double wall_distance(double x, double y) const;

    // Look for food along a direction. Returns 0 if there isn't any in range, rising to 1 right next to the agent.
    double look(double x, double y, double range) const;

    Point pos;
    Point vel;

    std::vector<Point> food;

private:
    void place_food(std::size_t index);

    b2WorldId world;
    b2BodyId agent;

    std::vector<b2BodyId> food_bodies;

    std::mt19937_64 rng;

    double box_size;
    double agent_radius;
    double food_radius;
};

}

#endif // FIND_FOOD_PHYSICS_HPP
