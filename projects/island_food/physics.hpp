#ifndef ISLAND_FOOD_PHYSICS_HPP
#define ISLAND_FOOD_PHYSICS_HPP

// Utility wrapper around box2d physics.

#include <box2d/id.h>

#include <cstddef>
#include <cstdint>
#include <random>
#include <vector>

namespace island_food {

struct Point {
    double x = 0;
    double y = 0;
};

// An island. Everything inside the walls that isn't on one of these is water.
struct Rect {
    double left = 0;
    double right = 0;
    double bottom = 0;
    double top = 0;
};

// Distance from the center of the object to whatever is in each direction, as the object sees it.
struct RaycastDistances {
    double front = 0;
    double back = 0;
    double left = 0;
    double right = 0;
};

// What happened during a call to World::sim()
struct SimResult {
    bool hit_wall = false;
    int eaten = 0;
};

// A circle with a thruster on each of its two legs, in a walled box of water with islands in it, as a
// Box2D world. Forward is +y on the body, and it starts out pointing up. Only the walls and the
// organism are real to Box2D. The water just changes how much friction the organism gets, and food
// is eaten by getting close to it. Not thread safe, the same as the Project which owns it.
class World {
public:
    // right_leg is where the right thruster sits on the body. The left one is the mirror image of it.
    World(Point initial_pos, double box_size, double radius, double food_radius, Point right_leg, std::vector<Rect> islands, uint64_t seed);
    ~World();

    World(const World&) = delete;
    World& operator=(const World&) = delete;

    // Add or remove food until there is this much. It only ever goes on land.
    void set_food_count(std::size_t count);

    // Simulate the object. Positive thrust pushes forward, and negative pushes backward. Water friction
    // is how quickly the water slows it down, per second. Anything eaten comes back somewhere else.
    SimResult sim(double left_thrust, double right_thrust, double water_friction, double dt);

    RaycastDistances raycast() const;

    // Turn a direction in the world in to the same direction as the object sees it.
    Point to_body(Point world_vector) const;

    bool in_water() const;

    Point nearest_food() const;

    const std::vector<Rect>& island_list() const { return islands; }

    Point pos;
    Point vel;

    double rotation = 0; // Radians, counter clockwise from pointing up.
    double spin = 0; // Radians per second, counter clockwise.

    std::vector<Point> food;

private:
    void place_food(std::size_t index);

    double raycast_distance(float x, float y) const;

    b2WorldId world;
    b2BodyId body;

    std::mt19937_64 rng;

    double box_size;
    double radius;
    double food_radius;
    Point right_leg;

    std::vector<Rect> islands;
};

}

#endif // ISLAND_FOOD_PHYSICS_HPP
