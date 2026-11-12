#ifndef ISLAND_EVOLUTION_PHYSICS_HPP
#define ISLAND_EVOLUTION_PHYSICS_HPP

// Utility wrapper around box2d physics.

#include <box2d/id.h>

#include <cstddef>
#include <cstdint>
#include <map>
#include <random>
#include <vector>

namespace island_evolution {

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

// Distance from the center of a body to the wall in each direction, as the body sees it.
struct RaycastDistances {
    double front = 0;
    double back = 0;
    double left = 0;
    double right = 0;
};

// One organism, as far as the physics is concerned. Everything in here is from the end of the last step.
struct Body {
    Point pos;
    Point vel;

    double rotation = 0; // Radians, counter clockwise from pointing up.
    double spin = 0; // Radians per second, counter clockwise.

    double left_thrust = 0; // What gets pushed with on the next step. Positive is forward.
    double right_thrust = 0;

    bool hit_wall = false; // During the last step.
    int eaten = 0; // During the last step.

    b2BodyId id;

    float friction = 0; // The water friction that Box2D has for it right now.
};

// Scatter islands around the box, keeping at least island_gap of water between any two of them, and
// wall_gap between them and the walls. The same seed always gives the same map. Gives up on adding
// more once there is no room left, so it can come back with fewer than asked for.
std::vector<Rect> generate_islands(double box_size, int count, double wall_gap, double island_gap, uint64_t seed);

// A walled box of water with islands in it, as a Box2D world, with any number of thruster bodies in it.
// Each body is a circle with a thruster on each of its two legs. Forward is +y on the body. The bodies
// only bump in to the walls, and go straight through each other. The water just changes how much
// friction a body gets, and food is eaten by getting close to it.
//
// The const functions only read, Box2D's ray casts included, so any number of threads can use them
// at once, as long as nothing changes the world in the meantime. Anything else is one thread at a time.
class World {
public:
    // right_leg is where the right thruster sits on a body. The left one is the mirror image of it.
    World(double box_size, double radius, double food_radius, Point right_leg, std::vector<Rect> islands, uint64_t seed);
    ~World();

    World(const World&) = delete;
    World& operator=(const World&) = delete;

    // Returns the handle for the new body, which stays the same for as long as it is around.
    int add_body(Point position, double rotation);
    void remove_body(int handle);

    Body& body(int handle) { return bodies.at(handle); }
    const Body& body(int handle) const { return bodies.at(handle); }

    // Grow food at this many pieces per second, until there is max_food of it lying around. Anything
    // over the max goes away straight off, so that turning the slider down takes effect.
    void grow_food(double per_second, std::size_t max_food, double dt);

    // Simulate every body, using the thrust set on each one. Water friction is how quickly the water
    // slows them down, per second.
    void sim(double water_friction, double dt);

    RaycastDistances raycast(int handle) const;

    // Turn a direction in the world in to the same direction as the body sees it.
    Point to_body(int handle, Point world_vector) const;

    bool in_water(Point position) const;

    // A spot on land somewhere, going by area, so that every island is just as likely per square meter.
    Point random_land();

    // Where the closest food to this spot is. Infinitely far away if there isn't any.
    Point nearest_food(Point position) const;

    const std::vector<Rect>& island_list() const { return islands; }

    std::vector<Point> food;

private:
    void place_food();

    double raycast_distance(const Body& body, float x, float y) const;

    b2WorldId world;

    std::map<int, Body> bodies; // Ordered, so that who gets to the food first doesn't depend on memory layout.
    int next_handle = 0;

    std::mt19937_64 rng;

    double box_size;
    double radius;
    double food_radius;
    Point right_leg;

    double food_due = 0; // Fractions of a piece of food, waiting to grow.

    std::vector<Rect> islands;
    std::discrete_distribution<std::size_t> pick_island;
};

}

#endif // ISLAND_EVOLUTION_PHYSICS_HPP
