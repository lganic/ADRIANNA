#ifndef CONTINENTS_PHYSICS_HPP
#define CONTINENTS_PHYSICS_HPP

// Utility wrapper around box2d physics.

#include "geometry.hpp"
#include "grid.hpp"
#include "terrain.hpp"

#include <box2d/id.h>

#include <cstddef>
#include <cstdint>
#include <map>
#include <random>
#include <vector>

namespace continents {

// One piece of food. Plants grow on their own, and meat is what is left when an agent dies, which
// rots away after a while.
struct Food {
    Point pos;
    double age = 0; // Seconds. Only meat goes off.
    bool meat = false;
    bool eaten = false; // Only ever true part way through World::sim().
};

// One organism, as far as the physics is concerned. Everything in here is from the end of the last step.
struct Body {
    Point pos;
    Point vel;

    double rotation = 0; // Radians, counter clockwise from pointing up.
    double spin = 0; // Radians per second, counter clockwise.

    double radius = 1;

    double left_thrust = 0; // What gets pushed with on the next step, before it gets scaled up by size. Positive is forward.
    double right_thrust = 0;

    bool hit_wall = false; // During the last step.
    int eaten_plants = 0; // During the last step.
    int eaten_meat = 0;

    int holding = -1; // The handle of the body this one has hold of, if any.
    int held_by = 0; // How many others have hold of this one.

    b2BodyId id;
    b2ShapeId shape;
    b2JointId hold;

    float friction = 0; // The water friction that Box2D has for it right now.
};

// The whole map as a Box2D world, with walls all the way around it, and any number of thruster
// bodies in it. Each body is a circle with a thruster on each of its two legs. Forward is +y on the
// body. Unlike projects/island_evolution, the bodies bump in to each other, so they can push each
// other around, and one can take hold of another, which joins them with a short rigid bar from the
// front of the one holding on. The water just changes how much friction a body gets, and food is
// eaten by getting close to it.
//
// Mass goes with area, so a body twice as wide weighs four times as much. Thrust goes with width,
// so bigger bodies are slower to get going and to turn, but get to the same speed on land.
//
// The const functions only read, so any number of threads can use them at once, as long as nothing
// changes the world in the meantime. Anything else is one thread at a time.
class World {
public:
    // right_leg is where the right thruster sits on a body of radius 1. The left one is the mirror image of it.
    World(const Terrain& terrain, double food_radius, Point right_leg, uint64_t seed);
    ~World();

    World(const World&) = delete;
    World& operator=(const World&) = delete;

    // Returns the handle for the new body, which stays the same for as long as it is around.
    int add_body(Point position, double rotation, double radius);

    // Anything that had hold of it lets go.
    void remove_body(int handle);

    void set_radius(int handle, double radius);

    Body& body(int handle) { return bodies.at(handle); }
    const Body& body(int handle) const { return bodies.at(handle); }

    // Grow plants at this many pieces per second, until there are max_plants of them. Anything over
    // the max goes away straight off, so that turning the slider down takes effect.
    void grow_food(double per_second, std::size_t max_plants, double dt);

    // Age the meat, and throw out anything older than the lifetime.
    void rot_meat(double lifetime, double dt);

    // Scatter this many pieces of meat over a circle of this radius.
    void drop_meat(Point center, double spread, int pieces);

    // Has to be called after any change to the food, and before anything looks for food near a spot.
    void index_food();

    // Visit(index into food) for every piece of food within range of a spot. Some of those can be a
    // little further than the range, so that still has to be checked.
    template <typename Visit>
    void visit_food(Point center, double range, Visit visit) const
    {
        food_grid.visit(center, range, visit);
    }

    std::size_t plant_count() const { return plants; }

    // Join the front of the holder to the target, if it isn't holding anything already.
    void hold(int holder, int target);
    void release(int holder);

    // Let go of anything that is getting pulled on harder than strength times the mass of the one
    // holding on. Only means anything right after sim().
    void break_holds(double strength);

    // Simulate every body, using the thrust set on each one. Water friction is how quickly the water
    // slows them down, per second.
    void sim(double water_friction, double dt);

    const Terrain& terrain;

    std::vector<Food> food;

private:
    void place_plant();

    b2WorldId world;

    std::map<int, Body> bodies; // Ordered, so that who gets to the food first doesn't depend on memory layout.
    int next_handle = 0;

    std::mt19937_64 rng;

    double food_radius;
    Point right_leg;

    double food_due = 0; // Fractions of a piece of food, waiting to grow.
    std::size_t plants = 0;

    Grid food_grid;
};

}

#endif // CONTINENTS_PHYSICS_HPP
