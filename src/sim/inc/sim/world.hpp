#ifndef SIM_WORLD_HPP
#define SIM_WORLD_HPP

#include <sim/models.hpp>
#include <sim/physics.hpp>

#include <array>
#include <cstdint>

namespace sim {

constexpr double BOX_SIZE = 2;
constexpr double TARGET_AMPLITUDE = 1;
constexpr double TARGET_SPEED = 10;

constexpr double DT = 0.2;

constexpr double VEL_SCALE = 5;

constexpr double WALL_PUNISH = 5;
constexpr double EFFORT_PUNISH = 0.1;

constexpr int INPUT_SIZE = 8;
constexpr int OUTPUT_SIZE = 4;

// One organism in a box, chasing a target. step() is one pass of the loop from tests/test_3/test.py,
// without any of the display stuff. Not thread safe. Only the thread running it should touch it.
class World {
public:
    explicit World(uint64_t seed);

    void step();

    double t = 0;
    Point target;

    PhysicsObject phys_obj;

    std::array<double, OUTPUT_SIZE> output_vec{}; // What the organism asked for on the last step. Up, down, left, right.
    double dist = 0; // Distance to the target after the last step.

private:
    Organism org;

    double total_score = 0;
};

}

#endif // SIM_WORLD_HPP
