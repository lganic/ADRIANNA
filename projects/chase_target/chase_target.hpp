#ifndef CHASE_TARGET_HPP
#define CHASE_TARGET_HPP

#include "physics.hpp"

#include <sim/models.hpp>
#include <sim/project.hpp>

#include <array>
#include <cstdint>

namespace chase_target {

constexpr double BOX_SIZE = 2;
constexpr double ORGANISM_RADIUS = .2;
constexpr double TARGET_AMPLITUDE = 1;
constexpr double TARGET_SPEED = 10;

constexpr double DT = 0.2;

constexpr double VEL_SCALE = 5;

constexpr int INPUT_SIZE = 8;
constexpr int OUTPUT_SIZE = 4;

// One organism in a box, chasing a target. step() is one pass of the loop from tests/test_3/test.py,
// without any of the display stuff, and with Box2D doing the physics instead of the Euler integration.
class ChaseTarget : public sim::Project {
public:
    ChaseTarget(sim::Ui& ui, uint64_t seed);

    double time_step() const override;
    sim::Bounds world_bounds() const override;

    void step(double t) override;
    void draw(sim::Scene& scene) override;

    Point target;

    PhysicsObject phys_obj;

    std::array<double, OUTPUT_SIZE> output_vec{}; // What the organism asked for on the last step. Up, down, left, right.
    double dist = 0; // Distance to the target after the last step.
    double total_score = 0;

private:
    sim::Organism org;

    sim::HistoryGraph score_graph;
    sim::HistoryGraph distance_graph;

    sim::Field distance_field;
    sim::Field score_field;

    sim::Control wall_punish;
    sim::Control effort_punish;

    double recent_dist = 0;
    double recent_score = 0;
};

}

#endif // CHASE_TARGET_HPP
