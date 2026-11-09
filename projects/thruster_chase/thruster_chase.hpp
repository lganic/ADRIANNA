#ifndef THRUSTER_CHASE_HPP
#define THRUSTER_CHASE_HPP

#include "organism_graphic.hpp"
#include "physics.hpp"

#include <sim/models.hpp>
#include <sim/project.hpp>

#include <array>
#include <cstdint>

namespace thruster_chase {

constexpr double BOX_SIZE = 2;
constexpr double ORGANISM_RADIUS = .2;
constexpr double TARGET_AMPLITUDE = 1;
constexpr double TARGET_SPEED = 10;

// Makes the drawing fit the collider exactly.
constexpr double ORGANISM_SIZE = ORGANISM_RADIUS / DRAWING_RADIUS;

constexpr double DT = 0.2;

constexpr double VEL_SCALE = 5;
constexpr double SPIN_SCALE = 5;

constexpr int INPUT_SIZE = 9;
constexpr int OUTPUT_SIZE = 2;

// The same goal as chase_target, but the organism has to steer. All it has is a thruster on each leg
// that pushes along the way it is facing, so pushing harder on one side is the only way to turn.
class ThrusterChase : public sim::Project {
public:
    ThrusterChase(sim::Ui& ui, uint64_t seed);

    double time_step() const override;
    void setup_world(plotato::Graph& graph) override;

    void step(double t) override;
    void draw(plotato::Graph& world) override;

    Point target;

    PhysicsObject phys_obj;

    std::array<double, OUTPUT_SIZE> output_vec{}; // What the organism asked for on the last step. Left, right.
    double dist = 0; // Distance to the target after the last step.
    double total_score = 0;

private:
    sim::Organism org;

    OrganismGraphic graphic;

    sim::HistoryGraph score_graph;
    sim::HistoryGraph distance_graph;

    sim::Field distance_field;
    sim::Field score_field;

    sim::Control wall_punish;
    sim::Control effort_punish;
    sim::Control spin_punish;

    double recent_dist = 0;
    double recent_score = 0;
};

}

#endif // THRUSTER_CHASE_HPP
