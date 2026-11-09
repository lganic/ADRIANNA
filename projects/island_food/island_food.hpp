#ifndef ISLAND_FOOD_HPP
#define ISLAND_FOOD_HPP

#include "organism_graphic.hpp"
#include "physics.hpp"

#include <sim/models.hpp>
#include <sim/project.hpp>

#include <array>
#include <cstdint>

namespace island_food {

constexpr double BOX_SIZE = 100;

// The organism, its thrust (see THRUST_GAIN), and its speeds are all 5x what they are in thruster_chase.
// That way it still handles the same, but it gets across this much bigger world. At the original
// size a single thrust barely changes anything out here, so it had nothing to learn from.
constexpr double ORGANISM_RADIUS = 1;
constexpr double FOOD_RADIUS = 1;

// Makes the drawing fit the collider exactly.
constexpr double ORGANISM_SIZE = ORGANISM_RADIUS / DRAWING_RADIUS;

constexpr double DT = 0.2;

constexpr double WALL_SENSE_RANGE = 15; // Short enough that no island can see a wall from anywhere on it.
constexpr double FOOD_SCALE = 10;
constexpr double VEL_SCALE = 25;
constexpr double SPIN_SCALE = 5;

constexpr int INPUT_SIZE = 11;
constexpr int OUTPUT_SIZE = 2;

// The body from thruster_chase going after the food from find_food, on a group of islands. The target
// is whichever piece of food is closest, and food only ever turns up on land, on any of the islands.
// Going in the water is allowed, but it is slow, and it costs score for as long as it stays in there.
class IslandFood : public sim::Project {
public:
    IslandFood(sim::Ui& ui, uint64_t seed);

    double time_step() const override;
    void setup_world(plotato::Graph& graph) override;

    void step(double t) override;
    void draw(plotato::Graph& graph) override;

    World world;

    Point target;

    std::array<double, OUTPUT_SIZE> output_vec{}; // What the organism asked for on the last step. Left, right.
    double dist = 0; // Distance to the closest food after the last step.
    double total_score = 0;
    uint64_t total_eaten = 0;

private:
    sim::Organism org;

    OrganismGraphic graphic;

    sim::Control food_count;
    sim::Control food_reward;
    sim::Control closeness_reward;
    sim::Control water_punish;
    sim::Control water_friction;
    sim::Control wall_punish;
    sim::Control effort_punish;
    sim::Control spin_punish;

    sim::Field eaten_field;
    sim::Field water_field;
    sim::Field score_field;

    sim::HistoryGraph food_graph;
    sim::HistoryGraph water_graph;
    sim::HistoryGraph distance_graph;
    sim::HistoryGraph score_graph;

    double recent_water = 0;
    double recent_score = 0;
};

}

#endif // ISLAND_FOOD_HPP
