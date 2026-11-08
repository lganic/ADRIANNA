#ifndef FIND_FOOD_HPP
#define FIND_FOOD_HPP

#include "physics.hpp"

#include <sim/models.hpp>
#include <sim/project.hpp>

#include <array>
#include <cstdint>

namespace find_food {

constexpr double BOX_SIZE = 2;
constexpr double AGENT_RADIUS = .2;
constexpr double FOOD_RADIUS = .12;

constexpr double DT = 0.2;

constexpr double VEL_SCALE = 5;

constexpr int VISION_RAYS = 8;

constexpr int INPUT_SIZE = 2 + 4 + 2; // Vision, then the walls, then velocity.
constexpr int OUTPUT_SIZE = 4;

// One agent in a box, which can only see so far, looking for food. The vision rays get folded into
// one vector pointing at the food, and there is a reward on every step for how close the food looks.
// Linear layers need both of those. With raw rays, or with a reward for eating alone, nearly
// every run just drifts in to a wall and stops.
class FindFood : public sim::Project {
public:
    FindFood(sim::Ui& ui, uint64_t seed);

    double time_step() const override;
    void setup_world(plotato::Graph& graph) override;

    void step(double t) override;
    void draw(plotato::Graph& world) override;

    Arena arena;

    std::array<double, VISION_RAYS> vision{};
    std::array<double, OUTPUT_SIZE> output_vec{}; // What the agent asked for on the last step. Up, down, left, right.

    double total_score = 0;
    uint64_t total_eaten = 0;

private:
    sim::Organism org;

    sim::Control food_count;
    sim::Control vision_range;
    sim::Control food_reward;
    sim::Control closeness_reward;
    sim::Control effort_punish;
    sim::Control wall_punish;

    sim::LiveGraph vision_graph;

    sim::Field eaten_field;

    sim::HistoryGraph food_graph;
    sim::HistoryGraph score_graph;
};

}

#endif // FIND_FOOD_HPP
