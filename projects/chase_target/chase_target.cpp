#include "chase_target.hpp"

#include <algorithm>
#include <cmath>
#include <iterator>
#include <memory>
#include <vector>

namespace chase_target {

constexpr double PI = 3.14159265358979323846;

// How much of each new value goes in to the averages shown in the fields.
constexpr double RECENT_RATE = .01;

ChaseTarget::ChaseTarget(sim::Ui& ui, uint64_t seed)
    : phys_obj({0, 0}, {0, 0}, BOX_SIZE, ORGANISM_RADIUS),
      org(INPUT_SIZE, OUTPUT_SIZE, seed),
      score_graph(ui.add_history_graph({"Score Graph", "Score"})),
      distance_graph(ui.add_history_graph({"Distance Graph", "Average distance to target"})),
      distance_field(ui.add_field({"Recent distance"})),
      score_field(ui.add_field({"Recent score"})),
      wall_punish(ui.add_slider({"Wall punishment", 0, 20, 5, .5})),
      effort_punish(ui.add_slider({"Effort punishment", 0, 1, .1, .01})) {}

double ChaseTarget::time_step() const
{
    return DT;
}

void ChaseTarget::setup_world(plotato::Graph& graph)
{
    graph.equal_aspect_framing = true;
    graph.set_bounds({-BOX_SIZE * 1.2, BOX_SIZE * 1.2, -BOX_SIZE * 1.2, BOX_SIZE * 1.2});
}

void ChaseTarget::step(double t)
{
    // Make the input vector
    // * Top distance raycast
    // * Bottom distance raycast
    // * Left distance raycast
    // * Right distance raycast
    // * Delta X to target
    // * Delta Y to target
    // * X Velocity
    // * Y Velocity

    // Get distance raycasts.
    const RaycastDistances distances = phys_obj.raycast();

    const double x = phys_obj.pos.x;
    const double y = phys_obj.pos.y;

    // Calculate the target position.
    const double phase = 2 * PI * TARGET_SPEED * std::pow(t / 10000, 2);

    target.x = TARGET_AMPLITUDE * std::cos(phase);
    target.y = TARGET_AMPLITUDE * std::sin(phase);

    // Calculate the delta
    const double dx = target.x - x;
    const double dy = target.y - y;

    // Now make the vector. Everything is scaled to roughly [-1, 1], otherwise the raycasts
    // (which always sum to a constant) swamp the delta and velocity inputs during learning.
    const double input_vec[INPUT_SIZE] = {
        distances.top / (2 * BOX_SIZE),
        distances.bottom / (2 * BOX_SIZE),
        distances.left / (2 * BOX_SIZE),
        distances.right / (2 * BOX_SIZE),
        dx / BOX_SIZE,
        dy / BOX_SIZE,
        phys_obj.vel.x / VEL_SCALE,
        phys_obj.vel.y / VEL_SCALE,
    };

    const cdn::Tensor state({INPUT_SIZE, 1}, std::vector<float>(std::begin(input_vec), std::end(input_vec)));

    const cdn::Tensor action = org.run(static_cast<float>(total_score), state);

    std::copy(action.data().begin(), action.data().end(), output_vec.begin());

    const bool hit_wall = phys_obj.sim(output_vec[0], output_vec[1], output_vec[2], output_vec[3], DT);

    // Get distance to target after moving, so the reward scores this action (for scoring)
    dist = std::hypot(target.x - phys_obj.pos.x, target.y - phys_obj.pos.y);

    total_score = 0;

    if (hit_wall) {
        total_score -= wall_punish.value();
    }

    total_score += -dist;

    for (double output : output_vec) {
        total_score -= std::max(0.0, output) * effort_punish.value();
    }

    score_graph.add(total_score);
    distance_graph.add(dist);

    recent_dist += RECENT_RATE * (dist - recent_dist);
    recent_score += RECENT_RATE * (total_score - recent_score);
}

void ChaseTarget::draw(plotato::Graph& world)
{
    constexpr double box = BOX_SIZE;

    plotato::PlotStyle wall_style;
    wall_style.line_color = plotato::Color(0, 0, 255);

    plotato::PlotStyle thrust_style;
    thrust_style.line_color = plotato::Color(255, 0, 0);

    plotato::ShapeStyle organism_style;
    organism_style.fill = plotato::Color(0, 200, 0);
    organism_style.outline = organism_style.fill;

    plotato::ShapeStyle target_style;
    target_style.fill = plotato::Color(255, 0, 0);
    target_style.outline = target_style.fill;

    const double x = phys_obj.pos.x;
    const double y = phys_obj.pos.y;

    world.plot({-box, -box, box, box, -box}, {-box, box, box, -box, -box}, wall_style);

    world.circle(x, y, ORGANISM_RADIUS, organism_style);
    world.circle(target.x, target.y, .1, target_style);

    // Clamped, so these show what the thrusters really did.
    const double up = std::max(0.0, output_vec[0]);
    const double down = std::max(0.0, output_vec[1]);
    const double left = std::max(0.0, output_vec[2]);
    const double right = std::max(0.0, output_vec[3]);

    // Thrusters go on last, so that they show up on top of the organism.
    world.plot({x, x}, {y, y + up}, thrust_style);
    world.plot({x, x}, {y, y - down}, thrust_style);
    world.plot({x, x - left}, {y, y}, thrust_style);
    world.plot({x, x + right}, {y, y}, thrust_style);

    distance_field.set(recent_dist);
    score_field.set(recent_score);
}

}

std::unique_ptr<sim::Project> sim::create_project(sim::Ui& ui, uint64_t seed)
{
    return std::make_unique<chase_target::ChaseTarget>(ui, seed);
}
