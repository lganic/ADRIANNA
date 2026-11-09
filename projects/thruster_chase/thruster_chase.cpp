#include "thruster_chase.hpp"

#include <algorithm>
#include <cmath>
#include <iterator>
#include <memory>
#include <vector>

namespace thruster_chase {

constexpr double PI = 3.14159265358979323846;

// How much of each new value goes in to the averages shown in the fields.
constexpr double RECENT_RATE = .01;

// Where the right thruster sits, relative to the middle of the organism.
constexpr Point RIGHT_LEG = {ORGANISM_SIZE * RIGHT_FOOT.x, ORGANISM_SIZE * (RIGHT_FOOT.y - DRAWING_CENTER_Y)};

// Thrust past this doesn't make the flames any bigger, so that a wild output can't fill the screen.
constexpr double MAX_FLAME_BOOST = 2;

ThrusterChase::ThrusterChase(sim::Ui& ui, uint64_t seed)
    : phys_obj({0, 0}, {0, 0}, BOX_SIZE, ORGANISM_RADIUS, RIGHT_LEG),
      org(INPUT_SIZE, OUTPUT_SIZE, seed),
      graphic(plotato::Color(0, 0, 255), ORGANISM_SIZE),
      score_graph(ui.add_history_graph({"Score Graph", "Score"})),
      distance_graph(ui.add_history_graph({"Distance Graph", "Average distance to target"})),
      distance_field(ui.add_field({"Recent distance"})),
      score_field(ui.add_field({"Recent score"})),
      wall_punish(ui.add_slider({"Wall punishment", 0, 20, 5, .5})),
      effort_punish(ui.add_slider({"Effort punishment", 0, 1, .1, .01})),
      spin_punish(ui.add_slider({"Spin punishment", 0, 1, .2, .01})) {}

double ThrusterChase::time_step() const
{
    return DT;
}

void ThrusterChase::setup_world(plotato::Graph& graph)
{
    graph.equal_aspect_framing = true;
    graph.set_bounds({-BOX_SIZE * 1.2, BOX_SIZE * 1.2, -BOX_SIZE * 1.2, BOX_SIZE * 1.2});
}

void ThrusterChase::step(double t)
{
    // Make the input vector. All of this is from the organism's point of view, since the thrusters
    // only push along the way it is facing. The actor is linear, so it could never work out how
    // to steer from world directions, which would need multiplying them by the heading.
    // * Front distance raycast
    // * Back distance raycast
    // * Left distance raycast
    // * Right distance raycast
    // * How far the target is to the right
    // * How far the target is ahead
    // * Velocity to the right
    // * Velocity ahead
    // * Spin

    // Get distance raycasts.
    const RaycastDistances distances = phys_obj.raycast();

    // Calculate the target position.
    const double phase = 2 * PI * TARGET_SPEED * std::pow(t / 10000, 2);

    target.x = TARGET_AMPLITUDE * std::cos(phase);
    target.y = TARGET_AMPLITUDE * std::sin(phase);

    const Point delta = phys_obj.to_body({target.x - phys_obj.pos.x, target.y - phys_obj.pos.y});
    const Point velocity = phys_obj.to_body(phys_obj.vel);

    // Now make the vector. Everything is scaled to roughly [-1, 1], otherwise the raycasts swamp
    // the delta and velocity inputs during learning.
    double input_vec[INPUT_SIZE] = {
        distances.front / (2 * BOX_SIZE),
        distances.back / (2 * BOX_SIZE),
        distances.left / (2 * BOX_SIZE),
        distances.right / (2 * BOX_SIZE),
        delta.x / BOX_SIZE,
        delta.y / BOX_SIZE,
        velocity.x / VEL_SCALE,
        velocity.y / VEL_SCALE,
        phys_obj.spin / SPIN_SCALE,
    };

    // How far each update moves the weights goes up with the size of the inputs. Left alone, a high
    // speed makes for a big update, which makes for a bigger action and a higher speed, until the
    // outputs are in the tens of thousands. That took out over a quarter of the seeds without this.
    for (double& input : input_vec) {
        input = std::min(1.0, std::max(-1.0, input));
    }

    const cdn::Tensor state({INPUT_SIZE, 1}, std::vector<float>(std::begin(input_vec), std::end(input_vec)));

    const cdn::Tensor action = org.run(static_cast<float>(total_score), state);

    std::copy(action.data().begin(), action.data().end(), output_vec.begin());

    const bool hit_wall = phys_obj.sim(output_vec[0], output_vec[1], DT);

    // Get distance to target after moving, so the reward scores this action (for scoring)
    dist = std::hypot(target.x - phys_obj.pos.x, target.y - phys_obj.pos.y);

    total_score = 0;

    if (hit_wall) {
        total_score -= wall_punish.value();
    }

    total_score += -dist;

    // Thrust in either direction costs the same.
    for (double output : output_vec) {
        total_score -= std::abs(output) * effort_punish.value();
    }

    // Without this, most organisms never learn to steer. They spin constantly instead, which sweeps
    // the thrust around every direction, and pulse it as the target goes past.
    total_score -= std::abs(phys_obj.spin) * spin_punish.value();

    score_graph.add(total_score);
    distance_graph.add(dist);

    recent_dist += RECENT_RATE * (dist - recent_dist);
    recent_score += RECENT_RATE * (total_score - recent_score);
}

static double flame_boost(double thrust)
{
    return std::min(MAX_FLAME_BOOST, std::max(-MAX_FLAME_BOOST, thrust));
}

void ThrusterChase::draw(plotato::Graph& world)
{
    constexpr double box = BOX_SIZE;

    plotato::PlotStyle wall_style;
    wall_style.line_color = plotato::Color(0, 0, 255);

    plotato::ShapeStyle target_style;
    target_style.fill = plotato::Color(255, 0, 0);
    target_style.outline = target_style.fill;

    world.plot({-box, -box, box, box, -box}, {-box, box, box, -box, -box}, wall_style);

    world.circle(target.x, target.y, .1, target_style);

    graphic.draw(world, phys_obj.pos, phys_obj.rotation, flame_boost(output_vec[0]), flame_boost(output_vec[1]));

    distance_field.set(recent_dist);
    score_field.set(recent_score);
}

}

std::unique_ptr<sim::Project> sim::create_project(sim::Ui& ui, uint64_t seed)
{
    return std::make_unique<thruster_chase::ThrusterChase>(ui, seed);
}
