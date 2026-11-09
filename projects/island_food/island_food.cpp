#include "island_food.hpp"

#include <algorithm>
#include <cmath>
#include <iterator>
#include <memory>
#include <vector>

namespace island_food {

// Left, right, bottom, top. Everything else goes by this list, so more islands can just get added here.
// The organism starts at the origin, so the first one needs to stay where it is to start out on land.
const std::vector<Rect> ISLANDS = {
    {-25, 25, -25, 25},
    {-80, -50, 40, 70},
    {50, 85, 35, 65},
    {-75, -45, -75, -50},
    {45, 70, -80, -45},
};

// How much of each new value goes in to the averages shown in the fields.
constexpr double RECENT_RATE = .01;

// Where the right thruster sits, relative to the middle of the organism.
constexpr Point RIGHT_LEG = {ORGANISM_SIZE * RIGHT_FOOT.x, ORGANISM_SIZE * (RIGHT_FOOT.y - DRAWING_CENTER_Y)};

// Thrust past this doesn't make the flames any bigger, so that a wild output can't fill the screen.
constexpr double MAX_FLAME_BOOST = 2;

// Shrinks a vector down to length 1 if it is any longer, which keeps it pointing the same way.
static Point limit_length(Point vector)
{
    const double length = std::hypot(vector.x, vector.y);

    return length > 1 ? Point{vector.x / length, vector.y / length} : vector;
}

// Goes from 0 when the wall is out of range, up to 1 when right on top of it. Only seeing walls from
// close up matters on an island away from the middle. Full length rays read very differently
// depending on which way the organism faces out there, which reaches the thrusters as bogus steering.
static double wall_closeness(double distance)
{
    return std::max(0.0, 1 - distance / WALL_SENSE_RANGE);
}

IslandFood::IslandFood(sim::Ui& ui, uint64_t seed)
    : world({0, 0}, BOX_SIZE, ORGANISM_RADIUS, FOOD_RADIUS, RIGHT_LEG, ISLANDS, seed),
      org(INPUT_SIZE, OUTPUT_SIZE, seed),
      graphic(plotato::Color(0, 0, 255), ORGANISM_SIZE),
      food_count(ui.add_slider({"Food pieces", 1, 100, 50, 1})),
      food_reward(ui.add_entry({"Food reward", 0, 100, 10, 1})),
      closeness_reward(ui.add_slider({"Closeness reward", 0, 5, 1, .1})),
      water_punish(ui.add_slider({"Water punishment", 0, 20, 5, .5})),
      water_friction(ui.add_slider({"Water friction", 0, 10, 2, .1})),
      wall_punish(ui.add_slider({"Wall punishment", 0, 20, 5, .5})),
      effort_punish(ui.add_slider({"Effort punishment", 0, 1, .1, .01})),
      spin_punish(ui.add_slider({"Spin punishment", 0, 1, .2, .01})),
      eaten_field(ui.add_field({"Food eaten", sim::Page::World, "%.0f"})),
      water_field(ui.add_field({"Recent time in water", sim::Page::World, "%.3f"})),
      score_field(ui.add_field({"Recent score"})),
      food_graph(ui.add_history_graph({"Food Graph", "Food per hour"})),
      water_graph(ui.add_history_graph({"Water Graph", "Fraction of time in water"})),
      distance_graph(ui.add_history_graph({"Distance Graph", "Average distance to food"})),
      score_graph(ui.add_history_graph({"Score Graph", "Score"})) {}

double IslandFood::time_step() const
{
    return DT;
}

void IslandFood::setup_world(plotato::Graph& graph)
{
    graph.equal_aspect_framing = true;
    graph.set_bounds({-BOX_SIZE * 1.2, BOX_SIZE * 1.2, -BOX_SIZE * 1.2, BOX_SIZE * 1.2});
}

void IslandFood::step(double)
{
    world.set_food_count(static_cast<std::size_t>(food_count.value()));

    // Make the input vector. All of this is from the organism's point of view, since the thrusters
    // only push along the way it is facing. See the note on this in projects/thruster_chase.
    // * How close the wall is in front
    // * How close the wall is behind
    // * How close the wall is to the left
    // * How close the wall is to the right
    // * How far the food is to the right
    // * How far the food is ahead
    // * Velocity to the right
    // * Velocity ahead
    // * Spin
    // * Whether it is in the water
    // * A constant, since the actor has no bias term of its own

    const RaycastDistances distances = world.raycast();

    target = world.nearest_food();

    const Point to_food = world.to_body({target.x - world.pos.x, target.y - world.pos.y});
    const Point velocity = world.to_body(world.vel);

    // How far each update moves the weights goes up with the size of the inputs, so these have to be
    // kept small, or else speed feeds itself. Limiting the whole vector instead of each part of it
    // matters out at sea. With everything at full size at once, the critic overshoots, and blows up.
    const Point food_input = limit_length({to_food.x / FOOD_SCALE, to_food.y / FOOD_SCALE});
    const Point velocity_input = limit_length({velocity.x / VEL_SCALE, velocity.y / VEL_SCALE});

    double input_vec[INPUT_SIZE] = {
        wall_closeness(distances.front),
        wall_closeness(distances.back),
        wall_closeness(distances.left),
        wall_closeness(distances.right),
        food_input.x,
        food_input.y,
        velocity_input.x,
        velocity_input.y,
        std::min(1.0, std::max(-1.0, world.spin / SPIN_SCALE)),
        world.in_water() ? 1.0 : 0.0,
        1.0,
    };

    const cdn::Tensor state({INPUT_SIZE, 1}, std::vector<float>(std::begin(input_vec), std::end(input_vec)));

    const cdn::Tensor action = org.run(static_cast<float>(total_score), state);

    std::copy(action.data().begin(), action.data().end(), output_vec.begin());

    const SimResult result = world.sim(output_vec[0], output_vec[1], water_friction.value(), DT);

    // Going again, since what was closest may have just been eaten. This is after moving, so that the
    // reward scores this action. It gets capped where the input does, so that the critic can see all of it.
    target = world.nearest_food();

    dist = std::hypot(target.x - world.pos.x, target.y - world.pos.y);

    const bool in_water = world.in_water();

    total_score = result.eaten * food_reward.value() - closeness_reward.value() * std::min(1.0, dist / FOOD_SCALE);

    if (in_water) {
        total_score -= water_punish.value();
    }

    if (result.hit_wall) {
        total_score -= wall_punish.value();
    }

    // Thrust in either direction costs the same.
    for (double output : output_vec) {
        total_score -= std::abs(output) * effort_punish.value();
    }

    // Without this, most organisms never learn to steer. See the note on this in projects/thruster_chase.
    total_score -= std::abs(world.spin) * spin_punish.value();

    total_eaten += result.eaten;

    food_graph.add(result.eaten * 3600 / DT);
    water_graph.add(in_water ? 1 : 0);
    distance_graph.add(dist);
    score_graph.add(total_score);

    recent_water += RECENT_RATE * ((in_water ? 1 : 0) - recent_water);
    recent_score += RECENT_RATE * (total_score - recent_score);
}

static double flame_boost(double thrust)
{
    return std::min(MAX_FLAME_BOOST, std::max(-MAX_FLAME_BOOST, thrust));
}

void IslandFood::draw(plotato::Graph& graph)
{
    constexpr double box = BOX_SIZE;

    plotato::ShapeStyle water_style;
    water_style.fill = plotato::Color(170, 215, 245);
    water_style.outline = plotato::Color(0, 0, 0, 0); // No outline.

    plotato::ShapeStyle land_style;
    land_style.fill = plotato::Color(235, 215, 160);
    land_style.outline = plotato::Color(0, 0, 0, 0); // No outline.

    plotato::PlotStyle wall_style;
    wall_style.line_color = plotato::Color(0, 0, 255);

    plotato::ShapeStyle food_style;
    food_style.fill = plotato::Color(230, 140, 0);
    food_style.outline = food_style.fill;

    plotato::ShapeStyle target_style;
    target_style.fill = plotato::Color(255, 0, 0);
    target_style.outline = target_style.fill;

    graph.polygon({-box, -box, box, box}, {-box, box, box, -box}, water_style);

    for (const Rect& island : world.island_list()) {
        graph.polygon({island.left, island.left, island.right, island.right}, {island.bottom, island.top, island.top, island.bottom}, land_style);
    }

    graph.plot({-box, -box, box, box, -box}, {-box, box, box, -box, -box}, wall_style);

    for (const Point& piece : world.food) {
        graph.circle(piece.x, piece.y, FOOD_RADIUS, food_style);
    }

    graph.circle(target.x, target.y, FOOD_RADIUS, target_style);

    graphic.draw(graph, world.pos, world.rotation, flame_boost(output_vec[0]), flame_boost(output_vec[1]));

    eaten_field.set(static_cast<double>(total_eaten));
    water_field.set(recent_water);
    score_field.set(recent_score);
}

}

std::unique_ptr<sim::Project> sim::create_project(sim::Ui& ui, uint64_t seed)
{
    return std::make_unique<island_food::IslandFood>(ui, seed);
}
