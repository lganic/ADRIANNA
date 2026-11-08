#include "find_food.hpp"

#include <algorithm>
#include <cmath>
#include <iterator>
#include <memory>
#include <vector>

namespace find_food {

constexpr double PI = 3.14159265358979323846;

// The direction of a vision ray, as an angle.
static double ray_angle(int ray)
{
    return 2 * PI * ray / VISION_RAYS;
}

FindFood::FindFood(sim::Ui& ui, uint64_t seed)
    : arena(BOX_SIZE, AGENT_RADIUS, FOOD_RADIUS, seed),
      org(INPUT_SIZE, OUTPUT_SIZE, seed),
      food_count(ui.add_slider({"Food pieces", 1, 12, 5, 1})),
      vision_range(ui.add_slider({"Vision range", .5, 4, 2, .1})),
      food_reward(ui.add_entry({"Food reward", 0, 100, 10, 1})),
      closeness_reward(ui.add_slider({"Closeness reward", 0, 5, 1, .1})),
      effort_punish(ui.add_slider({"Effort punishment", 0, 1, .1, .01})),
      wall_punish(ui.add_slider({"Wall punishment", 0, 20, 5, .5})),
      vision_graph(ui.add_curve_graph({"What the agent can see", "Closeness", sim::Page::World, "Direction (degrees)", true,
                                       sim::Bounds{0, 360.0 * (VISION_RAYS - 1) / VISION_RAYS, 0, 1}})),
      eaten_field(ui.add_field({"Food eaten", sim::Page::Stats, "%.0f"})),
      food_graph(ui.add_history_graph({"Food Graph", "Food per hour"})),
      score_graph(ui.add_history_graph({"Score Graph", "Score", sim::Page::Stats, "Simulated hours", false})) {}

double FindFood::time_step() const
{
    return DT;
}

sim::Bounds FindFood::world_bounds() const
{
    return {-BOX_SIZE * 1.2, BOX_SIZE * 1.2, -BOX_SIZE * 1.2, BOX_SIZE * 1.2};
}

void FindFood::step(double)
{
    arena.set_food_count(static_cast<std::size_t>(food_count.value()));

    // Make the input vector
    // * X and Y of the vision, which is each ray's direction, scaled by how close the food looks along it
    // * Top, bottom, left and right distance raycasts
    // * X Velocity
    // * Y Velocity
    double vision_x = 0;
    double vision_y = 0;

    for (int ray = 0; ray < VISION_RAYS; ray++) {
        vision[ray] = arena.look(std::cos(ray_angle(ray)), std::sin(ray_angle(ray)), vision_range.value());

        vision_x += vision[ray] * std::cos(ray_angle(ray));
        vision_y += vision[ray] * std::sin(ray_angle(ray));
    }

    const double closest = *std::max_element(vision.begin(), vision.end());

    const double input_vec[INPUT_SIZE] = {
        vision_x / (VISION_RAYS / 4.0),
        vision_y / (VISION_RAYS / 4.0),
        arena.wall_distance(0, 1) / (2 * BOX_SIZE),
        arena.wall_distance(0, -1) / (2 * BOX_SIZE),
        arena.wall_distance(-1, 0) / (2 * BOX_SIZE),
        arena.wall_distance(1, 0) / (2 * BOX_SIZE),
        arena.vel.x / VEL_SCALE,
        arena.vel.y / VEL_SCALE,
    };

    const cdn::Tensor state({INPUT_SIZE, 1}, std::vector<float>(std::begin(input_vec), std::end(input_vec)));

    const cdn::Tensor action = org.run(static_cast<float>(total_score), state);

    std::copy(action.data().begin(), action.data().end(), output_vec.begin());

    const SimResult result = arena.sim(output_vec[0], output_vec[1], output_vec[2], output_vec[3], DT);

    total_score = result.eaten * food_reward.value() - closeness_reward.value() * (1 - closest);

    if (result.hit_wall) {
        total_score -= wall_punish.value();
    }

    for (double output : output_vec) {
        total_score -= std::max(0.0, output) * effort_punish.value();
    }

    total_eaten += result.eaten;

    food_graph.add(result.eaten * 3600 / DT);
    score_graph.add(total_score);
}

void FindFood::draw(sim::Scene& scene)
{
    constexpr double box = BOX_SIZE;

    constexpr sim::Color wall_color = {0, 0, 255};
    constexpr sim::Color food_color = {230, 140, 0};
    constexpr sim::Color vision_color = {170, 170, 170};
    constexpr sim::Color agent_color = {0, 200, 0};
    constexpr sim::Color thrust_color = {255, 0, 0};

    const double x = arena.pos.x;
    const double y = arena.pos.y;

    scene.line({-box, -box, box, box, -box}, {-box, box, box, -box, -box}, wall_color);

    for (const Point& food : arena.food) {
        scene.circle(food.x, food.y, FOOD_RADIUS, food_color);
    }

    std::vector<double> degrees;
    std::vector<double> closeness;

    for (int ray = 0; ray < VISION_RAYS; ray++) {

        degrees.push_back(ray_angle(ray) * 180 / PI);
        closeness.push_back(vision[ray]);

        if (vision[ray] > 0) {
            const double seen_at = (1 - vision[ray]) * vision_range.value();

            scene.line({x, x + std::cos(ray_angle(ray)) * seen_at}, {y, y + std::sin(ray_angle(ray)) * seen_at}, vision_color, 1);
        }
    }

    scene.circle(x, y, AGENT_RADIUS, agent_color);

    // Clamped, so these show what the thrusters really did.
    const double up = std::max(0.0, output_vec[0]);
    const double down = std::max(0.0, output_vec[1]);
    const double left = std::max(0.0, output_vec[2]);
    const double right = std::max(0.0, output_vec[3]);

    scene.line({x, x}, {y, y + up}, thrust_color);
    scene.line({x, x}, {y, y - down}, thrust_color);
    scene.line({x, x - left}, {y, y}, thrust_color);
    scene.line({x, x + right}, {y, y}, thrust_color);

    vision_graph.set(std::move(degrees), std::move(closeness));

    eaten_field.set(static_cast<double>(total_eaten));
}

}

std::unique_ptr<sim::Project> sim::create_project(sim::Ui& ui, uint64_t seed)
{
    return std::make_unique<find_food::FindFood>(ui, seed);
}
