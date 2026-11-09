#include "organism_graphic.hpp"

#include <cmath>
#include <utility>

namespace thruster_chase {

constexpr double PI = 3.14159265358979323846;

constexpr double FLAME_SCALE = 3;
constexpr int FLAME_POINTS = 10;

// The flames spread out over this many degrees, around pointing straight down.
constexpr double FLAME_START = -135;
constexpr double FLAME_END = -45;

// The smallest that each point on a flame can randomly get, as a fraction of the full length.
constexpr double FLAME_MIN_NOISE = .2;

const plotato::Color FLAME_COLOR(255, 69, 0);

static double radians(double degrees)
{
    return degrees * PI / 180;
}

OrganismGraphic::OrganismGraphic(plotato::Color color, double size, int n_segment)
    : color(color), size(size), n_segment(n_segment) {}

std::pair<std::vector<double>, std::vector<double>> scale_rotate_offset(std::vector<double> x, std::vector<double> y, double rotation, double size, double base_x, double base_y)
{
    const double cos_rotation = std::cos(rotation);
    const double sin_rotation = std::sin(rotation);

    std::vector<double> out_x;
    std::vector<double> out_y;

    out_x.reserve(x.size());
    out_y.reserve(y.size());

    for (size_t i = 0; i < x.size(); i ++) {

        const double scaled_x = x[i] * size;
        const double scaled_y = (y[i] + .666666) * size; // Rotate around the middle of the thing (lazy).

        out_x.push_back(base_x + scaled_x * cos_rotation - scaled_y * sin_rotation);
        out_y.push_back(base_y + scaled_x * sin_rotation + scaled_y * cos_rotation);

    }

    return std::make_pair(out_x, out_y);
}

std::pair<std::vector<double>, std::vector<double>> OrganismGraphic::flame(double foot_x, double foot_y, double boost, double min_noise)
{
    std::uniform_real_distribution<double> noise(min_noise, 1);

    std::vector<double> x;
    std::vector<double> y;

    for (int i = 0; i < FLAME_POINTS; i++) {

        // Both ends stay on the foot, which is also what closes the loop.
        const bool on_foot = i == 0 || i == FLAME_POINTS - 1;
        const double length = on_foot ? 0 : FLAME_SCALE * boost * noise(flicker);

        const double degrees = FLAME_START + (FLAME_END - FLAME_START) * i / (FLAME_POINTS - 1);

        x.push_back(length * std::cos(radians(degrees)) + foot_x);
        y.push_back(length * std::sin(radians(degrees)) + foot_y);
    }

    return std::make_pair(x, y);
}

void OrganismGraphic::draw(plotato::Graph& world, Point location, double rotation, double boost_left, double boost_right)
{
    const int degree_delta = 180 / n_segment;

    std::vector<double> outline_x;
    std::vector<double> outline_y;

    outline_x.push_back(-1);
    outline_y.push_back(-1);

    for (int i = 1; i < n_segment; i++) {

        const double degrees = 180 - degree_delta * i;

        outline_x.push_back(std::cos(radians(degrees)));
        outline_y.push_back(std::sin(radians(degrees)));
    }

    outline_x.push_back(1);
    outline_y.push_back(-1);

    plotato::Color body_color(0, 0, 255);

    auto rotated_outline = scale_rotate_offset(outline_x, outline_y, rotation, size, location.x, location.y);

    auto body = world.polygon(rotated_outline.first, rotated_outline.second);
    body->style.fill = body_color;
    body->style.outline = plotato::Color(0, 0, 0, 0); // No outline.

    // Draw the legs
    auto leg_1_path = scale_rotate_offset({-RIGHT_FOOT.x, -RIGHT_FOOT.x}, {RIGHT_FOOT.y, -1}, rotation, size, location.x, location.y);
    auto leg_2_path = scale_rotate_offset({RIGHT_FOOT.x, RIGHT_FOOT.x}, {RIGHT_FOOT.y, -1}, rotation, size, location.x, location.y);

    auto leg_1 = world.plot(leg_1_path.first, leg_1_path.second);
    auto leg_2 = world.plot(leg_2_path.first, leg_2_path.second);

    leg_1->style.line_color = body_color;
    leg_2->style.line_color = body_color;

    auto left_flame_unscaled = flame(-RIGHT_FOOT.x, RIGHT_FOOT.y, boost_left, FLAME_MIN_NOISE);
    auto left_flame_scaled = scale_rotate_offset(left_flame_unscaled.first, left_flame_unscaled.second, rotation, size, location.x, location.y);

    auto right_flame_unscaled = flame(RIGHT_FOOT.x, RIGHT_FOOT.y, boost_right, FLAME_MIN_NOISE);
    auto right_flame_scaled = scale_rotate_offset(right_flame_unscaled.first, right_flame_unscaled.second, rotation, size, location.x, location.y);

    plotato::ShapeStyle flame_style;
    flame_style.fill = plotato::Color(255, 150, 0, 150);
    flame_style.outline = plotato::Color(255, 0, 0, 150);

    world.polygon(left_flame_scaled.first, left_flame_scaled.second, flame_style);
    world.polygon(right_flame_scaled.first, right_flame_scaled.second, flame_style);
}

}
