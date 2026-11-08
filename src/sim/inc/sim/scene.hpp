#ifndef SIM_SCENE_HPP
#define SIM_SCENE_HPP

#include <cstdint>
#include <utility>
#include <vector>

namespace sim {

struct Color {
    uint8_t r = 0;
    uint8_t g = 0;
    uint8_t b = 0;
};

// One frame of the world view, as a list of things to draw, back to front. This is plain data
// because it gets built on the simulation thread, and drawn later on the GUI thread.
class Scene {
public:
    struct Shape {
        std::vector<double> x;
        std::vector<double> y;
        double radius = 0; // A circle at the first point if this is set, otherwise a line through all the points.
        double line_width = 0;
        Color color;
    };

    void line(std::vector<double> x, std::vector<double> y, Color color, double line_width = 2)
    {
        shapes.push_back({std::move(x), std::move(y), 0, line_width, color});
    }

    void circle(double x, double y, double radius, Color color)
    {
        shapes.push_back({{x}, {y}, radius, 0, color});
    }

    std::vector<Shape> shapes;
};

}

#endif // SIM_SCENE_HPP
