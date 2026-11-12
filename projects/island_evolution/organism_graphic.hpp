#ifndef ISLAND_EVOLUTION_ORGANISM_GRAPHIC_HPP
#define ISLAND_EVOLUTION_ORGANISM_GRAPHIC_HPP

#include "physics.hpp"

#include <Plotato/Plotato.hpp>

#include <random>
#include <vector>
#include <utility>

namespace island_evolution {

// Where things are on the drawing, in multiples of its size, when it is pointing up. It is a dome on
// top of two legs, with the feet at the bottom.
constexpr Point RIGHT_FOOT = {1, -2};

// The smallest circle that the whole drawing fits inside. It touches the top of the dome and both feet.
constexpr double DRAWING_CENTER_Y = -2.0 / 3;
constexpr double DRAWING_RADIUS = 5.0 / 3;

// The organism's body, with a flame coming out of each of its feet.
class OrganismGraphic {
public:
    OrganismGraphic(plotato::Color color = plotato::Color(0, 0, 255), double size = .1, int n_segment = 5);

    // Location is the center of the circle above. A negative boost makes a flame point back up the leg.
    void draw(plotato::Graph& world, Point location, double rotation, double boost_left, double boost_right);

    // There is one of these for the whole population, so it gets recoloured for each organism.
    void set_color(plotato::Color new_color) { color = new_color; }

private:
    std::pair<std::vector<double>, std::vector<double>> flame(double foot_x, double foot_y, double boost, double min_noise);

    plotato::Color color;
    double size;
    int n_segment;

    // Its own one, since draw() only runs when the GUI is due a frame. If this shared with the
    // learning, then how a run turns out would depend on the frame rate.
    std::mt19937 flicker;
};

}

#endif // ISLAND_EVOLUTION_ORGANISM_GRAPHIC_HPP
