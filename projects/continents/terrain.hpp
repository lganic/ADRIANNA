#ifndef CONTINENTS_TERRAIN_HPP
#define CONTINENTS_TERRAIN_HPP

#include "geometry.hpp"

#include <cstdint>
#include <random>
#include <vector>

namespace continents {

// One stretch of land along a row of the map, for drawing.
struct LandRun {
    double left = 0;
    double right = 0;
    double bottom = 0;
    double top = 0;
    double cold = 0;
};

// The map: a few big continents with rough coastlines, a strip of land along the north and south
// edges for the poles, and sea everywhere else. It is kept as a grid of cells, each of which is all
// land or all water.
//
// The continents never touch. Wherever two of them would have run in to each other, a channel of
// sea gets cut between them, so the only way from one to another is to swim. Each separate piece
// of land is numbered as a landmass, biggest first, which is how the stats tell the continents apart.
//
// Cold only depends on how far north or south something is. It is nothing up to COLD_START either
// side of the equator, and then goes up evenly to 1 at the walls.
//
// Everything in here is const once it is made, so any number of threads can use it at once.
class Terrain {
public:
    // The same seed always makes the same map.
    Terrain(double half_width, double half_height, uint64_t seed);

    double half_width() const { return width / 2; }
    double half_height() const { return height / 2; }

    bool is_land(Point p) const { return landmass(p) >= 0; }

    // Which landmass this spot is on, or -1 for the sea, which includes anywhere outside the walls.
    int landmass(Point p) const;

    int landmass_count() const { return static_cast<int>(areas.size()); }
    double landmass_area(int landmass) const { return areas.at(landmass); }
    double land_area() const;

    // From 0 for the warm middle of the map, up to 1 at the north and south walls.
    double cold(double y) const;

    // How far it is going from p along the unit vector dir, until it goes from land to water or
    // back, or reaches a wall, whichever is first. Range if there is nothing in that far.
    double boundary_distance(Point p, Point dir, double range) const;

    // A spot on land somewhere, every square meter of land being just as likely.
    Point random_land(std::mt19937_64& rng) const;

    // Same, but only on the given landmass.
    Point random_land_on(int landmass, std::mt19937_64& rng) const;

    const std::vector<LandRun>& runs() const { return drawing; }

private:
    int cell_at(Point p) const; // -1 if it is outside the walls.

    void generate(uint64_t seed);
    void fill_small_pieces();
    void label();
    void make_runs();

    double width;
    double height;

    int columns;
    int rows;

    std::vector<int> mass; // One per cell, by rows from the bottom left. The landmass, or -1 for water.

    std::vector<double> areas; // Of each landmass.
    std::vector<int> land_cells; // Every cell of land, so that picking one at random is fair by area.
    std::vector<std::vector<int>> cells_by_mass;

    std::vector<LandRun> drawing;
};

}

#endif // CONTINENTS_TERRAIN_HPP
