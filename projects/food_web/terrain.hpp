#ifndef FOOD_WEB_TERRAIN_HPP
#define FOOD_WEB_TERRAIN_HPP

#include "geometry.hpp"

#include <cstdint>
#include <random>
#include <vector>

namespace food_web {

// One stretch of land along a row of the map, for drawing.
struct LandRun {
    double left = 0;
    double right = 0;
    double bottom = 0;
    double top = 0;
};

// The map: four big continents with rough coastlines, and sea everywhere else, out to walls all the
// way round. The same generator as projects/continents/terrain.cpp, made bigger, two by two, and
// without the poles. Wherever two continents would run in to each other, a channel of sea gets cut
// between them, so the only way from one to another is to swim.
//
// Each separate piece of land is numbered as a landmass, biggest first, which is how the stats tell
// the continents apart. Everything in here is const once it is made, so any number of threads can
// use it at once.
class Terrain {
public:
    // The same seed always makes the same map.
    Terrain(double half_width, double half_height, uint64_t seed);

    double half_width() const { return width / 2; }
    double half_height() const { return height / 2; }

    bool is_land(Vec p) const { return landmass(p) >= 0; }

    // Which landmass this spot is on, or -1 for the sea, which includes anywhere outside the walls.
    int landmass(Vec p) const;

    int landmass_count() const { return static_cast<int>(areas.size()); }
    double landmass_area(int landmass) const { return areas.at(static_cast<std::size_t>(landmass)); }

    // A spot on the given landmass, every square meter of it being just as likely.
    Vec random_land_on(int landmass, std::mt19937_64& rng) const;

    const std::vector<LandRun>& runs() const { return drawing; }

private:
    int cell_at(Vec p) const; // -1 if it is outside the walls.

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
    std::vector<std::vector<int>> cells_by_mass;

    std::vector<LandRun> drawing;
};

}

#endif // FOOD_WEB_TERRAIN_HPP
