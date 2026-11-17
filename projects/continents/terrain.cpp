#include "terrain.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>

namespace continents {

// Meters across one cell of the map. Small next to the bodies, so that coastlines look smooth to them.
constexpr double CELL = 2;

constexpr double COLD_START = 100; // Meters either side of the equator before it starts getting cold.

constexpr double POLE_DEPTH = 65; // How far the polar land reaches in from the north and south walls, on average.

// Every continent keeps this far from the east and west walls, so that no wall runs along a coast.
constexpr double WALL_MARGIN = 20;

// The least sea there is between two different landmasses, wherever they would have touched.
constexpr double CHANNEL = 50;

// How far the coast wanders in and out, in meters, from the smooth shape underneath.
constexpr double COAST_NOISE = 45;
constexpr double POLE_COAST_NOISE = 35;
constexpr double COAST_WAVELENGTH = 110; // Of the biggest bumps.

// How far the shape of a continent gets pushed around before the coast noise goes on, which bends
// the smooth blobs that it is made of in to something less round.
constexpr double WARP = 45;
constexpr double WARP_WAVELENGTH = 140;

// Specks of land smaller than this get sunk, and lakes smaller than this get filled in. In cells.
constexpr int SMALLEST_PIECE = 60;

// Each continent is a few overlapping ovals.
constexpr int CONTINENT_ROWS = 2;
constexpr int CONTINENT_COLUMNS = 3;
constexpr double ROW_SPACING = 240;
constexpr double MIN_CONTINENT_SIZE = 125;
constexpr double MAX_CONTINENT_SIZE = 170;
constexpr int MIN_LOBES = 4;
constexpr int MAX_LOBES = 7;

static uint64_t mix(uint64_t value)
{
    value += 0x9E3779B97F4A7C15ull;
    value = (value ^ (value >> 30)) * 0xBF58476D1CE4E5B9ull;
    value = (value ^ (value >> 27)) * 0x94D049BB133111EBull;

    return value ^ (value >> 31);
}

// A random number from -1 to 1 for each corner of a lattice, always the same for the same corner.
static double lattice(int64_t x, int64_t y, uint64_t seed)
{
    const uint64_t hash = mix(seed ^ mix(static_cast<uint64_t>(x) * 0x632BE59BD9B4E019ull ^ static_cast<uint64_t>(y)));

    return (hash >> 11) * (2.0 / 9007199254740992.0) - 1;
}

static double smooth(double t)
{
    return t * t * (3 - 2 * t);
}

// Smoothly blends between the lattice corners around a spot.
static double value_noise(double x, double y, uint64_t seed)
{
    const double floor_x = std::floor(x);
    const double floor_y = std::floor(y);

    const int64_t cx = static_cast<int64_t>(floor_x);
    const int64_t cy = static_cast<int64_t>(floor_y);

    const double tx = smooth(x - floor_x);
    const double ty = smooth(y - floor_y);

    const double bottom = lattice(cx, cy, seed) + (lattice(cx + 1, cy, seed) - lattice(cx, cy, seed)) * tx;
    const double top = lattice(cx, cy + 1, seed) + (lattice(cx + 1, cy + 1, seed) - lattice(cx, cy + 1, seed)) * tx;

    return bottom + (top - bottom) * ty;
}

// Bumps on top of bumps, each layer half the size of the last. Comes out from about -1 to 1.
static double layered_noise(Point p, double wavelength, int layers, uint64_t seed)
{
    double total = 0;
    double weight = 1;
    double weights = 0;

    for (int layer = 0; layer < layers; layer++) {
        total += weight * value_noise(p.x / wavelength, p.y / wavelength, seed + static_cast<uint64_t>(layer) * 7919);
        weights += weight;

        weight /= 2;
        wavelength /= 2;
    }

    return total / weights;
}

namespace {

struct Lobe {
    Point center;
    double across = 1; // Half widths along its own axes.
    double along = 1;
    double cos_angle = 1;
    double sin_angle = 0;

    // Roughly how far in from its edge a spot is, in meters. Negative outside of it.
    double depth(Point p) const
    {
        const double dx = p.x - center.x;
        const double dy = p.y - center.y;

        const double u = (dx * cos_angle + dy * sin_angle) / across;
        const double v = (-dx * sin_angle + dy * cos_angle) / along;

        return (1 - std::sqrt(u * u + v * v)) * (across + along) / 2;
    }
};

struct Continent {
    std::vector<Lobe> lobes;

    double depth(Point p) const
    {
        double deepest = -1e9;

        for (const Lobe& lobe : lobes) {
            deepest = std::max(deepest, lobe.depth(p));
        }

        return deepest;
    }
};

}

Terrain::Terrain(double half_width, double half_height, uint64_t seed)
    : width(2 * half_width), height(2 * half_height),
      columns(static_cast<int>(std::ceil(width / CELL))),
      rows(static_cast<int>(std::ceil(height / CELL))),
      mass(static_cast<std::size_t>(columns * rows), -1)
{
    generate(seed);
    fill_small_pieces();
    label();
    make_runs();
}

void Terrain::generate(uint64_t seed)
{
    std::mt19937_64 rng(seed);

    const auto uniform = [&rng](double low, double high) { return std::uniform_real_distribution<double>(low, high)(rng); };

    std::vector<Continent> continents;

    const double column_spacing = (half_width() - WALL_MARGIN) * 2 / CONTINENT_COLUMNS;

    for (int row = 0; row < CONTINENT_ROWS; row++) {
        for (int column = 0; column < CONTINENT_COLUMNS; column++) {
            const double size = uniform(MIN_CONTINENT_SIZE, MAX_CONTINENT_SIZE);

            const Point center = {
                (column - (CONTINENT_COLUMNS - 1) / 2.0) * column_spacing + uniform(-55, 55),
                (row - (CONTINENT_ROWS - 1) / 2.0) * ROW_SPACING + uniform(-35, 35),
            };

            Continent continent;

            const auto add_lobe = [&](Point at, double across, double along) {
                const double angle = uniform(0, PI);

                continent.lobes.push_back({at, across, along, std::cos(angle), std::sin(angle)});
            };

            add_lobe(center, size * uniform(.65, .85), size * uniform(.55, .8));

            const int extra = std::uniform_int_distribution<int>(MIN_LOBES, MAX_LOBES)(rng) - 1;

            for (int lobe = 0; lobe < extra; lobe++) {
                const double angle = uniform(0, 2 * PI);
                const double distance = size * uniform(.35, .8);

                add_lobe({center.x + distance * std::cos(angle), center.y + distance * std::sin(angle)}, size * uniform(.25, .55), size * uniform(.25, .55));
            }

            continents.push_back(continent);
        }
    }

    const uint64_t noise_seed = mix(seed);
    const double pole_edge = half_height() - POLE_DEPTH;

    for (int row = 0; row < rows; row++) {
        for (int column = 0; column < columns; column++) {
            const Point p = {-half_width() + (column + .5) * CELL, -half_height() + (row + .5) * CELL};

            const Point warped = {
                p.x + WARP * layered_noise(p, WARP_WAVELENGTH, 3, noise_seed + 1),
                p.y + WARP * layered_noise(p, WARP_WAVELENGTH, 3, noise_seed + 2),
            };

            const double coast = layered_noise(p, COAST_WAVELENGTH, 4, noise_seed + 3);

            // Continents sink away towards the east and west walls, so that the coast bends away from
            // the wall instead of getting cut off straight along it.
            const double from_wall = half_width() - std::abs(p.x);
            const double sinking = 1.5 * std::max(0.0, 3 * WALL_MARGIN - from_wall);

            // How high above the sea each piece of land would put this spot. The two poles come after the continents.
            std::vector<double> heights;

            for (const Continent& continent : continents) {
                heights.push_back(continent.depth(warped) + COAST_NOISE * coast - sinking);
            }

            heights.push_back(p.y - pole_edge + POLE_COAST_NOISE * coast);
            heights.push_back(-p.y - pole_edge + POLE_COAST_NOISE * coast);

            std::vector<std::size_t> order(heights.size());
            std::iota(order.begin(), order.end(), 0);
            std::partial_sort(order.begin(), order.begin() + 2, order.end(), [&](std::size_t a, std::size_t b) { return heights[a] > heights[b]; });

            const double highest = heights[order[0]];
            const double second = heights[order[1]];

            const bool is_continent = order[0] < continents.size();

            // Both heights fall by about a meter for every meter towards the other, so this cuts a
            // channel about CHANNEL wide wherever two of them meet.
            const bool land = highest > 0 && highest - second > CHANNEL && !(is_continent && std::abs(p.x) > half_width() - WALL_MARGIN);

            mass[row * columns + column] = land ? 0 : -1;
        }
    }
}

// Flood fill from a cell, over every cell with the same value. Returns the cells it reached.
static std::vector<int> flood(const std::vector<int>& grid, int columns, int rows, int start, std::vector<char>& seen)
{
    std::vector<int> piece;
    std::vector<int> stack = {start};

    seen[start] = 1;

    const int value = grid[start];

    while (!stack.empty()) {
        const int cell = stack.back();
        stack.pop_back();

        piece.push_back(cell);

        const int column = cell % columns;
        const int row = cell / columns;

        const int neighbors[4][2] = {{column - 1, row}, {column + 1, row}, {column, row - 1}, {column, row + 1}};

        for (const auto& neighbor : neighbors) {
            if (neighbor[0] < 0 || neighbor[0] >= columns || neighbor[1] < 0 || neighbor[1] >= rows) {
                continue;
            }

            const int next = neighbor[1] * columns + neighbor[0];

            if (!seen[next] && grid[next] == value) {
                seen[next] = 1;
                stack.push_back(next);
            }
        }
    }

    return piece;
}

void Terrain::fill_small_pieces()
{
    // Land first, since sinking a speck can only ever make the sea around it bigger.
    for (int value : {0, -1}) {
        std::vector<char> seen(mass.size(), 0);

        for (std::size_t cell = 0; cell < mass.size(); cell++) {
            if (seen[cell] || mass[cell] != value) {
                continue;
            }

            const std::vector<int> piece = flood(mass, columns, rows, static_cast<int>(cell), seen);

            // The open sea always reaches the walls, so a lake is water that doesn't.
            const bool touches_edge = std::any_of(piece.begin(), piece.end(), [&](int c) {
                const int column = c % columns;
                const int row = c / columns;

                return column == 0 || row == 0 || column == columns - 1 || row == rows - 1;
            });

            const bool lake = value == -1 && !touches_edge;

            if (static_cast<int>(piece.size()) < SMALLEST_PIECE && (value == 0 || lake)) {
                for (int c : piece) {
                    mass[c] = value == 0 ? -1 : 0;
                }
            }
        }
    }
}

void Terrain::label()
{
    std::vector<char> seen(mass.size(), 0);
    std::vector<std::vector<int>> pieces;

    for (std::size_t cell = 0; cell < mass.size(); cell++) {
        if (!seen[cell] && mass[cell] == 0) {
            pieces.push_back(flood(mass, columns, rows, static_cast<int>(cell), seen));
        }
    }

    // Biggest first, and then by where they start, so that the order is always the same.
    std::stable_sort(pieces.begin(), pieces.end(), [](const std::vector<int>& a, const std::vector<int>& b) { return a.size() > b.size(); });

    for (std::size_t id = 0; id < pieces.size(); id++) {
        std::vector<int> cells = pieces[id];
        std::sort(cells.begin(), cells.end());

        for (int c : cells) {
            mass[c] = static_cast<int>(id);
        }

        areas.push_back(static_cast<double>(cells.size()) * CELL * CELL);
        cells_by_mass.push_back(cells);
    }

    for (std::size_t cell = 0; cell < mass.size(); cell++) {
        if (mass[cell] >= 0) {
            land_cells.push_back(static_cast<int>(cell));
        }
    }
}

void Terrain::make_runs()
{
    for (int row = 0; row < rows; row++) {
        const double bottom = -half_height() + row * CELL;
        const double row_cold = cold(bottom + CELL / 2);

        int column = 0;

        while (column < columns) {
            if (mass[row * columns + column] < 0) {
                column++;
                continue;
            }

            const int start = column;

            while (column < columns && mass[row * columns + column] >= 0) {
                column++;
            }

            // A little taller than the row, so that the rows overlap, and the sea doesn't show
            // through in a thin line where the edges of two of them get smoothed.
            drawing.push_back({-half_width() + start * CELL, -half_width() + column * CELL, bottom, bottom + CELL * 1.25, row_cold});
        }
    }
}

int Terrain::cell_at(Point p) const
{
    const int column = static_cast<int>(std::floor((p.x + half_width()) / CELL));
    const int row = static_cast<int>(std::floor((p.y + half_height()) / CELL));

    if (column < 0 || column >= columns || row < 0 || row >= rows) {
        return -1;
    }

    return row * columns + column;
}

int Terrain::landmass(Point p) const
{
    const int cell = cell_at(p);

    return cell < 0 ? -1 : mass[cell];
}

double Terrain::land_area() const
{
    return std::accumulate(areas.begin(), areas.end(), 0.0);
}

double Terrain::cold(double y) const
{
    return clamp((std::abs(y) - COLD_START) / (half_height() - COLD_START), 0, 1);
}

double Terrain::boundary_distance(Point p, Point dir, double range) const
{
    // Half a cell at a time, which is close enough for something that only gets seen from a distance.
    constexpr double STEP = CELL / 2;

    const bool start = is_land(p);

    for (double travelled = STEP; travelled <= range; travelled += STEP) {
        const Point q = {p.x + dir.x * travelled, p.y + dir.y * travelled};

        if (std::abs(q.x) >= half_width() || std::abs(q.y) >= half_height() || is_land(q) != start) {
            return travelled;
        }
    }

    return range;
}

static Point spot_in(int cell, int columns, double half_width, double half_height, std::mt19937_64& rng)
{
    std::uniform_real_distribution<double> within(0, CELL);

    const double x = -half_width + (cell % columns) * CELL + within(rng);
    const double y = -half_height + (cell / columns) * CELL + within(rng);

    return {x, y};
}

Point Terrain::random_land(std::mt19937_64& rng) const
{
    const int cell = land_cells[std::uniform_int_distribution<std::size_t>(0, land_cells.size() - 1)(rng)];

    return spot_in(cell, columns, half_width(), half_height(), rng);
}

Point Terrain::random_land_on(int landmass, std::mt19937_64& rng) const
{
    const std::vector<int>& cells = cells_by_mass.at(landmass);
    const int cell = cells[std::uniform_int_distribution<std::size_t>(0, cells.size() - 1)(rng)];

    return spot_in(cell, columns, half_width(), half_height(), rng);
}

}
