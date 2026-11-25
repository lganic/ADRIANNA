#ifndef FOOD_WEB_GRID_HPP
#define FOOD_WEB_GRID_HPP

#include "geometry.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

namespace food_web {

// Buckets a list of points by where they are, so that finding everything near a spot only has to
// look at a few buckets. The same as projects/continents/grid.hpp: built all at once from the whole
// list, then only read, so any number of threads can look things up at once. Things stay in list
// order inside each bucket, so going through them comes out the same every run.
class Grid {
public:
    Grid(double half_width, double half_height, double cell_size)
        : left(-half_width), bottom(-half_height), cell(cell_size),
          columns(std::max(1, static_cast<int>(std::ceil(2 * half_width / cell_size)))),
          rows(std::max(1, static_cast<int>(std::ceil(2 * half_height / cell_size)))),
          starts(static_cast<std::size_t>(columns * rows + 1), 0)
    {
    }

    // Position(i) gives where the i-th thing is.
    template <typename Position>
    void build(int count, Position position)
    {
        cells.resize(static_cast<std::size_t>(count));
        std::fill(starts.begin(), starts.end(), 0);

        for (int i = 0; i < count; i++) {
            cells[i] = cell_of(position(i));
            starts[cells[i] + 1]++;
        }

        for (std::size_t i = 1; i < starts.size(); i++) {
            starts[i] += starts[i - 1];
        }

        items.resize(static_cast<std::size_t>(count));
        next.assign(starts.begin(), starts.end() - 1);

        for (int i = 0; i < count; i++) {
            items[next[cells[i]]++] = i;
        }
    }

    // Visit(i) for everything in the buckets that a circle of this range around the center touches.
    // Some of those will be further away than the range, so that still has to be checked.
    template <typename Visit>
    void visit(Vec center, double range, Visit visit) const
    {
        const int first_column = column_of(center.x - range);
        const int last_column = column_of(center.x + range);
        const int first_row = row_of(center.y - range);
        const int last_row = row_of(center.y + range);

        for (int row = first_row; row <= last_row; row++) {
            for (int column = first_column; column <= last_column; column++) {
                const int index = row * columns + column;

                for (int item = starts[index]; item < starts[index + 1]; item++) {
                    visit(items[item]);
                }
            }
        }
    }

private:
    int column_of(double x) const { return std::min(columns - 1, std::max(0, static_cast<int>(std::floor((x - left) / cell)))); }
    int row_of(double y) const { return std::min(rows - 1, std::max(0, static_cast<int>(std::floor((y - bottom) / cell)))); }
    int cell_of(Vec p) const { return row_of(p.y) * columns + column_of(p.x); }

    double left;
    double bottom;
    double cell;

    int columns;
    int rows;

    std::vector<int> starts;
    std::vector<int> items;
    std::vector<int> cells;
    std::vector<int> next;
};

}

#endif // FOOD_WEB_GRID_HPP
