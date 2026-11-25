#ifndef HAIL_MARY_GRID_HPP
#define HAIL_MARY_GRID_HPP

#include "geometry.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

namespace hail_mary {

// Buckets a list of points by where they are, so that finding everything near a spot only has to
// look at a few buckets. The same as projects/continents/grid.hpp: built all at once, then only
// read, and things stay in list order inside each bucket, so going through them is the same every
// run. Except that the map wraps round, so a search near one edge carries on from the other.
class Grid {
public:
    Grid(double half_size, double cell_size)
        : left(-half_size), bottom(-half_size), cell(cell_size),
          columns(std::max(1, static_cast<int>(std::ceil(2 * half_size / cell_size)))),
          rows(columns),
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

    // Visit(i) for everything in the buckets that a circle of this range around the center touches,
    // going round the edges. Some of those are further than the range, so that still has to be
    // checked, the same way round.
    template <typename Visit>
    void visit(Vec center, double range, Visit visit) const
    {
        int first_column = static_cast<int>(std::floor((center.x - range - left) / cell));
        int last_column = static_cast<int>(std::floor((center.x + range - left) / cell));
        int first_row = static_cast<int>(std::floor((center.y - range - bottom) / cell));
        int last_row = static_cast<int>(std::floor((center.y + range - bottom) / cell));

        // Never the same bucket twice, however big the range.
        last_column = std::min(last_column, first_column + columns - 1);
        last_row = std::min(last_row, first_row + rows - 1);

        for (int row = first_row; row <= last_row; row++) {
            for (int column = first_column; column <= last_column; column++) {
                const int index = round(row, rows) * columns + round(column, columns);

                for (int item = starts[index]; item < starts[index + 1]; item++) {
                    visit(items[item]);
                }
            }
        }
    }

private:
    static int round(int value, int size) { return ((value % size) + size) % size; }

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

#endif // HAIL_MARY_GRID_HPP
