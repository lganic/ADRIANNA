#ifndef CONTINENTS_GRID_HPP
#define CONTINENTS_GRID_HPP

#include "geometry.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

namespace continents {

// Buckets a list of points by where they are, so that finding everything near a spot only has to
// look at a few buckets, instead of at everything. It gets built all at once from the whole list,
// and then only read, so any number of threads can look things up in it at the same time.
//
// Inside each bucket, things stay in the order they were in the list, so anything that goes through
// them in order comes out the same every run.
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

        std::vector<int> next(starts.begin(), starts.end() - 1);

        for (int i = 0; i < count; i++) {
            items[next[cells[i]]++] = i;
        }
    }

    // Visit(i) for everything in the buckets that a circle of this range around the center touches.
    // Some of those will be further away than the range, so that still has to be checked.
    template <typename Visit>
    void visit(Point center, double range, Visit visit) const
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
    int cell_of(Point p) const { return row_of(p.y) * columns + column_of(p.x); }

    double left;
    double bottom;
    double cell;

    int columns;
    int rows;

    std::vector<int> starts; // Where each bucket starts in items, plus one more for where the last one ends.
    std::vector<int> items;
    std::vector<int> cells; // Which bucket each thing went in to, kept around so that building doesn't allocate.
};

}

#endif // CONTINENTS_GRID_HPP
