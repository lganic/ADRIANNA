#ifndef CONTINENTS_GEOMETRY_HPP
#define CONTINENTS_GEOMETRY_HPP

#include <algorithm>
#include <cmath>

namespace continents {

constexpr double PI = 3.14159265358979323846;

struct Point {
    double x = 0;
    double y = 0;
};

// Shrinks a vector down to length 1 if it is any longer, which keeps it pointing the same way.
inline Point limit_length(Point vector)
{
    const double length = std::hypot(vector.x, vector.y);

    return length > 1 ? Point{vector.x / length, vector.y / length} : vector;
}

// A NaN comes out as 0, since std::min and std::max would let it straight through.
inline double clamp(double value, double low, double high)
{
    return std::isnan(value) ? 0 : std::min(high, std::max(low, value));
}

// Which way a body is facing, in the world. Rotation is counter clockwise from pointing up.
inline Point heading(double rotation)
{
    return {-std::sin(rotation), std::cos(rotation)};
}

// Turn a direction in the world in to the same direction as a body sees it, with +y forward and +x
// to its right.
inline Point to_body(double rotation, Point world)
{
    const double c = std::cos(rotation);
    const double s = std::sin(rotation);

    return {world.x * c + world.y * s, -world.x * s + world.y * c};
}

// The angle from straight ahead to a direction in the world, counter clockwise, from -pi to pi.
inline double bearing(double rotation, Point world)
{
    const Point local = to_body(rotation, world);

    return std::atan2(-local.x, local.y);
}

}

#endif // CONTINENTS_GEOMETRY_HPP
