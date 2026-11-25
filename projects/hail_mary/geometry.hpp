#ifndef HAIL_MARY_GEOMETRY_HPP
#define HAIL_MARY_GEOMETRY_HPP

#include <algorithm>
#include <cmath>

namespace hail_mary {

constexpr double PI = 3.14159265358979323846;

struct Vec {
    double x = 0;
    double y = 0;
};

inline Vec operator+(Vec a, Vec b) { return {a.x + b.x, a.y + b.y}; }
inline Vec operator-(Vec a, Vec b) { return {a.x - b.x, a.y - b.y}; }
inline Vec operator*(Vec a, double s) { return {a.x * s, a.y * s}; }
inline Vec operator*(double s, Vec a) { return {a.x * s, a.y * s}; }
inline Vec& operator+=(Vec& a, Vec b) { a.x += b.x; a.y += b.y; return a; }
inline Vec& operator-=(Vec& a, Vec b) { a.x -= b.x; a.y -= b.y; return a; }

inline double dot(Vec a, Vec b) { return a.x * b.x + a.y * b.y; }
inline double length(Vec a) { return std::hypot(a.x, a.y); }

// The same direction at length 1, or nothing for nothing.
inline Vec unit(Vec a)
{
    const double l = length(a);

    return l > 1e-12 ? Vec{a.x / l, a.y / l} : Vec{0, 0};
}

// Shrinks a vector to length 1 if it is any longer, keeping its direction.
inline Vec limit_length(Vec a)
{
    const double l = length(a);

    return l > 1 ? Vec{a.x / l, a.y / l} : a;
}

// A NaN comes out as the low end, since std::min and std::max would let it straight through.
inline double clamp(double value, double low, double high)
{
    return std::isnan(value) ? low : std::min(high, std::max(low, value));
}

// Angles are counter clockwise from the +x axis, the usual way.
inline Vec heading_vector(double angle) { return {std::cos(angle), std::sin(angle)}; }

// Into -pi to pi.
inline double wrap_angle(double angle)
{
    angle = std::fmod(angle + PI, 2 * PI);

    return angle < 0 ? angle + PI : angle - PI;
}

// A direction in the world as a body facing this way sees it: x is ahead, y is to its left.
inline Vec to_body(double heading, Vec world)
{
    const double c = std::cos(heading);
    const double s = std::sin(heading);

    return {world.x * c + world.y * s, -world.x * s + world.y * c};
}

// The angle from straight ahead to a direction in the world, positive to the left, from -pi to pi.
inline double bearing(double heading, Vec world)
{
    const Vec local = to_body(heading, world);

    return std::atan2(local.y, local.x);
}

}

#endif // HAIL_MARY_GEOMETRY_HPP
