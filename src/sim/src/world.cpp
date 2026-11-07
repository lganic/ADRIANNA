#include <sim/world.hpp>

#include <algorithm>
#include <cmath>
#include <iterator>
#include <vector>

namespace sim {

constexpr double PI = 3.14159265358979323846;

World::World(uint64_t seed)
    : phys_obj({0, 0}, {0, 0}, BOX_SIZE, ORGANISM_RADIUS), org(INPUT_SIZE, OUTPUT_SIZE, seed) {}

void World::step()
{
    // Make the input vector
    // * Top distance raycast
    // * Bottom distance raycast
    // * Left distance raycast
    // * Right distance raycast
    // * Delta X to target
    // * Delta Y to target
    // * X Velocity
    // * Y Velocity

    // Get distance raycasts.
    const RaycastDistances distances = phys_obj.raycast();

    const double x = phys_obj.pos.x;
    const double y = phys_obj.pos.y;

    // Calculate the target position.
    t += DT;

    const double phase = 2 * PI * TARGET_SPEED * std::pow(t / 10000, 2);

    target.x = TARGET_AMPLITUDE * std::cos(phase);
    target.y = TARGET_AMPLITUDE * std::sin(phase);

    // Calculate the delta
    const double dx = target.x - x;
    const double dy = target.y - y;

    // Now make the vector. Everything is scaled to roughly [-1, 1], otherwise the raycasts
    // (which always sum to a constant) swamp the delta and velocity inputs during learning.
    const double input_vec[INPUT_SIZE] = {
        distances.top / (2 * BOX_SIZE),
        distances.bottom / (2 * BOX_SIZE),
        distances.left / (2 * BOX_SIZE),
        distances.right / (2 * BOX_SIZE),
        dx / BOX_SIZE,
        dy / BOX_SIZE,
        phys_obj.vel.x / VEL_SCALE,
        phys_obj.vel.y / VEL_SCALE,
    };

    const cdn::Tensor state({INPUT_SIZE, 1}, std::vector<float>(std::begin(input_vec), std::end(input_vec)));

    const cdn::Tensor action = org.run(static_cast<float>(total_score), state);

    std::copy(action.data().begin(), action.data().end(), output_vec.begin());

    const bool hit_wall = phys_obj.sim(output_vec[0], output_vec[1], output_vec[2], output_vec[3], DT);

    // Get distance to target after moving, so the reward scores this action (for scoring)
    dist = std::hypot(target.x - phys_obj.pos.x, target.y - phys_obj.pos.y);

    total_score = 0;

    if (hit_wall) {
        total_score -= WALL_PUNISH;
    }

    total_score += -dist;

    for (double output : output_vec) {
        total_score -= std::max(0.0, output) * EFFORT_PUNISH;
    }
}

}
