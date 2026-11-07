#include <sim/sim_thread.hpp>

#include <algorithm>
#include <chrono>
#include <cstddef>

namespace sim {

// Width of a history point at the start of a run, in simulated seconds.
constexpr double HISTORY_START_BIN_WIDTH = 20;

// When the history reaches this many points, neighbours get merged. Has to be even.
constexpr std::size_t HISTORY_MAX_POINTS = 2048;

SimThread::~SimThread()
{
    stop();
}

void SimThread::start(uint64_t seed)
{
    stop_requested = false;

    thread = std::thread(&SimThread::run, this, seed);
}

void SimThread::stop()
{
    stop_requested = true;

    if (thread.joinable()) {
        thread.join();
    }
}

void SimThread::set_fast_forward(bool enabled)
{
    fast_forward = enabled;
}

void SimThread::set_peek(bool held)
{
    peek = held;
}

bool SimThread::take_frame(SimFrame& out)
{
    std::lock_guard<std::mutex> lock(shared_mutex);

    if (!frame_changed) {
        return false;
    }

    out = latest_frame;
    frame_changed = false;

    return true;
}

bool SimThread::take_history(SimHistory& out)
{
    std::lock_guard<std::mutex> lock(shared_mutex);

    if (!history_changed) {
        return false;
    }

    out = history;
    history_changed = false;

    return true;
}

void SimThread::publish_frame(const SimFrame& frame)
{
    std::lock_guard<std::mutex> lock(shared_mutex);

    latest_frame = frame;
    frame_changed = true;
}

// Merge each pair of neighbouring points into one.
static void merge_pairs(std::vector<double>& values)
{
    const std::size_t merged_size = values.size() / 2;

    for (std::size_t i = 0; i < merged_size; i++) {
        values[i] = (values[2 * i] + values[2 * i + 1]) / 2;
    }

    values.resize(merged_size);
}

bool SimThread::publish_history_point(double hours, double avg_dist, double score)
{
    std::lock_guard<std::mutex> lock(shared_mutex);

    history.hours.push_back(hours);
    history.avg_dist.push_back(avg_dist);
    history.avg_score.push_back(score);
    history_changed = true;

    if (history.hours.size() < HISTORY_MAX_POINTS) {
        return false;
    }

    // Halving the resolution keeps the whole run on the graph, with evenly spaced points.
    merge_pairs(history.hours);
    merge_pairs(history.avg_dist);
    merge_pairs(history.avg_score);

    return true;
}

void SimThread::run(uint64_t seed)
{
    using clock = std::chrono::steady_clock;

    const auto frame_time = std::chrono::duration_cast<clock::duration>(std::chrono::duration<double>(1.0 / FPS));

    World world(seed);

    uint64_t steps = 0;

    double displayed_at = -99999;
    double frame_dist_sum = 0;
    uint64_t frame_dist_count = 0;

    double bin_width = HISTORY_START_BIN_WIDTH;
    double bin_start = 0;
    double bin_dist_sum = 0;
    double bin_score_sum = 0;
    uint64_t bin_dist_count = 0;

    clock::time_point next_frame = clock::now();

    while (!stop_requested.load(std::memory_order_relaxed)) {

        world.step();
        steps++;

        frame_dist_sum += world.dist;
        frame_dist_count++;

        bin_dist_sum += world.dist;
        bin_score_sum += world.total_score;
        bin_dist_count++;


        if (world.t - bin_start >= bin_width) {

            const double center = (bin_start + world.t) / 2;

            if (publish_history_point(center / 3600, bin_dist_sum / bin_dist_count, bin_score_sum / bin_dist_count)) {
                bin_width *= 2;
            }

            bin_start = world.t;
            bin_dist_sum = 0;
            bin_dist_count = 0;
            bin_score_sum = 0;
        }

        const bool fast = fast_forward.load(std::memory_order_relaxed) && !peek.load(std::memory_order_relaxed);
        const double display_after = fast ? FAST : -1;

        if (world.t - displayed_at < display_after) {
            continue;
        }

        displayed_at = world.t;

        SimFrame frame;

        frame.t = world.t;
        frame.organism = world.phys_obj.pos;
        frame.target = world.target;
        frame.avg_dist = frame_dist_sum / frame_dist_count;
        frame.steps = steps;

        for (std::size_t i = 0; i < frame.thrust.size(); i++) {
            frame.thrust[i] = std::max(0.0, world.output_vec[i]);
        }

        publish_frame(frame);

        frame_dist_sum = 0;
        frame_dist_count = 0;

        if (fast) {
            continue; // Fast forward never waits for anything.
        }

        // Otherwise hold the frame rate. Going by a deadline instead of sleeping for a fixed time
        // means that the sleep overshooting (which it does a lot on Windows) evens back out.
        next_frame += frame_time;

        const clock::time_point now = clock::now();

        if (next_frame < now) {
            next_frame = now; // We fell behind, or just left fast forward. Don't sprint to catch up.
        }
        else {
            std::this_thread::sleep_until(next_frame);
        }
    }
}

}
