#include <sim/sim_thread.hpp>

#include <chrono>
#include <cstddef>
#include <utility>

namespace sim {

// Width of a history point at the start of a run, in simulated seconds.
constexpr double HISTORY_START_BIN_WIDTH = 20;

// When the history reaches this many points, neighbours get merged. Has to be even.
constexpr std::size_t HISTORY_MAX_POINTS = 2048;

SimThread::SimThread(Project& project, Ui& ui) : project(project), ui(ui)
{
    history.values.resize(ui.history_graphs.size());
}

SimThread::~SimThread()
{
    stop();
}

void SimThread::start()
{
    stop_requested = false;

    thread = std::thread(&SimThread::run, this);
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

    out = std::move(latest_frame);
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

void SimThread::publish_frame(double t, uint64_t steps)
{
    SimFrame frame;

    frame.t = t;
    frame.steps = steps;

    // This goes first, since drawing is where a project updates anything that is slow to work out.
    project.draw(frame.scene);

    for (const FieldState& field : ui.fields) {
        frame.fields.push_back(field.value);
    }

    for (const CurveGraphState& curve : ui.curve_graphs) {
        frame.curves.push_back({curve.x, curve.y});
    }

    std::lock_guard<std::mutex> lock(shared_mutex);

    latest_frame = std::move(frame);
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

bool SimThread::publish_history_point(double hours)
{
    std::lock_guard<std::mutex> lock(shared_mutex);

    history.hours.push_back(hours);

    std::size_t index = 0;

    for (HistoryGraphState& graph : ui.history_graphs) {

        // A project doesn't have to add something every step, so a point can end up with nothing in it.
        if (graph.count > 0) {
            graph.last = graph.sum / graph.count;
        }

        history.values[index++].push_back(graph.last);

        graph.sum = 0;
        graph.count = 0;
    }

    history_changed = true;

    if (history.hours.size() < HISTORY_MAX_POINTS) {
        return false;
    }

    // Halving the resolution keeps the whole run on the graph, with evenly spaced points.
    merge_pairs(history.hours);

    for (std::vector<double>& values : history.values) {
        merge_pairs(values);
    }

    return true;
}

void SimThread::run()
{
    using clock = std::chrono::steady_clock;

    const auto frame_time = std::chrono::duration_cast<clock::duration>(std::chrono::duration<double>(1.0 / FPS));

    const double time_step = project.time_step();

    double t = 0;
    uint64_t steps = 0;

    double displayed_at = -99999;

    double bin_width = HISTORY_START_BIN_WIDTH;
    double bin_start = 0;

    clock::time_point next_frame = clock::now();

    while (!stop_requested.load(std::memory_order_relaxed)) {

        t += time_step;

        project.step(t);
        steps++;

        if (t - bin_start >= bin_width) {

            const double center = (bin_start + t) / 2;

            if (publish_history_point(center / 3600)) {
                bin_width *= 2;
            }

            bin_start = t;
        }

        const bool fast = fast_forward.load(std::memory_order_relaxed) && !peek.load(std::memory_order_relaxed);
        const double display_after = fast ? FAST : -1;

        if (t - displayed_at < display_after) {
            continue;
        }

        displayed_at = t;

        publish_frame(t, steps);

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
