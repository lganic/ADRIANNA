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

    ui.create_buffers();
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
    notify_control_change(); // It might be paused, or partway through a long slow motion frame.

    if (thread.joinable()) {
        thread.join();
    }
}

void SimThread::set_fast_forward(bool enabled)
{
    fast_forward = enabled;
    notify_control_change();
}

void SimThread::set_peek(bool held)
{
    peek = held;
    notify_control_change();
}

void SimThread::set_paused(bool paused_now)
{
    paused = paused_now;

    if (!paused_now) {
        step_requested = false; // Otherwise one left over would go off the next time it pauses.
    }

    notify_control_change();
}

void SimThread::step_once()
{
    if (!paused.load()) {
        return;
    }

    step_requested = true;
    notify_control_change();
}

void SimThread::set_speed(double fraction)
{
    if (!(fraction > 0)) {
        return; // Zero would be a frame that never ends. That's what pausing is for.
    }

    speed = fraction;
    notify_control_change();
}

void SimThread::notify_control_change()
{
    {
        std::lock_guard<std::mutex> lock(control_mutex);

        control_changes++;
    }

    control_changed.notify_all();
}

void SimThread::wait_for_control_change(uint64_t seen)
{
    std::unique_lock<std::mutex> lock(control_mutex);

    control_changed.wait(lock, [&] { return control_changes.load() != seen; });
}

bool SimThread::wait_for_control_change(uint64_t seen, clock::time_point deadline)
{
    std::unique_lock<std::mutex> lock(control_mutex);

    return control_changed.wait_until(lock, deadline, [&] { return control_changes.load() != seen; });
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
    project.draw(*ui.world_buffer);

    // The frame starts out empty, so swapping leaves the buffers blank and ready for next time.
    ui.world_buffer->swap_contents(frame.world);

    frame.live_graphs.resize(ui.live_graphs.size());

    std::size_t index = 0;

    for (LiveGraphState& graph : ui.live_graphs) {
        graph.buffer->swap_contents(frame.live_graphs[index++]);
    }

    for (const FieldState& field : ui.fields) {
        frame.fields.push_back(field.value);
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
    const auto fast_frame_time = std::chrono::duration_cast<clock::duration>(std::chrono::duration<double>(1.0 / FAST_FPS));

    const double time_step = project.time_step();

    double t = 0;
    uint64_t steps = 0;

    double bin_width = HISTORY_START_BIN_WIDTH;
    double bin_start = 0;

    clock::time_point next_frame = clock::now();
    clock::time_point published_at = clock::now() - fast_frame_time; // So the first step always shows.

    bool shown = true; // Whether the latest step has been published. Nothing to show before the first.

    while (!stop_requested.load(std::memory_order_relaxed)) {

        // Read before any of the controls, so that a change from here on cuts short the waits below.
        const uint64_t seen = control_changes.load();

        if (paused.load()) {

            // Fast forward only publishes now and then, so the step that it stopped on might not be up yet.
            if (!shown) {
                publish_frame(t, steps);
                shown = true;
            }

            if (!step_requested.exchange(false)) {
                wait_for_control_change(seen);

                next_frame = clock::now(); // So that if this was unpausing, the first frame back gets its full time.
                continue;
            }

            // Otherwise this is a single step. It goes through the same as any other, history and all.
        }

        t += time_step;

        project.step(t);
        steps++;

        shown = false;

        if (t - bin_start >= bin_width) {

            const double center = (bin_start + t) / 2;

            if (publish_history_point(center / 3600)) {
                bin_width *= 2;
            }

            bin_start = t;
        }

        if (paused.load()) {
            publish_frame(t, steps); // Every single step shows, since that's the whole point of asking for one.
            shown = true;

            continue;
        }

        const bool fast = fast_forward.load(std::memory_order_relaxed) && !peek.load(std::memory_order_relaxed);

        if (fast) {
            // Fast forward never waits for anything. It only stops to publish when a frame is due.
            const clock::time_point now = clock::now();

            if (now - published_at >= fast_frame_time) {
                published_at = now;

                publish_frame(t, steps);
                shown = true;
            }

            continue;
        }

        publish_frame(t, steps);
        shown = true;

        // Otherwise hold the frame rate, slowed down by however much slow motion asks for. Going by
        // a deadline instead of sleeping for a fixed time means that the sleep overshooting (which
        // it does a lot on Windows) evens back out.
        next_frame += std::chrono::duration_cast<clock::duration>(std::chrono::duration<double>(1.0 / (FPS * speed.load())));

        const clock::time_point now = clock::now();

        if (next_frame < now) {
            next_frame = now; // We fell behind, or just left fast forward. Don't sprint to catch up.
        }
        else if (wait_for_control_change(seen, next_frame)) {
            // Something changed partway through the frame, so go again from now. Waiting out the
            // rest of it would hold up a pause, or the jump from a slow speed up to a faster one.
            next_frame = clock::now();
        }
    }
}

}
