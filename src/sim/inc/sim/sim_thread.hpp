#ifndef SIM_SIM_THREAD_HPP
#define SIM_SIM_THREAD_HPP

#include <sim/project.hpp>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <mutex>
#include <thread>
#include <vector>

namespace sim {

// While fast forwarding, publish this many frames per second of real time. This used to go by
// simulated time, at one frame per 1000 simulated seconds, which only works while steps are cheap.
// With a whole population to step, that came to one frame every 3 seconds or so, and the GUI froze
// in between, with steps/s reading 0. Going by real time keeps it moving however slow a step is.
constexpr double FAST_FPS = 20;

// The speed of the simulation in real time: one step per frame. Slow motion spaces the frames out.
constexpr double FPS = 30;

// Everything the GUI needs to show one frame of the simulation.
struct SimFrame {
    double t = 0;

    uint64_t steps = 0; // Total steps simulated so far.

    PlotContents world;

    std::vector<PlotContents> live_graphs; // In the same order as Ui::live_graphs.
    std::vector<double> fields; // In the same order as Ui::fields.
};

// All the history graphs so far. Stays a bounded size, no matter how long the simulation has been running.
struct SimHistory {
    std::vector<double> hours;
    std::vector<std::vector<double>> values; // In the same order as Ui::history_graphs.
};

// Runs a Project on its own thread.
//
// The Project belongs to that thread once it starts, and all the GTK calls stay on the GUI thread
// (it isn't safe to call from another thread). Projects still get to draw with Plotato from over
// here, for how that works see canvas.hpp. So the two threads only ever meet here:
//   GUI -> simulation: the atomic control flags, and the Controls in the Ui. Read every step, for free.
//   simulation -> GUI: the latest frame and history, handed over under a mutex.
//
// The simulation never waits on the GUI. If it publishes faster than the GUI can look, the older
// frames just get overwritten, which is what lets fast forward run flat out. It does wait for the
// GUI while paused, and between frames in real time, but changing any of the controls wakes it up.
class SimThread {
public:
    using clock = std::chrono::steady_clock;

    SimThread(Project& project, Ui& ui);
    ~SimThread();

    SimThread(const SimThread&) = delete;
    SimThread& operator=(const SimThread&) = delete;

    void start();
    void stop();

    void set_fast_forward(bool enabled);

    // While held, fast forward is overridden so the project can be watched in real time.
    void set_peek(bool held);

    // Stops stepping until unpaused. Overrides fast forward and peeking. The step that it stopped on
    // always gets published, even out of fast forward, so what's on screen is where it really is.
    void set_paused(bool paused);

    // While paused, run exactly one step and publish it. Does nothing while running. Asks that come
    // in before the simulation gets to them count as one, so that holding a key down on a slow
    // project doesn't leave a backlog of steps to run after letting go.
    void step_once();

    // Slow motion, as a fraction of real time: 1 is one step per frame, 0.25 is a step every 4
    // frames. Only applies while not fast forwarding.
    void set_speed(double fraction);

    // Get the latest data. These return false, and leave `out` alone, if nothing changed since the last call.
    bool take_frame(SimFrame& out);
    bool take_history(SimHistory& out);

private:
    void run();

    void publish_frame(double t, uint64_t steps);

    // Returns true if this made the history compact itself, which halves its resolution.
    bool publish_history_point(double hours);

    // Let the simulation thread know that one of the controls changed, in case it is waiting.
    void notify_control_change();

    // Wait until a control changes from how it was at `seen`.
    void wait_for_control_change(uint64_t seen);

    // The same, but give up at the deadline. Returns false if the deadline came first.
    bool wait_for_control_change(uint64_t seen, clock::time_point deadline);

    Project& project;
    Ui& ui;

    std::thread thread;

    std::atomic<bool> stop_requested{false};
    std::atomic<bool> fast_forward{false};
    std::atomic<bool> peek{false};
    std::atomic<bool> paused{false};
    std::atomic<bool> step_requested{false};
    std::atomic<double> speed{1};

    // Goes up by one every time a control changes. Only ever increased under control_mutex, so
    // that a wait on control_changed can't miss it.
    std::atomic<uint64_t> control_changes{0};

    std::mutex control_mutex;
    std::condition_variable control_changed;

    std::mutex shared_mutex; // Guards everything below.

    SimFrame latest_frame;
    bool frame_changed = false;

    SimHistory history;
    bool history_changed = false;
};

}

#endif // SIM_SIM_THREAD_HPP
