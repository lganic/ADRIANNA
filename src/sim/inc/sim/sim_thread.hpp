#ifndef SIM_SIM_THREAD_HPP
#define SIM_SIM_THREAD_HPP

#include <sim/project.hpp>

#include <atomic>
#include <cstdint>
#include <mutex>
#include <thread>
#include <vector>

namespace sim {

// While fast forwarding, publish one frame per this many simulated seconds.
constexpr double FAST = 1000;

// The speed of the simulation when it is not fast forwarding: one step per frame.
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
// frames just get overwritten, which is what lets fast forward run flat out.
class SimThread {
public:
    SimThread(Project& project, Ui& ui);
    ~SimThread();

    SimThread(const SimThread&) = delete;
    SimThread& operator=(const SimThread&) = delete;

    void start();
    void stop();

    void set_fast_forward(bool enabled);

    // While held, fast forward is overridden so the project can be watched in real time.
    void set_peek(bool held);

    // Get the latest data. These return false, and leave `out` alone, if nothing changed since the last call.
    bool take_frame(SimFrame& out);
    bool take_history(SimHistory& out);

private:
    void run();

    void publish_frame(double t, uint64_t steps);

    // Returns true if this made the history compact itself, which halves its resolution.
    bool publish_history_point(double hours);

    Project& project;
    Ui& ui;

    std::thread thread;

    std::atomic<bool> stop_requested{false};
    std::atomic<bool> fast_forward{false};
    std::atomic<bool> peek{false};

    std::mutex shared_mutex; // Guards everything below.

    SimFrame latest_frame;
    bool frame_changed = false;

    SimHistory history;
    bool history_changed = false;
};

}

#endif // SIM_SIM_THREAD_HPP
