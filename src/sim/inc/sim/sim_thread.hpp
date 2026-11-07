#ifndef SIM_SIM_THREAD_HPP
#define SIM_SIM_THREAD_HPP

#include <sim/world.hpp>

#include <array>
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

// Everything the GUI needs to draw one frame of the simulation.
struct SimFrame {
    double t = 0;

    Point organism;
    Point target;

    std::array<double, OUTPUT_SIZE> thrust{}; // Up, down, left, right. Clamped, so what the thrusters really did.

    double avg_dist = 0; // Average distance to the target since the previous frame.

    uint64_t steps = 0; // Total steps simulated so far.
};

// The learning curve so far. Stays a bounded size, no matter how long the simulation has been running.
struct SimHistory {
    std::vector<double> hours;
    std::vector<double> avg_dist;
    std::vector<double> avg_score;
};

// Runs a World on its own thread.
//
// The World lives entirely on that thread, and all the GTK / Plotato calls stay on the GUI thread
// (neither is safe to call from another thread). So the two only ever meet here:
//   GUI -> simulation: the atomic control flags. Read every step, for free.
//   simulation -> GUI: the latest frame and history, copied in and out under a mutex.
//
// The simulation never waits on the GUI. If it publishes faster than the GUI can look, the older
// frames just get overwritten, which is what lets fast forward run flat out.
class SimThread {
public:
    SimThread() = default;
    ~SimThread();

    SimThread(const SimThread&) = delete;
    SimThread& operator=(const SimThread&) = delete;

    void start(uint64_t seed);
    void stop();

    void set_fast_forward(bool enabled);

    // While held, fast forward is overridden so the organism can be watched in real time.
    void set_peek(bool held);

    // Copy out the latest data. These return false, and leave `out` alone, if nothing changed since the last call.
    bool take_frame(SimFrame& out);
    bool take_history(SimHistory& out);

private:
    void run(uint64_t seed);

    void publish_frame(const SimFrame& frame);

    // Returns true if this made the history compact itself, which halves its resolution.
    bool publish_history_point(double hours, double avg_dist, double score);

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
