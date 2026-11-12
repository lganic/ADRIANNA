#ifndef ISLAND_EVOLUTION_POOL_THREADS_HPP
#define ISLAND_EVOLUTION_POOL_THREADS_HPP

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <exception>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

namespace island_evolution {

// A fixed set of threads, one for each pool, which all get handed the same job at once. The thread
// that hands it out does pool 0 itself while it waits, so there is one thread fewer than pools.
//
// While fast forwarding, jobs come one after another with only the physics in between, which takes
// less time than waking a sleeping thread does on Windows. So a thread that runs out of work spins
// for a little while, watching for the next job, and only goes to sleep if none turns up.
class PoolThreads {
public:
    explicit PoolThreads(int pools);
    ~PoolThreads();

    PoolThreads(const PoolThreads&) = delete;
    PoolThreads& operator=(const PoolThreads&) = delete;

    int pools() const { return static_cast<int>(threads.size()) + 1; }

    // Call job(pool) for every pool at once, each on its own thread, and return once they are all
    // done. If any of them throws, the exception from the lowest pool comes back out of here.
    void run(const std::function<void(int)>& job);

private:
    void work(int pool);

    std::vector<std::thread> threads;
    std::vector<std::exception_ptr> errors; // One per pool.

    const std::function<void(int)>* job = nullptr;

    std::atomic<uint64_t> round{0}; // Goes up by one for every job handed out, which is what the threads watch for.
    std::atomic<int> busy{0}; // Threads still working on this round, not counting the one that handed it out.
    std::atomic<bool> stopping{false};

    std::mutex mutex;
    std::condition_variable wake; // Where the threads sleep, waiting on the next round.
    std::condition_variable finished; // Where run() sleeps, waiting on the threads.

    // Both of these only change with the mutex held, so that a wake up can't go missing.
    int sleeping = 0;
    bool caller_sleeping = false;
};

}

#endif // ISLAND_EVOLUTION_POOL_THREADS_HPP
