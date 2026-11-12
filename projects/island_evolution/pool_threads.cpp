#include "pool_threads.hpp"

#include <algorithm>
#include <chrono>

#if defined(__x86_64__) || defined(__i386__) || defined(_M_X64) || defined(_M_IX86)
#include <immintrin.h>
#define ISLAND_EVOLUTION_PAUSE() _mm_pause()
#else
#define ISLAND_EVOLUTION_PAUSE() std::this_thread::yield()
#endif

namespace island_evolution {

// How long a thread with nothing to do keeps watching for more, before it goes to sleep. This has to
// cover the physics and the rest of what the main thread does between jobs, which is around 100 us
// with a full population. When not fast forwarding, it is what gets wasted on every step.
constexpr auto SPIN_TIME = std::chrono::microseconds(500);

// Checking the clock is slow next to checking an atomic, so it only happens every so often.
constexpr int SPINS_PER_CLOCK_CHECK = 64;

// Spin until ready() comes true, or until SPIN_TIME is up. Returns whether it came true.
template <typename Ready>
static bool spin(Ready ready)
{
    const auto give_up = std::chrono::steady_clock::now() + SPIN_TIME;

    for (;;) {
        for (int spin = 0; spin < SPINS_PER_CLOCK_CHECK; spin++) {
            if (ready()) {
                return true;
            }

            // Lets the other thread on the same core have it, and stops the CPU from guessing wrong about the loop.
            ISLAND_EVOLUTION_PAUSE();
        }

        if (std::chrono::steady_clock::now() >= give_up) {
            return ready();
        }
    }
}

PoolThreads::PoolThreads(int pools) : errors(static_cast<std::size_t>(std::max(1, pools)))
{
    for (int pool = 1; pool < pools; pool++) {
        threads.emplace_back(&PoolThreads::work, this, pool);
    }
}

PoolThreads::~PoolThreads()
{
    {
        std::lock_guard<std::mutex> lock(mutex);

        stopping.store(true);
        round.fetch_add(1);
    }

    wake.notify_all();

    for (std::thread& thread : threads) {
        thread.join();
    }
}

void PoolThreads::run(const std::function<void(int)>& task)
{
    job = &task;
    busy.store(static_cast<int>(threads.size()), std::memory_order_relaxed);

    // Everything above gets to the threads along with the new round.
    bool anyone_asleep;

    {
        std::lock_guard<std::mutex> lock(mutex);

        round.fetch_add(1, std::memory_order_release);
        anyone_asleep = sleeping > 0;
    }

    if (anyone_asleep) {
        wake.notify_all();
    }

    try {
        task(0);
    } catch (...) {
        errors[0] = std::current_exception();
    }

    const auto done = [this] { return busy.load(std::memory_order_acquire) == 0; };

    if (!spin(done)) {
        std::unique_lock<std::mutex> lock(mutex);

        caller_sleeping = true;
        finished.wait(lock, done);
        caller_sleeping = false;
    }

    job = nullptr;

    for (std::exception_ptr& error : errors) {
        if (error) {
            const std::exception_ptr first = error;

            std::fill(errors.begin(), errors.end(), nullptr);
            std::rethrow_exception(first);
        }
    }
}

void PoolThreads::work(int pool)
{
    uint64_t seen = 0;

    for (;;) {
        const auto ready = [&] { return round.load(std::memory_order_acquire) != seen; };

        if (!spin(ready)) {
            std::unique_lock<std::mutex> lock(mutex);

            sleeping++;
            wake.wait(lock, ready);
            sleeping--;
        }

        // The next round can't start until this one is done, so this is the round that woke it up.
        seen = round.load(std::memory_order_acquire);

        if (stopping.load()) {
            return;
        }

        try {
            (*job)(pool);
        } catch (...) {
            errors[pool] = std::current_exception();
        }

        // The last one done has to wake up run(), if it got tired of waiting.
        if (busy.fetch_sub(1, std::memory_order_acq_rel) == 1) {
            std::lock_guard<std::mutex> lock(mutex);

            if (caller_sleeping) {
                finished.notify_one();
            }
        }
    }
}

}
