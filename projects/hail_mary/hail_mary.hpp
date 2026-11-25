#ifndef HAIL_MARY_HAIL_MARY_HPP
#define HAIL_MARY_HAIL_MARY_HPP

#include "continent.hpp"
#include "pool_threads.hpp"

#include <sim/project.hpp>

#include <atomic>
#include <cstdint>
#include <memory>
#include <vector>

namespace hail_mary {

constexpr int DEFAULT_CONTINENTS = 8;

// Wolves, bison, and hares, on a handful of continents that only ever meet through the odd pack
// that migrates. See the README for what this is for and why it is built the way it is.
//
// Wolves live in packs, which share a genome and a food store, and evolve. Bison and hares run on
// fixed rules. A bison heals faster than one wolf can bite it, and gores anything in front of it,
// so it takes two or more wolves at once, working round it, to bring one down, and it is worth
// twelve hares. Wolves see 30 m, but hear calls from all over the continent, so a wolf that finds
// a herd has to call the rest of its pack in if it is to eat it.
//
// Each continent steps on its own, so they are spread over a set of threads, and a run comes out
// exactly the same however many threads there are.
class HailMary : public sim::Project {
public:
    static int default_threads();

    HailMary(sim::Ui& ui, uint64_t seed, int continent_count = DEFAULT_CONTINENTS, int thread_count = default_threads());

    double time_step() const override { return DT; }

    void setup_world(plotato::Graph& graph) override;
    void step(double t) override;
    void draw(plotato::Graph& graph) override;

    Settings read_settings() const;

    // Whether calls can't be heard on this continent, with the mute slider where it is now.
    bool muted(int continent) const;

    std::vector<std::unique_ptr<Continent>> continents;

    double time = 0;
    uint64_t migrations = 0;

private:
    void migrate(const Settings& settings);
    void record(const Settings& settings);

    PoolThreads threads;
    std::atomic<int> next_continent{0};

    double migration_wait = 0;

    // Each continent's counts as of the last step, to graph how much they changed by.
    std::vector<Stats> last_stats;

    sim::Control shown;
    sim::Control mute;
    sim::Control migration;

    sim::Control bison_wanted;
    sim::Control hares_wanted;
    sim::Control bison_calving;
    sim::Control hare_breeding;
    sim::Control bison_healing;
    sim::Control bite_damage;
    sim::Control horn_damage;
    sim::Control bison_fight;
    sim::Control herd_defence;
    sim::Control bison_energy;
    sim::Control hare_energy;
    sim::Control old_bison_energy;
    sim::Control carcass_lifetime;
    sim::Control eat_rate;

    sim::Control living_energy;
    sim::Control running_energy;
    sim::Control call_energy;
    sim::Control bite_energy;
    sim::Control pup_cost;
    sim::Control pup_reserve;
    sim::Control max_pack;
    sim::Control daughter_energy;
    sim::Control max_wolves;
    sim::Control mutation_chance;

    sim::Control sight;
    sim::Control noise_volume;
    sim::Control prey_hearing;
    sim::Control founders;
    sim::Control scripted;

    sim::Field continent_field;
    sim::Field muted_field;
    sim::Field wolves_field;
    sim::Field packs_field;
    sim::Field bison_field;
    sim::Field hares_field;
    sim::Field generation_field;
    sim::Field bison_kills_field;
    sim::Field hare_kills_field;
    sim::Field migrations_field;

    sim::HistoryGraph wolves_graph;
    sim::HistoryGraph packs_graph;
    sim::HistoryGraph bison_graph;
    sim::HistoryGraph hares_graph;
    sim::HistoryGraph bison_kills_heard_graph;
    sim::HistoryGraph bison_kills_muted_graph;
    sim::HistoryGraph hare_kills_graph;
    sim::HistoryGraph kill_wolves_graph;
    sim::HistoryGraph surround_graph;
    sim::HistoryGraph front_graph;
    sim::HistoryGraph bison_share_graph;
    sim::HistoryGraph hare_share_graph;
    sim::HistoryGraph scavenged_graph;
    sim::HistoryGraph calling_graph;
    sim::HistoryGraph signal_graph;
    sim::HistoryGraph approach_heard_graph;
    sim::HistoryGraph approach_muted_graph;
    sim::HistoryGraph wolves_killed_graph;
    sim::HistoryGraph gored_graph;
    sim::HistoryGraph daughters_graph;
    sim::HistoryGraph generation_graph;
    sim::HistoryGraph mutation_graph;
};

// Every continent's stats added together. Only the ones that are muted, or only the ones that
// aren't, going by heard.
Stats total(const HailMary& world, bool heard);

}

#endif // HAIL_MARY_HAIL_MARY_HPP
