#ifndef FOOD_WEB_FOOD_WEB_HPP
#define FOOD_WEB_FOOD_WEB_HPP

#include "world.hpp"

#include <sim/project.hpp>

#include <deque>
#include <memory>
#include <vector>

namespace food_web {

// The lessons of projects/hail_mary, taken back to one big map with continents on it, and with prey
// whose numbers come from being born and dying instead of being topped up. See the README.
class FoodWeb : public sim::Project {
public:
    static int default_threads();

    FoodWeb(sim::Ui& ui, uint64_t seed, int threads = default_threads());

    double time_step() const override { return DT; }

    void setup_world(plotato::Graph& graph) override;
    void step(double t) override;
    void draw(plotato::Graph& graph) override;

    Settings read_settings() const;

    std::unique_ptr<World> world;

private:
    void sample();

    // How many of each there were on each continent, once a simulated minute, for the population
    // and phase graphs.
    struct Sample {
        double hours = 0;
        std::vector<int> bison;
        std::vector<int> hares;
        std::vector<int> wolves;
    };

    std::deque<Sample> samples;
    double sample_wait = 0;

    Tally last_total{};

    sim::Control shown;

    sim::Control grass_max;
    sim::Control grass_growth;
    sim::Control bison_metabolism;
    sim::Control bison_graze;
    sim::Control bison_calf_cost;
    sim::Control bison_calving_wait;
    sim::Control hare_metabolism;
    sim::Control hare_graze;
    sim::Control hare_kit_cost;
    sim::Control hare_breeding_wait;
    sim::Control breed_fill;
    sim::Control bison_floor;
    sim::Control hare_floor;
    sim::Control min_packs;
    sim::Control swim_speed;
    sim::Control wolf_swim_energy;
    sim::Control bison_swim_energy;
    sim::Control hare_swim_energy;
    sim::Control bison_healing;
    sim::Control bison_weakest;
    sim::Control bite_damage;
    sim::Control horn_damage;
    sim::Control bison_energy;
    sim::Control hare_energy;
    sim::Control living_energy;
    sim::Control running_energy;
    sim::Control call_energy;
    sim::Control prey_call_energy;
    sim::Control max_pack;
    sim::Control daughter_energy;
    sim::Control max_wolves;
    sim::Control mutation_chance;
    sim::Control sight;

    sim::LiveGraph populations_graph;
    sim::LiveGraph phase_graph;

    sim::Field continent_field;
    sim::Field bison_field;
    sim::Field hares_field;
    sim::Field wolves_field;
    sim::Field packs_field;
    sim::Field grass_field;
    sim::Field bison_generation_field;
    sim::Field hare_generation_field;

    sim::HistoryGraph bison_graph;
    sim::HistoryGraph hares_graph;
    sim::HistoryGraph wolves_graph;
    sim::HistoryGraph packs_graph;
    sim::HistoryGraph grass_graph;
    sim::HistoryGraph bison_births_graph;
    sim::HistoryGraph bison_kills_graph;
    sim::HistoryGraph hare_births_graph;
    sim::HistoryGraph hare_kills_graph;
    sim::HistoryGraph starved_graph;
    sim::HistoryGraph kill_wolves_graph;
    sim::HistoryGraph weak_graph;
    sim::HistoryGraph bison_alarm_graph;
    sim::HistoryGraph hare_alarm_graph;
    sim::HistoryGraph wolf_call_graph;
};

}

#endif // FOOD_WEB_FOOD_WEB_HPP
