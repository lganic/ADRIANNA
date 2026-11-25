#ifndef FOOD_WEB_WORLD_HPP
#define FOOD_WEB_WORLD_HPP

#include "brain.hpp"
#include "geometry.hpp"
#include "grid.hpp"
#include "pool_threads.hpp"
#include "terrain.hpp"

#include <array>
#include <atomic>
#include <cstdint>
#include <map>
#include <memory>
#include <random>
#include <vector>

namespace food_web {

constexpr double DT = .1;

// The whole map, out to the walls. Four continents, two by two, with sea between them.
constexpr double HALF_WIDTH = 900;
constexpr double HALF_HEIGHT = 700;

// The map is the same on every run, whatever the seed, so that runs can be compared.
constexpr uint64_t MAP_SEED = 2026;

// Bodies. Speeds in meters per second, turning in radians per second, and how quickly each gets up
// to the speed it wants as a time constant, in seconds. All the same as projects/hail_mary: wolves
// are a little faster than hares, and as fast as a healthy grown bison.
constexpr double WOLF_RADIUS = .8;
constexpr double WOLF_MASS = 1;
constexpr double WOLF_SPEED = 9;
constexpr double WOLF_TURN = 4;
constexpr double WOLF_PICKUP = .5;
constexpr double WOLF_HEALTH = 30;

constexpr double BISON_RADIUS = 1.5;
constexpr double BISON_MASS = 5;
constexpr double BISON_SPEED = 9;
constexpr double BISON_TURN = 2;
constexpr double BISON_PICKUP = 1;
constexpr double BISON_HEALTH = 80;

constexpr double HARE_RADIUS = .5;
constexpr double HARE_MASS = .3;
constexpr double HARE_SPEED = 8;
constexpr double HARE_TURN = 7;
constexpr double HARE_PICKUP = .3;
constexpr double HARE_HEALTH = 3;

// Wolves see 270 degrees, prey nearly all the way round, the way eyes on the front or the sides of
// the head do.
constexpr double WOLF_FIELD_OF_VIEW = 2.356;
constexpr double PREY_FIELD_OF_VIEW = 2.88;

constexpr double HEAR_SCALE = 10; // Sound falls off with the square of distance, beyond this.
constexpr double HEAR_RANGE = 120; // Nothing further than this is heard at all. A full call from here is quiet already.
constexpr double QUIETEST = .01; // Loudness heard as 0.

constexpr double BITE_REACH = .6;
constexpr double BITE_ANGLE = 1.05;
constexpr double HORN_REACH = .8;
constexpr double HORN_ANGLE = .8;
constexpr double EAT_REACH = .5;

constexpr double GRASS_CELL = 4; // Meters across each patch of grass.

constexpr int CHANNELS = 3; // That wolves call on. Prey have one call each.

// What a wolf senses, in order. See World::think_wolf().
constexpr int W_SPEED = 0;
constexpr int W_SIDEWAYS = 1;
constexpr int W_TURN = 2;
constexpr int W_HEALTH = 3;
constexpr int W_FOOD = 4;
constexpr int W_PAIN = 5;
constexpr int W_AT_BISON = 6;
constexpr int W_AT_HARE = 7;
constexpr int W_AT_STRANGER = 8;
constexpr int W_EATING = 9;
constexpr int W_WET = 10;
constexpr int W_NOISE = 11;
constexpr int W_CONSTANT = 12;
// Each thing seen takes three: how close, and which way, as how far ahead and how far left.
constexpr int SEEN_CLOSE = 0;
constexpr int SEEN_AHEAD = 1;
constexpr int SEEN_LEFT = 2;
constexpr int W_HARE = 13;
constexpr int W_BISON = 16;
constexpr int W_BISON_FACING = 19;
constexpr int W_BISON_WEAK = 20;
constexpr int W_CARCASS = 21;
constexpr int W_MATE = 24;
constexpr int W_SECOND_MATE = 27;
constexpr int W_STRANGER = 30;
constexpr int W_MATES_SEEN = 33;
constexpr int W_WATER_LEFT = 34; // Whether there is water this far ahead, 45 degrees left, ahead, and right.
constexpr int W_WATER_AHEAD = 35;
constexpr int W_WATER_RIGHT = 36;
constexpr int W_SOUNDS = 37; // Each channel from packmates, then from strangers: loudness, ahead, left.
constexpr int WOLF_INPUTS = W_SOUNDS + 2 * CHANNELS * 3;

constexpr int W_OUT_SPEED = 0;
constexpr int W_OUT_TURN = 1;
constexpr int W_OUT_BITE = 2;
constexpr int W_OUT_CALL = 3;
constexpr int WOLF_OUTPUTS = W_OUT_CALL + CHANNELS;

constexpr Layout WOLF_LAYOUT = {WOLF_INPUTS, 24, WOLF_OUTPUTS};

// What a bison or a hare senses, in order. See World::think_prey().
constexpr int P_SPEED = 0;
constexpr int P_SIDEWAYS = 1;
constexpr int P_TURN = 2;
constexpr int P_HEALTH = 3;
constexpr int P_ENERGY = 4;
constexpr int P_PAIN = 5;
constexpr int P_WET = 6;
constexpr int P_NOISE = 7;
constexpr int P_CONSTANT = 8;
constexpr int P_YOUNG = 9;
constexpr int P_GRASS_HERE = 10;
constexpr int P_GRASS_LEFT = 11;
constexpr int P_GRASS_AHEAD = 12;
constexpr int P_GRASS_RIGHT = 13;
constexpr int P_WATER_LEFT = 14;
constexpr int P_WATER_AHEAD = 15;
constexpr int P_WATER_RIGHT = 16;
constexpr int P_WOLF = 17;
constexpr int P_SECOND_WOLF = 20;
constexpr int P_WOLVES_SEEN = 23;
constexpr int P_KIN = 24; // The nearest of its own kind.
constexpr int P_KIN_SEEN = 27;
constexpr int P_KIN_CALLS = 28; // Its own kind's calls, and the noise of one of them being bitten.
constexpr int P_WOLF_SOUNDS = 31; // Wolves calling, biting, and feeding.
constexpr int PREY_INPUTS = 34;

constexpr int P_OUT_SPEED = 0;
constexpr int P_OUT_TURN = 1;
constexpr int P_OUT_CALL = 2;
constexpr int PREY_OUTPUTS = 3;

constexpr Layout PREY_LAYOUT = {PREY_INPUTS, 16, PREY_OUTPUTS};

// The settings, as they were at the start of a step. See food_web.cpp for what each slider does.
struct Settings {
    double grass_max = .05; // Energy per square meter.
    double grass_growth = .006; // Per second, at its fastest.
    double grass_seed = .05; // Share of the most grass that bare ground grows back from.

    double bison_metabolism = .04;
    double bison_running = .12;
    double bison_graze = .3;
    double bison_capacity = 80;
    double bison_calf_cost = 35;
    double bison_calving_wait = 1800;
    double hare_metabolism = .02;
    double hare_running = .04;
    double hare_graze = .06;
    double hare_capacity = 15;
    double hare_kit_cost = 6;
    double hare_breeding_wait = 480;
    double breed_fill = .85; // How full a prey animal has to be to have young.
    double prey_call_energy = .02;

    double bison_floor = 4; // On one continent, below which a grown one wanders in now and then.
    double hare_floor = 6;
    double min_packs = 1; // On one continent.

    double swim_speed = .35; // Of top speed.
    double wolf_swim_energy = .3; // Per second, on top.
    double bison_swim_energy = .3;
    double hare_swim_energy = .05;

    double bison_healing = 8;
    double bison_weakest = .5; // How strong a newborn calf, or a bison at the very end of its life, is next to one in its prime.
    double bite_damage = 6;
    double horn_damage = 8;
    double bison_energy = 360; // Food in a hunted bison in its prime.
    double hare_energy = 25;
    double dead_bison_energy = 60; // In a bison that died of old age or hunger.
    double carcass_lifetime = 120;
    double eat_rate = 10;

    double living_energy = .1;
    double running_energy = .25;
    double call_energy = .05;
    double bite_energy = .1;
    double pup_cost = 40;
    double pup_reserve = 60;
    double max_pack = 8;
    double daughter_energy = 400;
    double max_wolves = 400;
    double max_prey = 6000;
    double mutation_chance = 1;

    double sight = 30;
    double noise_volume = .3;
};

// Counts since the start, for one continent: who was born, who died of what, and who ate what.
struct Tally {
    uint64_t bison_born = 0;
    uint64_t hares_born = 0;
    uint64_t bison_killed = 0;
    uint64_t hares_killed = 0;
    uint64_t weak_killed = 0; // Bison that were calves, or old.
    uint64_t calves_killed = 0; // Not yet grown.
    uint64_t old_killed = 0; // Fading at the end of their lives.
    uint64_t hungry_killed = 0; // Grown, not old, but starving or nearly.
    uint64_t bison_starved = 0;
    uint64_t hares_starved = 0;
    uint64_t bison_old = 0;
    uint64_t hares_old = 0;
    uint64_t bison_arrived = 0; // Wandered in, because there were almost none.
    uint64_t hares_arrived = 0;

    uint64_t pups = 0;
    uint64_t daughters = 0;
    uint64_t refounded = 0;
    uint64_t wolves_starved = 0;
    uint64_t wolves_gored = 0;
    uint64_t wolves_killed = 0;
    uint64_t wolves_old = 0;

    double kill_wolves = 0; // Wolves that bit each bison in its last 10 seconds, added up.
    double surround = 0; // See projects/hail_mary.
    uint64_t surround_kills = 0;

    double wolf_food_bison = 0;
    double wolf_food_hare = 0;
    double wolf_food_scavenged = 0;
    double bison_grazed = 0;
    double hares_grazed = 0;

    // Added up over every step, for averages.
    double wolf_steps = 0;
    double bison_steps = 0;
    double hare_steps = 0;
    double wolf_wet_steps = 0; // Of those, how many were in the water.
    double bison_wet_steps = 0;
    double hare_wet_steps = 0;
    double grass = 0; // Energy standing.

    // Whether calls mean anything: how loud, while a wolf is in sight and while none is, for prey,
    // and while a bison is in sight and while nothing is, for wolves.
    std::array<double, 2> bison_calls{};
    std::array<double, 2> bison_call_steps{};
    std::array<double, 2> hare_calls{};
    std::array<double, 2> hare_call_steps{};
    std::array<double, 2> wolf_calls{};
    std::array<double, 2> wolf_call_steps{};

    Tally& operator+=(const Tally& other);
};

Tally operator-(Tally a, const Tally& b);

using Colour = std::array<double, 3>;

struct Body {
    Vec pos;
    Vec vel;
    double heading = 0;

    double radius = 1;
    double mass = 1;
    double top_speed = 1;
    double top_turn = 1;
    double pickup = 1;

    double want_speed = 0;
    double want_turn = 0;

    bool wet = false; // In the water as of the end of the last step.
};

// The same as projects/hail_mary: a genome that every wolf in the pack shares, and the food store
// they all live off.
struct Pack {
    int id = 0;
    std::shared_ptr<const Genome> genome;
    Colour colour{};

    double energy = 0;
    double gathered = 0;
    double eaten = 0;
    double intake = 0;
    double founded = 0;
    double pup_wait = 0;
    int members = 0;
    int daughters = 0;
};

struct Wolf {
    int id = 0;
    int pack = 0;
    Body body;
    std::unique_ptr<Brain> brain;
    double health = WOLF_HEALTH;
    double age = 0;
    double lifespan = 0;
    uint64_t noise = 1; // Its own random numbers, for its noise input.
    int land = -1; // Which continent it is on, or -1 at sea.

    bool biting = false;
    std::array<float, CHANNELS> calls{};
    double turn = 0;

    double pain = 0;
    bool at_bison = false;
    bool at_hare = false;
    bool at_stranger = false;
    bool eating = false;
    bool noisy = false;
    bool starving = false;
    int hurt_by = 0; // 1 for another pack's wolf, 2 for a bison.

    bool saw_bison = false; // On its last look round, for the call stats.
};

struct Bite {
    int wolf = 0;
    int pack = 0;
    double time = 0;
    Vec from;
};

// A bison or a hare. Unlike projects/hail_mary, prey have brains that evolve too, and live off
// grass, and have young of their own when they are well fed, each with its own mutated genome.
struct Prey {
    int id = 0;
    bool bison = false;
    Body body;
    std::unique_ptr<Brain> brain;
    double health = 1;
    double energy = 0;
    double age = 0;
    double lifespan = 0;
    double breed_wait = 0;
    uint64_t noise = 1;
    int land = -1;

    double call = 0;
    double turn = 0;
    double pain = 0;
    bool noisy = false; // Bitten on the last step, which others of its kind hear.
    bool starving = false;
    int bitten_by = -1;
    std::vector<Bite> bites;

    bool saw_wolf = false; // On its last look round, for the call stats.
};

enum Source { HUNTED_BISON, HUNTED_HARE, SCAVENGED };

struct Carcass {
    Vec pos;
    double energy = 0;
    double radius = 1;
    double age = 0;
    Source source = SCAVENGED;
};

struct Caller {
    Vec pos;
    int id = 0;
    int pack = 0; // For wolves. For prey, 1 for bison and 0 for hares.
    std::array<float, CHANNELS> volume{}; // Prey only use the first.
    float noise = 0;
};

struct Archived {
    std::shared_ptr<const Genome> genome;
    Colour colour{};
    double gathered = 0;
};

// Grass, bison, hares, and wolves, on four continents in a walled sea. Grass grows back on its own,
// bison and hares eat it, and wolves eat them, and how many of each there are comes from who gets
// born and who dies, so they rise and fall against each other.
//
// One world, so everything happens in one place, but nearly all of the work is each animal looking
// round and thinking, which only reads the world. That part is spread over a set of threads, and
// everything that changes the world happens after, on one thread, in order of id. So a run comes
// out exactly the same however many threads there are.
class World {
public:
    World(uint64_t seed, const Settings& settings, int threads);

    void step(const Settings& settings);

    const Terrain terrain;

    double time = 0;

    std::vector<Wolf> wolves; // In order of id.
    std::map<int, Pack> packs;
    std::vector<Prey> prey; // In order of id.
    std::vector<Carcass> carcasses;

    // Energy of grass in each patch, by rows from the bottom left. Water patches stay at 0.
    std::vector<float> grass;
    int grass_columns = 0;
    int grass_rows = 0;

    // One for each continent, then one more for anything that happens at sea.
    std::vector<Tally> tallies;

    int continents() const { return terrain.landmass_count(); }

    // Grass here, as a share of the most there can be, or 0 for water.
    double grass_at(Vec p, const Settings& settings) const;

    // How much grass each continent has standing right now.
    std::vector<double> grass_by_continent() const;

    int highest_generation(bool bison) const;

    // Seconds spent in each part of step(), when profiling is on: looking round, then everything
    // else in the order step() does it. For the harness.
    bool profiling = false;
    std::array<double, 12> profile{};

private:
    Tally& tally(int land) { return tallies[static_cast<std::size_t>(land >= 0 ? land : continents())]; }

    void index_everything();
    void think_wolf(Wolf& wolf, const Settings& settings);
    void think_prey(Prey& animal, const Settings& settings);
    void move(const Settings& settings);
    void fight(const Settings& settings);
    void eat(const Settings& settings);
    void upkeep(const Settings& settings);
    void kill(Prey& animal, const Settings& settings);
    void lifecycle(const Settings& settings);
    void breed_prey(const Settings& settings);
    void breed_wolves(const Settings& settings);
    void rescue(const Settings& settings);
    void grow_grass(const Settings& settings);
    void listen_list(const Settings& settings);
    void record(const Settings& settings);

    Pack& found_pack(std::shared_ptr<const Genome> genome, const Colour& colour, int count, double energy, Vec around, double spread);
    void add_wolf(Pack& pack, Vec pos, double heading);
    Prey& add_prey(bool bison, Vec pos, double heading, std::shared_ptr<const Genome> genome, double energy);

    std::shared_ptr<const Genome> wolf_ancestor();
    std::shared_ptr<const Genome> prey_ancestor(bool bison);

    Vec spot_on(int land);
    Vec inside(Vec pos, double radius) const;
    Colour drift(const Colour& colour, double amount);
    const Wolf* find_wolf(int id) const;
    int grass_cell(Vec p) const; // -1 if outside the walls.

    std::mt19937_64 rng;

    int next_wolf = 0;
    int next_pack = 0;
    int next_prey = 0;

    double rescue_wait = 0;

    double weakest = .4; // Settings::bison_weakest, as of this step.
    double refound_wait = 0;

    std::vector<int> grass_land; // Which continent each patch of grass is on, or -1 for water.

    std::vector<Caller> wolf_callers; // From the last step, which is what everyone hears on this one.
    std::vector<Caller> prey_callers;

    std::vector<Archived> wolf_archive; // Packs that died most recently.
    std::vector<std::shared_ptr<const Genome>> bison_archive; // Grown prey that died most recently.
    std::vector<std::shared_ptr<const Genome>> hare_archive;

    Grid wolf_grid;
    Grid prey_grid;
    Grid carcass_grid;
    Grid body_grid;
    Grid wolf_caller_grid;
    Grid prey_caller_grid;

    int grass_wait = 0;

    std::vector<Body*> bodies;

    PoolThreads threads;
    std::atomic<int> next_job{0};
};

}

#endif // FOOD_WEB_WORLD_HPP
