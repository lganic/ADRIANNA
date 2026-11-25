#ifndef HAIL_MARY_CONTINENT_HPP
#define HAIL_MARY_CONTINENT_HPP

#include "brain.hpp"
#include "geometry.hpp"
#include "grid.hpp"

#include <array>
#include <cmath>
#include <cstdint>
#include <initializer_list>
#include <map>
#include <memory>
#include <random>
#include <vector>

namespace hail_mary {

constexpr double DT = .1;

// Each continent is a square of open plain this far from its middle to each edge. It wraps round:
// going off one edge comes back on at the other, so there are no walls or corners to trap prey
// against. On an open plain, the only way to stop a bison that can run as fast as a wolf is to
// get in its way, or to wear it down, which takes more than one wolf.
constexpr double HALF_SIZE = 200;

// Back on to the map, the other side, if it went off an edge.
inline Vec wrapped(Vec p)
{
    const auto round = [](double x) { return x - 2 * HALF_SIZE * std::floor((x + HALF_SIZE) / (2 * HALF_SIZE)); };

    return {round(p.x), round(p.y)};
}

// The shortest way from one spot to another, which can be across an edge.
inline Vec offset(Vec from, Vec to)
{
    return wrapped(to - from);
}

// The bodies. Wolves are a little faster than hares, which turn much faster, and zigzag, and as fast
// as a healthy grown bison, which is heavier, and slower to turn and to get going. Speeds are in
// meters per second, turning in radians per second, and how quickly each gets up to the speed it
// wants is the time constant, in seconds.
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

// What wolves see, and how they hear. See Continent::sense().
constexpr double FIELD_OF_VIEW = 2.356; // Out to each side, so 270 degrees in all. Nothing behind.
constexpr double HEAR_SCALE = 10; // Sound falls off with the square of distance, beyond this.
constexpr double QUIETEST = .01; // Loudness heard as 0. Full volume is heard as 0 past about 100 m.

// Bites reach this far past the front of the biter, at anything within this angle of straight ahead.
constexpr double BITE_REACH = .6;
constexpr double BITE_ANGLE = 1.05;

// The horns of a bison reach this far, at anything within this angle of its nose.
constexpr double HORN_REACH = .8;
constexpr double HORN_ANGLE = .8;

// How close to a carcass a wolf has to be to eat from it, past touching it.
constexpr double EAT_REACH = .5;

constexpr int MAX_PACKS = 16; // On one continent.
constexpr int MIN_PACKS = 6; // Fewer than this, and a new one is founded from the best genomes so far.

// The settings, as they were at the start of a step. See hail_mary.cpp for what each slider does.
struct Settings {
    double bison_wanted = 30;
    double hares_wanted = 30;
    double bison_calving = 30;
    double hare_breeding = 20;
    double bison_healing = 8;
    double bite_damage = 6;
    double horn_damage = 8;
    double bison_fight = .3;
    double herd_defence = .5;
    double bison_energy = 240;
    double hare_energy = 12;
    double old_bison_energy = 60;
    double carcass_lifetime = 120;
    double eat_rate = 10;

    double living_energy = .12;
    double running_energy = .25;
    double call_energy = .05;
    double bite_energy = .1;
    double pup_cost = 40;
    double pup_reserve = 60;
    double max_pack = 6;
    double daughter_energy = 400;
    double max_wolves = 60;
    double mutation_chance = 1;

    double sight = 30;
    double noise_volume = .3; // Of biting and feeding, on channel 0.
    double prey_hearing = 20;

    // Wolves follow a fixed rule instead of their brains, for testing what the bodies and prey
    // allow: 0 for off, then run at the nearest hare (1), bison (2), or either (3), and bite it.
    int scripted = 0;

    // What brains new packs start with, when there are no dead packs yet to found them from:
    // 0 for random, 1 for plain predators (see Genome::ancestor()).
    int founders = 1;

    bool muted = false; // Calls on this continent are made, and paid for, but nobody hears them.
};

// Where the energy that wolves eat came from.
enum Source { HUNTED_BISON, HUNTED_HARE, OLD_AGE, DEAD_WOLF, SOURCES };

// What a wolf was doing when it called, from the most pressing down. See Continent::sense().
enum Context { AT_BISON, EATING, SEES_BISON, SEES_HARE, SEES_STRANGER, ALONE, WITH_PACK, CONTEXTS };

// Counts since the start, for the graphs and the harness, which looks at how they change.
struct Stats {
    uint64_t bison_kills = 0;
    uint64_t weak_kills = 0; // Of bison that were calves, or old.
    uint64_t hare_kills = 0;
    uint64_t old_bison = 0; // Died of old age, and left a carcass behind.
    uint64_t wolves_killed = 0; // By wolves of another pack.
    uint64_t wolves_gored = 0;
    uint64_t wolves_starved = 0;
    uint64_t wolves_old = 0;
    uint64_t pups = 0;
    uint64_t daughters = 0;
    uint64_t refounded = 0; // Packs founded from the best genomes so far, when there were too few.
    uint64_t immigrants = 0;

    std::array<double, SOURCES> eaten{};

    // For every bison killed: how many wolves had bitten it in the last 10 seconds, and from how
    // many packs. And for every one killed by two or more, how far round it they were (see kill()).
    double kill_wolves = 0;
    double kill_packs = 0;
    double surround = 0;
    uint64_t surround_kills = 0;

    uint64_t bites = 0; // On bison.
    uint64_t front_bites = 0; // On bison, from in front, where the horns are.

    // How loud each channel was, added up over every step of every wolf, by what it was doing.
    std::array<std::array<double, CHANNELS>, CONTEXTS> calls{};
    std::array<uint64_t, CONTEXTS> context_steps{};

    // Whether wolves go towards packmates' calls. Over every step that a wolf hears a packmate
    // calling loudly, out of sight of any packmate, prey, or carcass: its speed towards the call,
    // as a fraction of top speed. Worked out the same with calls muted, from what would have been
    // heard, so the two can be compared.
    double approach = 0;
    uint64_t approach_steps = 0;

    // How far each wolf is from its nearest packmate, added up over every step of every wolf that
    // has one. Packs that spread out to search have this bigger.
    double packmate_distance = 0;
    uint64_t packmate_steps = 0;

    // The same as approach, for each channel on its own, and for the noise of packmates biting and
    // feeding, which is heard even where calls are muted.
    std::array<double, CHANNELS> channel_approach{};
    std::array<uint64_t, CHANNELS> channel_approach_steps{};
    double noise_approach = 0;
    uint64_t noise_approach_steps = 0;

    // Added up over every step, for averages.
    double wolf_steps = 0;
    double pack_steps = 0;
    double bison_steps = 0;
    double hare_steps = 0;
};

Stats& operator+=(Stats& a, const Stats& b);

// What changed between two sets of counts, for a stretch of time.
Stats operator-(Stats a, const Stats& b);

// How loud a channel was on average, over the steps a wolf was doing any of these.
double mean_call(const Stats& stats, std::initializer_list<Context> contexts, int channel);

// How much calls say about whether there is a bison: for whichever channel says the most, how much
// louder (or quieter) it was while a wolf was at or could see a bison than while it had nothing in
// sight but packmates, if that. 0 means calls don't depend on it at all (positive signalling, in
// the terms of Lowe et al. 2019).
double call_signal(const Stats& stats);

// Three numbers from 0 to 1, which is how a pack gets drawn. A daughter pack gets its mother's with
// a little drift, so lineages look alike, and immigrants stand out.
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
};

// One pack: a genome that all its wolves share, and the food store that they all eat in to and
// live off of. A pack is the thing that reproduces, by having pups while it has room, and once it
// is full, by founding a daughter pack somewhere else, with a mutated genome.
//
// Wolves in the same pack are clones, so whatever helps the pack helps every gene in each of them.
// That is the condition under which robots evolved to signal honestly and cooperate (Floreano et
// al. 2007, Waibel et al. 2011), and close to how real packs work: a breeding pair and its young,
// sharing every kill. Setting the most wolves in a pack to 1 turns that off, for comparison.
struct Pack {
    int id = 0;
    std::shared_ptr<const Genome> genome;
    Colour colour{};

    double energy = 0;
    double gathered = 0; // Over its whole life. How the best genomes are picked.
    double eaten = 0; // On this step.

    // Energy eaten per wolf per second, averaged over the last few minutes. When there isn't room
    // for every pack that can afford a pup, the ones eating best get them first.
    double intake = 0;
    double founded = 0; // Seconds, on the continent's clock.
    double pup_wait = 0;
    int members = 0;
    int daughters = 0;
    bool immigrant = false;
};

struct Wolf {
    int id = 0;
    int pack = 0;
    Body body;
    std::unique_ptr<Brain> brain;
    double health = WOLF_HEALTH;
    double age = 0;
    double lifespan = 0;

    // What it is doing, from the last time it thought.
    bool biting = false;
    std::array<float, CHANNELS> calls{};
    double turn = 0;

    // What happened to it on the last step.
    double pain = 0;
    bool at_bison = false;
    bool at_hare = false;
    bool at_stranger = false;
    bool eating = false;
    bool noisy = false; // Bit, was gored, or ate.
    bool starving = false;
    int hurt_by = 0; // 1 for another pack's wolf, 2 for a bison, 0 for nothing yet.
};

struct Bite {
    int wolf = 0;
    int pack = 0;
    double time = 0;
    Vec from; // Unit vector from the bison to the wolf, when it bit.
};

struct Prey {
    int id = 0;
    bool bison = false;
    Body body;
    double health = 1;
    double age = 0;
    double lifespan = 0;

    double alert = 0; // Seconds left fleeing.
    Vec flee; // Which way it is fleeing.
    double jink = 0; // How far off of straight away from the threat a hare is running right now.
    double jink_wait = 0;
    double wander = 0;

    double fight = 0; // Seconds left facing up to a wolf, for a bison.
    int fight_target = -1; // That wolf's id.
    int bitten_by = -1; // Id of a wolf that bit it on the last step, or -1.

    std::vector<Bite> bites; // Recent ones.
};

struct Carcass {
    Vec pos;
    double energy = 0;
    double radius = 1;
    double age = 0;
    Source source = OLD_AGE;
};

// Any wolf that made a sound, with where it was when it did.
struct Caller {
    Vec pos;
    int wolf = 0;
    int pack = 0;
    std::array<float, CHANNELS> volume{}; // Calls, which it chose to make.
    float noise = 0; // Biting or feeding, on channel 0, which it didn't choose. Heard even when calls are muted.
};

// A pack that has died, for founding new packs from when there are too few.
struct Archived {
    std::shared_ptr<const Genome> genome;
    Colour colour{};
    double gathered = 0;
};

// One continent: a walled square of plain, with herds of bison and hares that run on simple rules,
// and packs of wolves that run on evolved brains. Continents only ever meet through migration,
// which happens between steps, so each one can be stepped on its own thread, and a run comes out
// the same however many threads there are.
class Continent {
public:
    Continent(int index, uint64_t seed, const Settings& settings);

    void step(const Settings& settings);

    // The genome of the pack alive now that has gathered the most, or the best ever if none are
    // alive. False if there has never been one.
    bool best(Archived& out) const;

    // A new pack of two, somewhere at random, if there is room.
    void immigrate(const Archived& migrant, const Settings& settings);

    int index;
    double time = 0;

    std::vector<Wolf> wolves; // In order of id.
    std::map<int, Pack> packs;
    std::vector<Prey> prey; // In order of id.
    std::vector<Carcass> carcasses;
    std::vector<Caller> callers; // From the last step, which is what wolves hear on this one.

    Stats stats;

    int highest_generation() const;

    // Null if there is no wolf with that id any more.
    const Wolf* find_wolf(int id) const;

private:
    void index_everything();

    void sense(const Wolf& wolf, const Settings& settings, Inputs& in);
    void think(Wolf& wolf, const Settings& settings);
    void steer(Prey& animal, const Settings& settings);
    void move();
    void fight(const Settings& settings);
    void eat(const Settings& settings);
    void upkeep(const Settings& settings);
    void kill(Prey& animal, const Settings& settings);
    void lifecycle(const Settings& settings);
    void breed_prey(const Settings& settings);

    Pack& found_pack(std::shared_ptr<const Genome> genome, const Colour& colour, int wolves, double energy, Vec around, double spread);
    void add_wolf(Pack& pack, Vec pos, double heading);
    void add_prey(bool bison, Vec pos, double heading);
    void archive(const Pack& pack);

    Vec random_spot();
    Colour drift(const Colour& colour, double amount);

    std::mt19937_64 rng;

    int next_wolf = 0;
    int next_pack = 0;
    int next_prey = 0;

    double refound_wait = 0;

    std::vector<Body*> bodies; // Every wolf and prey, while moving them.

    std::vector<Archived> archived; // The packs that died most recently, oldest first.

    Grid wolf_grid;
    Grid prey_grid;
    Grid carcass_grid;
    Grid body_grid;
};

}

#endif // HAIL_MARY_CONTINENT_HPP
