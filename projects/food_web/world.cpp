#include "world.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <set>
#include <utility>

namespace food_web {

// How far prey see wolves and each other, all round but for a little behind.
constexpr double BISON_SIGHT = 30;
constexpr double HARE_SIGHT = 25;

// How far ahead animals feel for water, and prey for grass.
constexpr double WOLF_WATER_PROBE = 10;
constexpr double PREY_WATER_PROBE = 8;
constexpr double GRASS_PROBE = 6;

// Bison only have calves, and hares young, once they are this old, in seconds.
constexpr double BISON_ADULT = 900;
constexpr double HARE_ADULT = 300;

constexpr double BISON_LIFE_MIN = 3600;
constexpr double BISON_LIFE_MAX = 7200;
constexpr double HARE_LIFE_MIN = 1200;
constexpr double HARE_LIFE_MAX = 2400;
constexpr double WOLF_LIFE_MIN = 3600;
constexpr double WOLF_LIFE_MAX = 7200;

// What a newborn starts with, as a share of what its mother spent on it. The rest is lost.
constexpr double YOUNG_SHARE = .7;

// Health lost per second while starving, and healed per second otherwise.
constexpr double BISON_STARVING = 1;
constexpr double HARE_STARVING = .1;
constexpr double HARE_HEALING = 1;
constexpr double WOLF_HEALING = 1;
constexpr double WOLF_STARVING = .5;

constexpr double BITE_MEMORY = 10;
constexpr double SURROUND_MEMORY = 3;

constexpr double PUP_WAIT = 5;
constexpr double INTAKE_MEMORY = 300;

constexpr int START_HERDS = 3; // On each continent.
constexpr int START_HERD_SIZE = 10;
constexpr int START_HARES = 30;
constexpr int START_PACKS = 3;
constexpr int START_WOLVES = 4;
constexpr double START_PACK_ENERGY = 200;

// How often each continent is checked for having too few prey, and the rescue for wolves.
constexpr double RESCUE_WAIT = 60;
constexpr double REFOUND_WAIT = 60;

// Continents smaller than this share of the biggest are islands, which don't get founders or rescues.
constexpr double ISLAND_SHARE = .25;

constexpr int ARCHIVE_SIZE = 64;

constexpr double COLOUR_DRIFT = .05;

constexpr int THINK_CHUNK = 16; // Animals each thread takes at a time.

namespace {

double falloff(double distance)
{
    const double scaled = distance / HEAR_SCALE;

    return 1 / (1 + scaled * scaled);
}

// Logarithmic, the way hearing works. See projects/hail_mary.
float loudness(double heard)
{
    return static_cast<float>(clamp(std::log1p(heard / QUIETEST) / std::log1p(1 / QUIETEST), 0, 1));
}

// Each animal's own stream of random numbers, from -1 to 1, so that thinking can happen on any
// thread and still come out the same.
double next_noise(uint64_t& state)
{
    state ^= state << 13;
    state ^= state >> 7;
    state ^= state << 17;

    return static_cast<double>(state >> 11) * (2.0 / 9007199254740992.0) - 1;
}

// How strong a bison is for its age, from .3 to 1, the same as projects/hail_mary.
// Of its lifespan, after which a bison fades.
constexpr double BISON_OLD = .8;

double vigour(const Prey& animal, double weakest)
{
    if (!animal.bison) {
        return 1;
    }

    const double WEAKEST = weakest;
    constexpr double OLD = BISON_OLD;

    const double growing = std::min(1.0, animal.age / BISON_ADULT);
    const double fading = std::max(0.0, (animal.age - OLD * animal.lifespan) / ((1 - OLD) * animal.lifespan));

    return WEAKEST + (1 - WEAKEST) * std::min(growing, 1 - std::min(1.0, fading));
}

void grow(Prey& animal)
{
    if (!animal.bison) {
        return;
    }

    const double scale = .65 + .35 * std::min(1.0, animal.age / BISON_ADULT);

    animal.body.radius = BISON_RADIUS * scale;
    animal.body.mass = BISON_MASS * scale * scale;
}

struct Seen {
    double gap = std::numeric_limits<double>::infinity();
    Vec direction;
    int index = -1;

    bool any() const { return index >= 0; }

    void consider(double distance, Vec local, int which)
    {
        if (distance < gap) {
            gap = distance;
            direction = local;
            index = which;
        }
    }
};

struct Heard {
    double loud = 0;
    Vec from;

    void add(Vec toward, double volume)
    {
        loud += volume;
        from += toward * volume;
    }
};

// Fills in sensed values one after another.
struct Senses {
    float* in;
    int at = 0;

    void put(double value) { in[at++] = static_cast<float>(value); }

    void seen(const Seen& thing, double range)
    {
        if (!thing.any()) {
            put(0);
            put(0);
            put(0);
            return;
        }

        put(1 - thing.gap / range);
        put(thing.direction.x);
        put(thing.direction.y);
    }

    void sound(const Heard& heard, double heading)
    {
        if (heard.loud <= 1e-12) {
            put(0);
            put(0);
            put(0);
            return;
        }

        const Vec local = limit_length(to_body(heading, heard.from * (1 / heard.loud)));

        put(loudness(heard.loud));
        put(local.x);
        put(local.y);
    }
};

}

Tally& Tally::operator+=(const Tally& b)
{
    bison_born += b.bison_born;
    hares_born += b.hares_born;
    bison_killed += b.bison_killed;
    calves_killed += b.calves_killed;
    old_killed += b.old_killed;
    hungry_killed += b.hungry_killed;
    hares_killed += b.hares_killed;
    weak_killed += b.weak_killed;
    bison_starved += b.bison_starved;
    hares_starved += b.hares_starved;
    bison_old += b.bison_old;
    hares_old += b.hares_old;
    bison_arrived += b.bison_arrived;
    hares_arrived += b.hares_arrived;
    pups += b.pups;
    daughters += b.daughters;
    refounded += b.refounded;
    wolves_starved += b.wolves_starved;
    wolves_gored += b.wolves_gored;
    wolves_killed += b.wolves_killed;
    wolves_old += b.wolves_old;
    kill_wolves += b.kill_wolves;
    surround += b.surround;
    surround_kills += b.surround_kills;
    wolf_food_bison += b.wolf_food_bison;
    wolf_food_hare += b.wolf_food_hare;
    wolf_food_scavenged += b.wolf_food_scavenged;
    bison_grazed += b.bison_grazed;
    hares_grazed += b.hares_grazed;
    wolf_steps += b.wolf_steps;
    bison_steps += b.bison_steps;
    hare_steps += b.hare_steps;
    wolf_wet_steps += b.wolf_wet_steps;
    bison_wet_steps += b.bison_wet_steps;
    hare_wet_steps += b.hare_wet_steps;
    grass += b.grass;

    for (int i = 0; i < 2; i++) {
        bison_calls[i] += b.bison_calls[i];
        bison_call_steps[i] += b.bison_call_steps[i];
        hare_calls[i] += b.hare_calls[i];
        hare_call_steps[i] += b.hare_call_steps[i];
        wolf_calls[i] += b.wolf_calls[i];
        wolf_call_steps[i] += b.wolf_call_steps[i];
    }

    return *this;
}

Tally operator-(Tally a, const Tally& b)
{
    a.bison_born -= b.bison_born;
    a.hares_born -= b.hares_born;
    a.bison_killed -= b.bison_killed;
    a.calves_killed -= b.calves_killed;
    a.old_killed -= b.old_killed;
    a.hungry_killed -= b.hungry_killed;
    a.hares_killed -= b.hares_killed;
    a.weak_killed -= b.weak_killed;
    a.bison_starved -= b.bison_starved;
    a.hares_starved -= b.hares_starved;
    a.bison_old -= b.bison_old;
    a.hares_old -= b.hares_old;
    a.bison_arrived -= b.bison_arrived;
    a.hares_arrived -= b.hares_arrived;
    a.pups -= b.pups;
    a.daughters -= b.daughters;
    a.refounded -= b.refounded;
    a.wolves_starved -= b.wolves_starved;
    a.wolves_gored -= b.wolves_gored;
    a.wolves_killed -= b.wolves_killed;
    a.wolves_old -= b.wolves_old;
    a.kill_wolves -= b.kill_wolves;
    a.surround -= b.surround;
    a.surround_kills -= b.surround_kills;
    a.wolf_food_bison -= b.wolf_food_bison;
    a.wolf_food_hare -= b.wolf_food_hare;
    a.wolf_food_scavenged -= b.wolf_food_scavenged;
    a.bison_grazed -= b.bison_grazed;
    a.hares_grazed -= b.hares_grazed;
    a.wolf_steps -= b.wolf_steps;
    a.bison_steps -= b.bison_steps;
    a.hare_steps -= b.hare_steps;
    a.wolf_wet_steps -= b.wolf_wet_steps;
    a.bison_wet_steps -= b.bison_wet_steps;
    a.hare_wet_steps -= b.hare_wet_steps;
    a.grass -= b.grass;

    for (int i = 0; i < 2; i++) {
        a.bison_calls[i] -= b.bison_calls[i];
        a.bison_call_steps[i] -= b.bison_call_steps[i];
        a.hare_calls[i] -= b.hare_calls[i];
        a.hare_call_steps[i] -= b.hare_call_steps[i];
        a.wolf_calls[i] -= b.wolf_calls[i];
        a.wolf_call_steps[i] -= b.wolf_call_steps[i];
    }

    return a;
}

World::World(uint64_t seed, const Settings& settings, int thread_count)
    : terrain(HALF_WIDTH, HALF_HEIGHT, MAP_SEED), rng(seed),
      wolf_grid(HALF_WIDTH, HALF_HEIGHT, 15), prey_grid(HALF_WIDTH, HALF_HEIGHT, 15), carcass_grid(HALF_WIDTH, HALF_HEIGHT, 15), body_grid(HALF_WIDTH, HALF_HEIGHT, 5),
      wolf_caller_grid(HALF_WIDTH, HALF_HEIGHT, 60), prey_caller_grid(HALF_WIDTH, HALF_HEIGHT, 60),
      threads(std::max(1, thread_count))
{
    std::uniform_real_distribution<double> uniform(0, 1);
    std::normal_distribution<double> normal(0, 1);

    tallies.resize(static_cast<std::size_t>(continents() + 1));

    weakest = settings.bison_weakest;

    // Grass starts out fully grown everywhere on land.
    grass_columns = static_cast<int>(std::ceil(2 * HALF_WIDTH / GRASS_CELL));
    grass_rows = static_cast<int>(std::ceil(2 * HALF_HEIGHT / GRASS_CELL));
    grass.assign(static_cast<std::size_t>(grass_columns * grass_rows), 0);
    grass_land.assign(grass.size(), -1);

    const double most = settings.grass_max * GRASS_CELL * GRASS_CELL;

    for (int row = 0; row < grass_rows; row++) {
        for (int column = 0; column < grass_columns; column++) {
            const Vec middle = {-HALF_WIDTH + (column + .5) * GRASS_CELL, -HALF_HEIGHT + (row + .5) * GRASS_CELL};
            const std::size_t cell = static_cast<std::size_t>(row * grass_columns + column);

            grass_land[cell] = terrain.landmass(middle);
            grass[cell] = grass_land[cell] >= 0 ? static_cast<float>(most) : 0.0f;
        }
    }

    // Founders on every continent big enough: a few herds of bison, hares all over, and a few packs.
    const double biggest = continents() > 0 ? terrain.landmass_area(0) : 0;

    for (int land = 0; land < continents(); land++) {
        if (terrain.landmass_area(land) < ISLAND_SHARE * biggest) {
            continue;
        }

        for (int herd = 0; herd < START_HERDS; herd++) {
            const Vec middle = spot_on(land);
            const double facing = uniform(rng) * 2 * PI;

            for (int i = 0; i < START_HERD_SIZE; i++) {
                Vec spot = inside(middle + Vec{normal(rng) * 8, normal(rng) * 8}, BISON_RADIUS);

                if (terrain.landmass(spot) != land) {
                    spot = middle;
                }

                add_prey(true, spot, facing + normal(rng) * .3, prey_ancestor(true), settings.bison_capacity * .6);
            }
        }

        for (int i = 0; i < START_HARES; i++) {
            add_prey(false, spot_on(land), uniform(rng) * 2 * PI, prey_ancestor(false), settings.hare_capacity * .6);
        }

        for (int i = 0; i < START_PACKS; i++) {
            Colour colour;

            for (double& part : colour) {
                part = uniform(rng);
            }

            found_pack(wolf_ancestor(), colour, START_WOLVES, START_PACK_ENERGY, spot_on(land), 5);
        }
    }

    // Grown, with their ages spread out, so that they don't all die of old age at once.
    for (Prey& animal : prey) {
        animal.age = (animal.bison ? BISON_ADULT : HARE_ADULT) + uniform(rng) * animal.lifespan * .6;
        animal.health = animal.bison ? BISON_HEALTH * vigour(animal, weakest) : HARE_HEALTH;
        animal.breed_wait = uniform(rng) * (animal.bison ? settings.bison_calving_wait : settings.hare_breeding_wait);

        grow(animal);
    }

    listen_list(settings);
}

Vec World::spot_on(int land)
{
    return terrain.random_land_on(land, rng);
}

Vec World::inside(Vec pos, double radius) const
{
    return {clamp(pos.x, -HALF_WIDTH + radius, HALF_WIDTH - radius), clamp(pos.y, -HALF_HEIGHT + radius, HALF_HEIGHT - radius)};
}

Colour World::drift(const Colour& colour, double amount)
{
    std::normal_distribution<double> normal(0, 1);

    Colour child = colour;

    for (double& part : child) {
        part += amount * normal(rng);
        part = std::abs(part);
        part = part > 1 ? 2 - part : part;
        part = clamp(part, 0, 1);
    }

    return child;
}

int World::grass_cell(Vec p) const
{
    const int column = static_cast<int>(std::floor((p.x + HALF_WIDTH) / GRASS_CELL));
    const int row = static_cast<int>(std::floor((p.y + HALF_HEIGHT) / GRASS_CELL));

    if (column < 0 || column >= grass_columns || row < 0 || row >= grass_rows) {
        return -1;
    }

    return row * grass_columns + column;
}

double World::grass_at(Vec p, const Settings& settings) const
{
    const int cell = grass_cell(p);

    if (cell < 0) {
        return 0;
    }

    return grass[static_cast<std::size_t>(cell)] / (settings.grass_max * GRASS_CELL * GRASS_CELL + 1e-12);
}

std::vector<double> World::grass_by_continent() const
{
    std::vector<double> totals(static_cast<std::size_t>(continents()), 0);

    for (std::size_t cell = 0; cell < grass.size(); cell++) {
        if (grass_land[cell] >= 0) {
            totals[static_cast<std::size_t>(grass_land[cell])] += grass[cell];
        }
    }

    return totals;
}

int World::highest_generation(bool bison) const
{
    int highest = 0;

    for (const Prey& animal : prey) {
        if (animal.bison == bison) {
            highest = std::max(highest, animal.brain->genome()->generation);
        }
    }

    return highest;
}

const Wolf* World::find_wolf(int id) const
{
    const auto at = std::lower_bound(wolves.begin(), wolves.end(), id, [](const Wolf& wolf, int value) { return wolf.id < value; });

    return at != wolves.end() && at->id == id ? &*at : nullptr;
}

// ---------------------------------------------------------------------------------------- founders

// The same plain predator as projects/hail_mary: it runs at any hare or bison it sees, bites
// whatever it catches, goes to carcasses and eats, and otherwise ambles. On top of that, it turns
// away from water. It takes no notice of other wolves, and never calls.
std::shared_ptr<const Genome> World::wolf_ancestor()
{
    Genome genome = Genome::random(WOLF_LAYOUT, rng);

    genome.quieten(.3f);

    const Layout& layout = genome.layout;
    float* w = genome.weights.data();

    for (const int seen : {W_HARE, W_BISON, W_CARCASS}) {
        w[layout.direct(W_OUT_TURN, seen + SEEN_LEFT)] = 3;
    }

    w[layout.direct(W_OUT_TURN, W_WATER_LEFT)] = -2;
    w[layout.direct(W_OUT_TURN, W_WATER_RIGHT)] = 2;
    w[layout.direct(W_OUT_TURN, W_WATER_AHEAD)] = 1.5f;

    w[layout.bias(W_OUT_SPEED)] = .4f;
    w[layout.direct(W_OUT_SPEED, W_HARE + SEEN_CLOSE)] = 3;
    w[layout.direct(W_OUT_SPEED, W_BISON + SEEN_CLOSE)] = 3;
    w[layout.direct(W_OUT_SPEED, W_CARCASS + SEEN_CLOSE)] = .5f;
    w[layout.direct(W_OUT_SPEED, W_EATING)] = -3;

    w[layout.bias(W_OUT_BITE)] = -1.5f;
    w[layout.direct(W_OUT_BITE, W_AT_HARE)] = 3;
    w[layout.direct(W_OUT_BITE, W_AT_BISON)] = 3;

    for (int channel = 0; channel < CHANNELS; channel++) {
        w[layout.bias(W_OUT_CALL + channel)] = -.3f;
    }

    genome.squash();

    return std::make_shared<const Genome>(std::move(genome));
}

// Plain prey: it wanders towards more grass and slows down to graze, turns away from water, turns
// away from the nearest wolf and runs from it, and wanders a little at random, which hares do more.
// Bison also drift towards the nearest bison, which is all it takes to keep a herd together. Calls
// sit just short of being made at all, and nothing about facing up to wolves, or warning others,
// is there to start with.
std::shared_ptr<const Genome> World::prey_ancestor(bool bison)
{
    Genome genome = Genome::random(PREY_LAYOUT, rng);

    genome.quieten(.3f);

    const Layout& layout = genome.layout;
    float* w = genome.weights.data();

    w[layout.bias(P_OUT_SPEED)] = bison ? .15f : .2f;
    w[layout.direct(P_OUT_SPEED, P_GRASS_HERE)] = -.3f;
    w[layout.direct(P_OUT_SPEED, P_WOLF + SEEN_CLOSE)] = 3;

    w[layout.direct(P_OUT_TURN, P_GRASS_LEFT)] = 2;
    w[layout.direct(P_OUT_TURN, P_GRASS_RIGHT)] = -2;
    w[layout.direct(P_OUT_TURN, P_WATER_LEFT)] = -2.5f;
    w[layout.direct(P_OUT_TURN, P_WATER_RIGHT)] = 2.5f;
    w[layout.direct(P_OUT_TURN, P_WATER_AHEAD)] = 2;
    w[layout.direct(P_OUT_TURN, P_WOLF + SEEN_LEFT)] = -3;
    w[layout.direct(P_OUT_TURN, P_NOISE)] = bison ? .6f : 1.5f;

    if (bison) {
        w[layout.direct(P_OUT_TURN, P_KIN + SEEN_LEFT)] = 1;
    }

    w[layout.bias(P_OUT_CALL)] = -.3f;

    genome.squash();

    return std::make_shared<const Genome>(std::move(genome));
}

// ---------------------------------------------------------------------------------------- adding

Pack& World::found_pack(std::shared_ptr<const Genome> genome, const Colour& colour, int count, double energy, Vec around, double spread)
{
    std::uniform_real_distribution<double> uniform(0, 1);

    Pack& pack = packs[next_pack];

    pack.id = next_pack++;
    pack.genome = std::move(genome);
    pack.colour = colour;
    pack.energy = energy;
    pack.founded = time;

    for (int i = 0; i < count; i++) {
        const double angle = uniform(rng) * 2 * PI;
        const double distance = spread * std::sqrt(uniform(rng));

        add_wolf(pack, inside(around + heading_vector(angle) * distance, WOLF_RADIUS), uniform(rng) * 2 * PI);
    }

    return pack;
}

void World::add_wolf(Pack& pack, Vec pos, double heading)
{
    std::uniform_real_distribution<double> life(WOLF_LIFE_MIN, WOLF_LIFE_MAX);

    Wolf wolf;

    wolf.id = next_wolf++;
    wolf.pack = pack.id;
    wolf.body.pos = pos;
    wolf.body.heading = heading;
    wolf.body.radius = WOLF_RADIUS;
    wolf.body.mass = WOLF_MASS;
    wolf.body.top_speed = WOLF_SPEED;
    wolf.body.top_turn = WOLF_TURN;
    wolf.body.pickup = WOLF_PICKUP;
    wolf.brain = std::make_unique<Brain>(pack.genome);
    wolf.lifespan = life(rng);
    wolf.noise = rng() | 1;
    wolf.land = terrain.landmass(pos);

    wolves.push_back(std::move(wolf));

    pack.members++;
}

Prey& World::add_prey(bool bison, Vec pos, double heading, std::shared_ptr<const Genome> genome, double energy)
{
    std::uniform_real_distribution<double> uniform(0, 1);

    Prey animal;

    animal.id = next_prey++;
    animal.bison = bison;
    animal.body.pos = pos;
    animal.body.heading = heading;
    animal.body.radius = bison ? BISON_RADIUS : HARE_RADIUS;
    animal.body.mass = bison ? BISON_MASS : HARE_MASS;
    animal.body.top_speed = bison ? BISON_SPEED : HARE_SPEED;
    animal.body.top_turn = bison ? BISON_TURN : HARE_TURN;
    animal.body.pickup = bison ? BISON_PICKUP : HARE_PICKUP;
    animal.brain = std::make_unique<Brain>(std::move(genome));
    animal.energy = energy;
    animal.lifespan = bison ? BISON_LIFE_MIN + uniform(rng) * (BISON_LIFE_MAX - BISON_LIFE_MIN) : HARE_LIFE_MIN + uniform(rng) * (HARE_LIFE_MAX - HARE_LIFE_MIN);
    animal.health = bison ? BISON_HEALTH * vigour(animal, weakest) : HARE_HEALTH;
    animal.noise = rng() | 1;
    animal.land = terrain.landmass(pos);

    grow(animal);

    prey.push_back(std::move(animal));

    return prey.back();
}

void World::index_everything()
{
    wolf_grid.build(static_cast<int>(wolves.size()), [this](int i) { return wolves[static_cast<std::size_t>(i)].body.pos; });
    prey_grid.build(static_cast<int>(prey.size()), [this](int i) { return prey[static_cast<std::size_t>(i)].body.pos; });
    carcass_grid.build(static_cast<int>(carcasses.size()), [this](int i) { return carcasses[static_cast<std::size_t>(i)].pos; });
}

// ---------------------------------------------------------------------------------------- thinking

void World::think_wolf(Wolf& wolf, const Settings& settings)
{
    // The same senses as projects/hail_mary, which explains them, plus whether it is in the water,
    // whether there is water ahead of it, and a random number for its own use.
    const Body& body = wolf.body;
    const Pack& pack = packs.at(wolf.pack);
    const double sight = settings.sight;

    Seen hare;
    Seen bison;
    Seen carcass;
    Seen mate;
    Seen second_mate;
    Seen stranger;
    int mates_seen = 0;

    const auto look = [&](Vec pos, double radius, double& gap, Vec& local) {
        const Vec offset = pos - body.pos;

        gap = std::max(0.0, length(offset) - radius);

        if (gap > sight) {
            return false;
        }

        local = to_body(body.heading, unit(offset));

        return std::abs(std::atan2(local.y, local.x)) <= WOLF_FIELD_OF_VIEW;
    };

    wolf_grid.visit(body.pos, sight + WOLF_RADIUS, [&](int i) {
        const Wolf& other = wolves[static_cast<std::size_t>(i)];

        double gap;
        Vec local;

        if (other.id == wolf.id || !look(other.body.pos, other.body.radius, gap, local)) {
            return;
        }

        if (other.pack != wolf.pack) {
            stranger.consider(gap, local, i);
            return;
        }

        mates_seen++;

        if (gap < mate.gap) {
            second_mate = mate;
            mate = Seen{gap, local, i};
        } else {
            second_mate.consider(gap, local, i);
        }
    });

    prey_grid.visit(body.pos, sight + BISON_RADIUS, [&](int i) {
        const Prey& animal = prey[static_cast<std::size_t>(i)];

        double gap;
        Vec local;

        if (look(animal.body.pos, animal.body.radius, gap, local)) {
            (animal.bison ? bison : hare).consider(gap, local, i);
        }
    });

    carcass_grid.visit(body.pos, sight + BISON_RADIUS, [&](int i) {
        double gap;
        Vec local;

        if (carcasses[static_cast<std::size_t>(i)].energy > 0 && look(carcasses[static_cast<std::size_t>(i)].pos, carcasses[static_cast<std::size_t>(i)].radius, gap, local)) {
            carcass.consider(gap, local, i);
        }
    });

    std::array<Heard, CHANNELS> from_pack;
    std::array<Heard, CHANNELS> from_strangers;

    wolf_caller_grid.visit(body.pos, HEAR_RANGE, [&](int i) {
        const Caller& caller = wolf_callers[static_cast<std::size_t>(i)];
        const Vec offset = caller.pos - body.pos;
        const double distance = length(offset);

        if (caller.id == wolf.id || distance > HEAR_RANGE) {
            return;
        }

        const double fall = falloff(distance);
        const Vec toward = unit(offset);

        std::array<Heard, CHANNELS>& heard = caller.pack == wolf.pack ? from_pack : from_strangers;

        for (int channel = 0; channel < CHANNELS; channel++) {
            heard[static_cast<std::size_t>(channel)].add(toward, (caller.volume[static_cast<std::size_t>(channel)] + (channel == 0 ? caller.noise : 0)) * fall);
        }
    });

    const auto water = [&](double angle) { return terrain.is_land(body.pos + heading_vector(body.heading + angle) * WOLF_WATER_PROBE) ? 0.0 : 1.0; };

    std::array<float, WOLF_INPUTS> in{};
    Senses senses{in.data()};

    const Vec velocity = to_body(body.heading, body.vel);

    senses.put(clamp(velocity.x / WOLF_SPEED, -1, 1));
    senses.put(clamp(velocity.y / WOLF_SPEED, -1, 1));
    senses.put(wolf.turn);
    senses.put(clamp(wolf.health / WOLF_HEALTH, 0, 1));
    senses.put(std::tanh(pack.energy / (settings.pup_reserve * std::max(1, pack.members) + 1e-9)));
    senses.put(clamp(wolf.pain, 0, 1));
    senses.put(wolf.at_bison ? 1 : 0);
    senses.put(wolf.at_hare ? 1 : 0);
    senses.put(wolf.at_stranger ? 1 : 0);
    senses.put(wolf.eating ? 1 : 0);
    senses.put(body.wet ? 1 : 0);
    senses.put(next_noise(wolf.noise));
    senses.put(1);

    senses.seen(hare, sight);
    senses.seen(bison, sight);

    if (bison.any()) {
        const Prey& animal = prey[static_cast<std::size_t>(bison.index)];

        senses.put(dot(heading_vector(animal.body.heading), unit(body.pos - animal.body.pos)));
        senses.put(1 - clamp(animal.health / BISON_HEALTH, 0, 1));
    } else {
        senses.put(0);
        senses.put(0);
    }

    senses.seen(carcass, sight);
    senses.seen(mate, sight);
    senses.seen(second_mate, sight);
    senses.seen(stranger, sight);
    senses.put(std::min(1.0, mates_seen / 5.0));
    senses.put(water(PI / 4));
    senses.put(water(0));
    senses.put(water(-PI / 4));

    for (const Heard& heard : from_pack) {
        senses.sound(heard, body.heading);
    }

    for (const Heard& heard : from_strangers) {
        senses.sound(heard, body.heading);
    }

    std::array<float, WOLF_OUTPUTS> out{};

    wolf.brain->think(in.data(), out.data());

    const double speed = clamp(out[W_OUT_SPEED], -1, 1);

    wolf.body.want_speed = speed >= 0 ? speed * WOLF_SPEED : speed * .3 * WOLF_SPEED;
    wolf.turn = clamp(out[W_OUT_TURN], -1, 1);
    wolf.body.want_turn = wolf.turn * WOLF_TURN;
    wolf.biting = out[W_OUT_BITE] > 0;

    for (int channel = 0; channel < CHANNELS; channel++) {
        wolf.calls[static_cast<std::size_t>(channel)] = static_cast<float>(clamp(out[W_OUT_CALL + channel], 0, 1));
    }

    wolf.saw_bison = bison.any();
}

void World::think_prey(Prey& animal, const Settings& settings)
{
    // All from its own point of view:
    // * About itself: how fast it is going ahead and to the left, how hard it was turning, its
    //   health, how full it is, how much it got hurt, whether it is in the water, a random number
    //   for its own use, a constant, and whether it is still young.
    // * Grass: how much there is where it stands, and a few meters off 45 degrees to the left,
    //   straight ahead, and 45 degrees to the right.
    // * Water: whether there is any a few meters off in the same three directions.
    // * The two nearest wolves it can see, how close and which way, and how many it can see.
    // * The nearest of its own kind, the same way, and how many of them it can see.
    // * What it hears: its own kind calling, along with the noise of one of them being bitten, and
    //   wolves calling, biting, and feeding. Each as how loud, and which way it is coming from.
    Body& body = animal.body;

    const double sight = animal.bison ? BISON_SIGHT : HARE_SIGHT;
    const double top = animal.bison ? BISON_SPEED : HARE_SPEED;

    // A healthy grown bison runs as fast as a wolf. Calves, old ones, and the hurt are slower.
    if (animal.bison) {
        body.top_speed = BISON_SPEED * (.6 + .4 * clamp(animal.health / BISON_HEALTH, 0, 1));
    }

    Seen wolf;
    Seen second_wolf;
    Seen kin;
    int wolves_seen = 0;
    int kin_seen = 0;

    const auto look = [&](Vec pos, double radius, double& gap, Vec& local) {
        const Vec offset = pos - body.pos;

        gap = std::max(0.0, length(offset) - radius);

        if (gap > sight) {
            return false;
        }

        local = to_body(body.heading, unit(offset));

        return std::abs(std::atan2(local.y, local.x)) <= PREY_FIELD_OF_VIEW;
    };

    wolf_grid.visit(body.pos, sight + WOLF_RADIUS, [&](int i) {
        double gap;
        Vec local;

        if (!look(wolves[static_cast<std::size_t>(i)].body.pos, WOLF_RADIUS, gap, local)) {
            return;
        }

        wolves_seen++;

        if (gap < wolf.gap) {
            second_wolf = wolf;
            wolf = Seen{gap, local, i};
        } else {
            second_wolf.consider(gap, local, i);
        }
    });

    prey_grid.visit(body.pos, sight + BISON_RADIUS, [&](int i) {
        const Prey& other = prey[static_cast<std::size_t>(i)];

        double gap;
        Vec local;

        if (other.id == animal.id || other.bison != animal.bison || !look(other.body.pos, other.body.radius, gap, local)) {
            return;
        }

        kin_seen++;
        kin.consider(gap, local, i);
    });

    Heard kin_calls;
    Heard wolf_sounds;

    prey_caller_grid.visit(body.pos, HEAR_RANGE, [&](int i) {
        const Caller& caller = prey_callers[static_cast<std::size_t>(i)];
        const Vec offset = caller.pos - body.pos;
        const double distance = length(offset);

        if (caller.id == animal.id || (caller.pack == 1) != animal.bison || distance > HEAR_RANGE) {
            return;
        }

        kin_calls.add(unit(offset), (caller.volume[0] + caller.noise) * falloff(distance));
    });

    wolf_caller_grid.visit(body.pos, HEAR_RANGE, [&](int i) {
        const Caller& caller = wolf_callers[static_cast<std::size_t>(i)];
        const Vec offset = caller.pos - body.pos;
        const double distance = length(offset);

        if (distance > HEAR_RANGE) {
            return;
        }

        double volume = caller.noise;

        for (float call : caller.volume) {
            volume += call;
        }

        wolf_sounds.add(unit(offset), volume * falloff(distance));
    });

    const auto probe = [&](double angle, double distance) { return body.pos + heading_vector(body.heading + angle) * distance; };
    const auto water = [&](double angle) { return terrain.is_land(probe(angle, PREY_WATER_PROBE)) ? 0.0 : 1.0; };

    const double capacity = animal.bison ? settings.bison_capacity : settings.hare_capacity;
    const double adult = animal.bison ? BISON_ADULT : HARE_ADULT;

    std::array<float, PREY_INPUTS> in{};
    Senses senses{in.data()};

    const Vec velocity = to_body(body.heading, body.vel);

    senses.put(clamp(velocity.x / top, -1, 1));
    senses.put(clamp(velocity.y / top, -1, 1));
    senses.put(animal.turn);
    senses.put(clamp(animal.health / (animal.bison ? BISON_HEALTH : HARE_HEALTH), 0, 1));
    senses.put(clamp(animal.energy / capacity, 0, 1));
    senses.put(clamp(animal.pain / (animal.bison ? 2.0 : .5), 0, 1));
    senses.put(body.wet ? 1 : 0);
    senses.put(next_noise(animal.noise));
    senses.put(1);
    senses.put(animal.age < adult ? 1 : 0);

    senses.put(grass_at(body.pos, settings));
    senses.put(grass_at(probe(PI / 4, GRASS_PROBE), settings));
    senses.put(grass_at(probe(0, GRASS_PROBE), settings));
    senses.put(grass_at(probe(-PI / 4, GRASS_PROBE), settings));

    senses.put(water(PI / 4));
    senses.put(water(0));
    senses.put(water(-PI / 4));

    senses.seen(wolf, sight);
    senses.seen(second_wolf, sight);
    senses.put(std::min(1.0, wolves_seen / 5.0));
    senses.seen(kin, sight);
    senses.put(std::min(1.0, kin_seen / 10.0));

    senses.sound(kin_calls, body.heading);
    senses.sound(wolf_sounds, body.heading);

    std::array<float, PREY_OUTPUTS> out{};

    animal.brain->think(in.data(), out.data());

    const double speed = clamp(out[P_OUT_SPEED], -1, 1);

    body.want_speed = speed >= 0 ? speed * top : speed * .3 * top;
    animal.turn = clamp(out[P_OUT_TURN], -1, 1);
    body.want_turn = animal.turn * body.top_turn;
    animal.call = clamp(out[P_OUT_CALL], 0, 1);
    animal.saw_wolf = wolf.any();
}

// ---------------------------------------------------------------------------------------- moving

void World::move(const Settings& settings)
{
    // Every body the same way: turn, then ease towards the velocity it wants along the way it is
    // facing, then move. In the water, everybody swims, at a fraction of their top speed.
    bodies.clear();

    for (Wolf& wolf : wolves) {
        bodies.push_back(&wolf.body);
    }

    for (Prey& animal : prey) {
        bodies.push_back(&animal.body);
    }

    for (Body* body : bodies) {
        const double most = body->top_speed * (body->wet ? settings.swim_speed : 1);

        body->heading = wrap_angle(body->heading + clamp(body->want_turn, -body->top_turn, body->top_turn) * DT);

        const double speed = clamp(body->want_speed, -.3 * most, most);
        const Vec wanted = heading_vector(body->heading) * speed;
        const double ease = 1 - std::exp(-DT / body->pickup);

        body->vel += (wanted - body->vel) * ease;
        body->pos += body->vel * DT;
    }

    // Bodies push each other apart, by mass, and stop running in to each other.
    const int count = static_cast<int>(bodies.size());

    for (int pass = 0; pass < 2; pass++) {
        body_grid.build(count, [this](int i) { return bodies[static_cast<std::size_t>(i)]->pos; });

        for (int i = 0; i < count; i++) {
            Body& a = *bodies[static_cast<std::size_t>(i)];

            body_grid.visit(a.pos, a.radius + BISON_RADIUS, [&](int j) {
                if (j <= i) {
                    return;
                }

                Body& b = *bodies[static_cast<std::size_t>(j)];

                const Vec offset = b.pos - a.pos;
                const double distance = length(offset);
                const double touching = a.radius + b.radius;

                if (distance >= touching) {
                    return;
                }

                const Vec normal = distance > 1e-9 ? offset * (1 / distance) : heading_vector(i * .7 + j);
                const double overlap = touching - distance;
                const double give_a = 1 / a.mass;
                const double give_b = 1 / b.mass;
                const double total = give_a + give_b;

                a.pos -= normal * (overlap * give_a / total);
                b.pos += normal * (overlap * give_b / total);

                const double closing = dot(b.vel - a.vel, normal);

                if (closing < 0) {
                    const double impulse = -closing / total;

                    a.vel -= normal * (impulse * give_a);
                    b.vel += normal * (impulse * give_b);
                }
            });
        }
    }

    // The walls round the edge of the map.
    for (Body* body : bodies) {
        const double x_edge = HALF_WIDTH - body->radius;
        const double y_edge = HALF_HEIGHT - body->radius;

        if (body->pos.x < -x_edge) {
            body->pos.x = -x_edge;
            body->vel.x = std::max(0.0, body->vel.x);
        } else if (body->pos.x > x_edge) {
            body->pos.x = x_edge;
            body->vel.x = std::min(0.0, body->vel.x);
        }

        if (body->pos.y < -y_edge) {
            body->pos.y = -y_edge;
            body->vel.y = std::max(0.0, body->vel.y);
        } else if (body->pos.y > y_edge) {
            body->pos.y = y_edge;
            body->vel.y = std::min(0.0, body->vel.y);
        }

        body->wet = !terrain.is_land(body->pos);
    }

    for (Wolf& wolf : wolves) {
        wolf.land = terrain.landmass(wolf.body.pos);
    }

    for (Prey& animal : prey) {
        animal.land = terrain.landmass(animal.body.pos);
    }
}

// ---------------------------------------------------------------------------------------- fighting

void World::fight(const Settings& settings)
{
    for (Wolf& wolf : wolves) {
        wolf.pain = 0;
        wolf.at_bison = false;
        wolf.at_hare = false;
        wolf.at_stranger = false;
        wolf.eating = false;
        wolf.noisy = false;
    }

    for (Prey& animal : prey) {
        animal.bitten_by = -1;
        animal.pain = 0;
        animal.noisy = false;
    }

    const double damage = settings.bite_damage * DT;

    for (Wolf& wolf : wolves) {
        const Body& body = wolf.body;

        Wolf* wolf_target = nullptr;
        Prey* prey_target = nullptr;
        double nearest = std::numeric_limits<double>::infinity();

        wolf_grid.visit(body.pos, 2 * WOLF_RADIUS + BITE_REACH, [&](int i) {
            Wolf& other = wolves[static_cast<std::size_t>(i)];

            if (other.id == wolf.id || other.pack == wolf.pack) {
                return;
            }

            const Vec offset = other.body.pos - body.pos;
            const double gap = length(offset) - body.radius - other.body.radius;

            if (gap > BITE_REACH || std::abs(bearing(body.heading, offset)) > BITE_ANGLE) {
                return;
            }

            wolf.at_stranger = true;

            if (gap < nearest) {
                nearest = gap;
                wolf_target = &other;
                prey_target = nullptr;
            }
        });

        prey_grid.visit(body.pos, WOLF_RADIUS + BISON_RADIUS + BITE_REACH, [&](int i) {
            Prey& animal = prey[static_cast<std::size_t>(i)];

            const Vec offset = animal.body.pos - body.pos;
            const double gap = length(offset) - body.radius - animal.body.radius;

            if (gap > BITE_REACH || std::abs(bearing(body.heading, offset)) > BITE_ANGLE) {
                return;
            }

            (animal.bison ? wolf.at_bison : wolf.at_hare) = true;

            if (gap < nearest) {
                nearest = gap;
                prey_target = &animal;
                wolf_target = nullptr;
            }
        });

        if (!wolf.biting || (!wolf_target && !prey_target)) {
            continue;
        }

        wolf.noisy = true;

        if (wolf_target) {
            wolf_target->health -= damage;
            wolf_target->pain += damage;
            wolf_target->hurt_by = 1;
            continue;
        }

        Prey& animal = *prey_target;

        animal.health -= damage;
        animal.pain += damage;
        animal.bitten_by = wolf.id;
        animal.noisy = true;

        if (!animal.bison) {
            continue;
        }

        const Vec from = unit(body.pos - animal.body.pos);
        const auto previous = std::find_if(animal.bites.begin(), animal.bites.end(), [&](const Bite& bite) { return bite.wolf == wolf.id; });

        if (previous != animal.bites.end()) {
            previous->time = time;
            previous->from = from;
        } else {
            animal.bites.push_back({wolf.id, wolf.pack, time, from});
        }
    }

    // Horns: anything in front of a bison gets hurt, whether the bison means it or not. Facing up to
    // wolves is something bison can evolve to do.
    const double gore = settings.horn_damage * DT;

    for (Prey& animal : prey) {
        if (!animal.bison) {
            continue;
        }

        const Body& body = animal.body;

        wolf_grid.visit(body.pos, BISON_RADIUS + WOLF_RADIUS + HORN_REACH, [&](int i) {
            Wolf& wolf = wolves[static_cast<std::size_t>(i)];

            const Vec offset = wolf.body.pos - body.pos;
            const double gap = length(offset) - body.radius - wolf.body.radius;

            if (gap > HORN_REACH || std::abs(bearing(body.heading, offset)) > HORN_ANGLE) {
                return;
            }

            wolf.health -= gore;
            wolf.pain += gore;
            wolf.hurt_by = 2;
            wolf.noisy = true;
        });

        animal.bites.erase(std::remove_if(animal.bites.begin(), animal.bites.end(), [this](const Bite& bite) { return bite.time < time - BITE_MEMORY; }), animal.bites.end());
    }
}

// ---------------------------------------------------------------------------------------- eating

void World::eat(const Settings& settings)
{
    for (Wolf& wolf : wolves) {
        Carcass* nearest = nullptr;
        double best_gap = std::numeric_limits<double>::infinity();

        carcass_grid.visit(wolf.body.pos, WOLF_RADIUS + BISON_RADIUS + EAT_REACH, [&](int i) {
            Carcass& carcass = carcasses[static_cast<std::size_t>(i)];

            if (carcass.energy <= 0) {
                return;
            }

            const double gap = length(carcass.pos - wolf.body.pos) - carcass.radius - wolf.body.radius;

            if (gap <= EAT_REACH && gap < best_gap) {
                best_gap = gap;
                nearest = &carcass;
            }
        });

        if (!nearest) {
            continue;
        }

        Pack& pack = packs.at(wolf.pack);

        const double taken = std::min(nearest->energy, settings.eat_rate * DT);

        nearest->energy -= taken;
        pack.energy += taken;
        pack.gathered += taken;
        pack.eaten += taken;
        wolf.eating = true;
        wolf.noisy = true;

        Tally& here = tally(wolf.land);

        (nearest->source == HUNTED_BISON ? here.wolf_food_bison : nearest->source == HUNTED_HARE ? here.wolf_food_hare : here.wolf_food_scavenged) += taken;
    }

    // Grazing, up to how much each animal can hold. A hare crops the one patch it is standing on. A
    // bison crops every patch under its muzzle and body, up to four of them, the way big grazers can
    // live on shorter grass than small ones, so that hares can't simply out-breed bison for food.
    for (Prey& animal : prey) {
        if (animal.body.wet) {
            continue;
        }

        const double rate = animal.bison ? settings.bison_graze : settings.hare_graze;
        double room = std::max(0.0, (animal.bison ? settings.bison_capacity : settings.hare_capacity) - animal.energy);
        double wanted = rate * DT;

        std::array<int, 4> patches = {grass_cell(animal.body.pos), -1, -1, -1};

        if (animal.bison) {
            const double reach = animal.body.radius;
            const Vec corners[3] = {animal.body.pos + Vec{reach, 0}, animal.body.pos + Vec{0, reach}, animal.body.pos + Vec{reach, reach}};
            const Vec flip = {animal.body.pos.x - std::floor((animal.body.pos.x + HALF_WIDTH) / GRASS_CELL) * GRASS_CELL + HALF_WIDTH > GRASS_CELL / 2 ? 1.0 : -1.0,
                              animal.body.pos.y - std::floor((animal.body.pos.y + HALF_HEIGHT) / GRASS_CELL) * GRASS_CELL + HALF_HEIGHT > GRASS_CELL / 2 ? 1.0 : -1.0};

            // Towards whichever neighbouring patches it is nearest.
            for (int i = 0; i < 3; i++) {
                const Vec offset = corners[i] - animal.body.pos;

                patches[static_cast<std::size_t>(i + 1)] = grass_cell(animal.body.pos + Vec{offset.x * flip.x, offset.y * flip.y});
            }
        }

        double taken = 0;

        for (std::size_t i = 0; i < patches.size(); i++) {
            const int cell = patches[i];

            if (cell < 0 || std::find(patches.begin(), patches.begin() + static_cast<std::ptrdiff_t>(i), cell) != patches.begin() + static_cast<std::ptrdiff_t>(i)
                || grass_land[static_cast<std::size_t>(cell)] < 0) {
                continue;
            }

            float& patch = grass[static_cast<std::size_t>(cell)];
            const double bite = std::min({static_cast<double>(patch), wanted, room});

            patch -= static_cast<float>(bite);
            taken += bite;
            wanted -= bite;
            room -= bite;
        }

        animal.energy += taken;

        (animal.bison ? tally(animal.land).bison_grazed : tally(animal.land).hares_grazed) += taken;
    }
}

void World::upkeep(const Settings& settings)
{
    for (Wolf& wolf : wolves) {
        double calling = 0;

        for (float call : wolf.calls) {
            calling += call;
        }

        const double pace = length(wolf.body.vel) / WOLF_SPEED;
        const double spent = settings.living_energy + settings.running_energy * pace * pace + (wolf.biting ? settings.bite_energy : 0) + settings.call_energy * calling
                             + (wolf.body.wet ? settings.wolf_swim_energy : 0);

        packs.at(wolf.pack).energy -= spent * DT;
    }

    const double most = settings.daughter_energy + settings.pup_reserve * std::max(1.0, settings.max_pack);

    for (auto& entry : packs) {
        entry.second.energy = clamp(entry.second.energy, 0, most);
    }

    for (Wolf& wolf : wolves) {
        wolf.starving = packs.at(wolf.pack).energy <= 0;

        if (wolf.starving) {
            wolf.health -= WOLF_STARVING * DT;
        } else {
            wolf.health = std::min(WOLF_HEALTH, wolf.health + WOLF_HEALING * DT);
        }
    }

    for (Prey& animal : prey) {
        const double top = animal.bison ? BISON_SPEED : HARE_SPEED;
        const double pace = length(animal.body.vel) / top;

        const double spent = animal.bison
                                 ? settings.bison_metabolism + settings.bison_running * pace * pace + (animal.body.wet ? settings.bison_swim_energy : 0)
                                 : settings.hare_metabolism + settings.hare_running * pace * pace + (animal.body.wet ? settings.hare_swim_energy : 0);

        animal.energy -= (spent + settings.prey_call_energy * animal.call) * DT;

        animal.starving = animal.energy <= 0;

        if (animal.starving) {
            animal.energy = 0;
            animal.health -= (animal.bison ? BISON_STARVING : HARE_STARVING) * DT;
        } else if (animal.bison) {
            const double strength = vigour(animal, weakest);

            animal.health = std::min(BISON_HEALTH * strength, animal.health + settings.bison_healing * strength * DT);
        } else {
            animal.health = std::min(HARE_HEALTH, animal.health + HARE_HEALING * DT);
        }
    }
}

// ---------------------------------------------------------------------------------------- living and dying

void World::kill(Prey& animal, const Settings& settings)
{
    Tally& here = tally(animal.land);

    Carcass carcass;

    carcass.pos = animal.body.pos;
    carcass.radius = animal.body.radius;

    if (!animal.bison) {
        here.hares_killed++;

        carcass.energy = settings.hare_energy;
        carcass.source = HUNTED_HARE;
        carcasses.push_back(carcass);
        return;
    }

    here.bison_killed++;

    if (vigour(animal, weakest) < .99) {
        here.weak_killed++;
    }

    if (animal.age < BISON_ADULT) {
        here.calves_killed++;
    } else if (animal.age > BISON_OLD * animal.lifespan) {
        here.old_killed++;
    } else if (animal.starving || animal.energy < .15 * settings.bison_capacity) {
        here.hungry_killed++;
    }

    carcass.energy = settings.bison_energy * vigour(animal, weakest);
    carcass.source = HUNTED_BISON;
    carcasses.push_back(carcass);

    std::set<int> killers;

    Vec round;
    int recent = 0;

    for (const Bite& bite : animal.bites) {
        if (bite.time < time - BITE_MEMORY) {
            continue;
        }

        killers.insert(bite.wolf);

        if (bite.time >= time - SURROUND_MEMORY) {
            round += bite.from;
            recent++;
        }
    }

    here.kill_wolves += static_cast<double>(killers.size());

    if (recent >= 2) {
        here.surround += length(round) / recent;
        here.surround_kills++;
    }
}

void World::lifecycle(const Settings& settings)
{
    // Prey that were killed, starved, or died of old age. Grown ones go in to the archive, for
    // rescues to come from if a kind dies out everywhere.
    for (Prey& animal : prey) {
        animal.age += DT;
        animal.breed_wait -= DT;

        grow(animal);

        const bool old = animal.age >= animal.lifespan;

        if (animal.health > 0 && !old) {
            continue;
        }

        Tally& here = tally(animal.land);

        if (animal.health <= 0 && !animal.starving) {
            kill(animal, settings);
        } else {
            if (animal.bison) {
                (old ? here.bison_old : here.bison_starved)++;

                Carcass carcass;

                carcass.pos = animal.body.pos;
                carcass.radius = animal.body.radius;
                carcass.energy = settings.dead_bison_energy;
                carcass.source = SCAVENGED;

                carcasses.push_back(carcass);
            } else {
                (old ? here.hares_old : here.hares_starved)++;
            }
        }

        if (animal.age >= (animal.bison ? BISON_ADULT : HARE_ADULT)) {
            auto& archive = animal.bison ? bison_archive : hare_archive;

            archive.push_back(animal.brain->genome());

            if (static_cast<int>(archive.size()) > ARCHIVE_SIZE) {
                archive.erase(archive.begin());
            }
        }

        animal.health = -1;
    }

    prey.erase(std::remove_if(prey.begin(), prey.end(), [](const Prey& animal) { return animal.health < 0 || animal.health == 0; }), prey.end());

    // Wolves that ran out of health, or time.
    for (Wolf& wolf : wolves) {
        wolf.age += DT;

        if (wolf.health > 0 && wolf.age < wolf.lifespan) {
            continue;
        }

        Tally& here = tally(wolf.land);

        if (wolf.health > 0) {
            here.wolves_old++;
        } else if (wolf.starving) {
            here.wolves_starved++;
        } else if (wolf.hurt_by == 2) {
            here.wolves_gored++;
        } else if (wolf.hurt_by == 1) {
            here.wolves_killed++;
        } else {
            here.wolves_starved++;
        }

        wolf.health = 0;
        packs.at(wolf.pack).members--;
    }

    wolves.erase(std::remove_if(wolves.begin(), wolves.end(), [](const Wolf& wolf) { return wolf.health <= 0; }), wolves.end());

    for (auto entry = packs.begin(); entry != packs.end();) {
        if (entry->second.members <= 0) {
            wolf_archive.push_back({entry->second.genome, entry->second.colour, entry->second.gathered});

            if (static_cast<int>(wolf_archive.size()) > ARCHIVE_SIZE) {
                wolf_archive.erase(wolf_archive.begin());
            }

            entry = packs.erase(entry);
        } else {
            ++entry;
        }
    }

    breed_wolves(settings);
    breed_prey(settings);

    for (Carcass& carcass : carcasses) {
        carcass.age += DT;
    }

    const double lifetime = settings.carcass_lifetime;

    carcasses.erase(std::remove_if(carcasses.begin(), carcasses.end(), [lifetime](const Carcass& carcass) { return carcass.energy <= 1e-6 || carcass.age >= lifetime; }),
                    carcasses.end());
}

void World::breed_wolves(const Settings& settings)
{
    // The same as projects/hail_mary: packs have pups while they have room and food to spare, the
    // ones eating best first, and full, well fed packs send their two youngest off as a daughter
    // pack with a mutated genome.
    std::uniform_real_distribution<double> uniform(0, 1);

    const int most = std::max(1, static_cast<int>(settings.max_pack));
    const int founders = std::min(2, most);

    std::vector<Pack*> by_intake;

    for (auto& entry : packs) {
        Pack& pack = entry.second;

        pack.intake += (pack.eaten / DT / std::max(1, pack.members) - pack.intake) * DT / INTAKE_MEMORY;
        pack.eaten = 0;

        by_intake.push_back(&pack);
    }

    std::stable_sort(by_intake.begin(), by_intake.end(), [](const Pack* a, const Pack* b) { return a->intake > b->intake; });

    std::vector<std::pair<int, Vec>> new_pups;

    for (Pack* growing : by_intake) {
        Pack& pack = *growing;

        pack.pup_wait -= DT;

        const double crowd = static_cast<double>(wolves.size() + new_pups.size());

        if (pack.members < most) {
            if (pack.energy >= settings.pup_cost + settings.pup_reserve * pack.members && pack.pup_wait <= 0 && crowd + 1 <= settings.max_wolves) {
                std::vector<const Wolf*> members;

                for (const Wolf& wolf : wolves) {
                    if (wolf.pack == pack.id) {
                        members.push_back(&wolf);
                    }
                }

                const Wolf& parent = *members[static_cast<std::size_t>(uniform(rng) * static_cast<double>(members.size())) % members.size()];
                const Vec spot = parent.body.pos - heading_vector(parent.body.heading) * (2 * WOLF_RADIUS + .2);

                new_pups.push_back({pack.id, inside(spot, WOLF_RADIUS)});

                pack.energy -= settings.pup_cost;
                pack.pup_wait = PUP_WAIT;
                pack.members++;

                tally(parent.land).pups++;
            }

            continue;
        }

        if (pack.energy < settings.daughter_energy || most < 3) {
            continue;
        }

        pack.energy -= settings.daughter_energy / 2;
        pack.daughters++;

        auto genome = std::make_shared<const Genome>(pack.genome->mutated(settings.mutation_chance, rng));
        const Colour colour = drift(pack.colour, COLOUR_DRIFT);

        Pack& daughter = found_pack(std::move(genome), colour, 0, settings.daughter_energy / 2, Vec{}, 0);

        std::vector<Wolf*> members;

        for (Wolf& wolf : wolves) {
            if (wolf.pack == pack.id) {
                members.push_back(&wolf);
            }
        }

        std::stable_sort(members.begin(), members.end(), [](const Wolf* a, const Wolf* b) { return a->age < b->age; });

        for (int i = 0; i < founders && i < static_cast<int>(members.size()); i++) {
            Wolf& wolf = *members[static_cast<std::size_t>(i)];

            wolf.pack = daughter.id;
            wolf.brain = std::make_unique<Brain>(daughter.genome);

            daughter.members++;
            pack.members--;
        }

        if (!members.empty()) {
            tally(members.front()->land).daughters++;
        }
    }

    for (const auto& pup : new_pups) {
        Pack& pack = packs.at(pup.first);

        pack.members--; // add_wolf() counts it again.

        add_wolf(pack, pup.second, uniform(rng) * 2 * PI);
    }
}

void World::breed_prey(const Settings& settings)
{
    // Each grown, well fed bison or hare has young of its own, at its side, with its genome mutated.
    std::uniform_real_distribution<double> uniform(0, 1);
    std::normal_distribution<double> normal(0, 1);

    struct Birth {
        bool bison;
        Vec pos;
        double heading;
        std::shared_ptr<const Genome> genome;
        double energy;
        int land;
    };

    std::vector<Birth> born;

    double room = settings.max_prey - static_cast<double>(prey.size());

    for (Prey& mother : prey) {
        if (room < 1) {
            break;
        }

        const bool bison = mother.bison;
        const double capacity = bison ? settings.bison_capacity : settings.hare_capacity;
        const double cost = bison ? settings.bison_calf_cost : settings.hare_kit_cost;

        if (mother.age < (bison ? BISON_ADULT : HARE_ADULT) || mother.breed_wait > 0 || mother.energy < settings.breed_fill * capacity || mother.energy < cost
            || mother.body.wet) {
            continue;
        }

        mother.energy -= cost;
        mother.breed_wait = bison ? settings.bison_calving_wait : settings.hare_breeding_wait;

        const Vec spot = mother.body.pos - heading_vector(mother.body.heading) * (2 * mother.body.radius + .3) + Vec{normal(rng) * .3, normal(rng) * .3};

        born.push_back({bison, inside(spot, mother.body.radius), mother.body.heading, std::make_shared<const Genome>(mother.brain->genome()->mutated(settings.mutation_chance, rng)),
                        cost * YOUNG_SHARE, mother.land});

        room--;
    }

    for (Birth& birth : born) {
        Prey& young = add_prey(birth.bison, birth.pos, birth.heading, std::move(birth.genome), birth.energy);

        young.age = 0;
        young.health = birth.bison ? BISON_HEALTH * vigour(young, weakest) : HARE_HEALTH;

        grow(young);

        (birth.bison ? tally(birth.land).bison_born : tally(birth.land).hares_born)++;
    }
}

void World::rescue(const Settings& settings)
{
    // Not how numbers are kept up: only a way back from dying out. Every so often, each continent
    // with almost none of a kind gets one grown one, wandering in from somewhere else. Its genes come
    // from a live one elsewhere if there is one, then from one that died lately, then from scratch.
    std::uniform_real_distribution<double> uniform(0, 1);

    rescue_wait -= DT;

    if (rescue_wait <= 0) {
        rescue_wait = RESCUE_WAIT;

        const double biggest = continents() > 0 ? terrain.landmass_area(0) : 0;

        for (int land = 0; land < continents(); land++) {
            if (terrain.landmass_area(land) < ISLAND_SHARE * biggest) {
                continue;
            }

            for (const bool bison : {true, false}) {
                int here = 0;
                std::vector<std::size_t> elsewhere;

                for (std::size_t i = 0; i < prey.size(); i++) {
                    if (prey[i].bison != bison) {
                        continue;
                    }

                    if (prey[i].land == land) {
                        here++;
                    } else {
                        elsewhere.push_back(i);
                    }
                }

                if (here >= (bison ? settings.bison_floor : settings.hare_floor)) {
                    continue;
                }

                std::shared_ptr<const Genome> genome;
                const auto& archive = bison ? bison_archive : hare_archive;

                if (!elsewhere.empty()) {
                    genome = prey[elsewhere[static_cast<std::size_t>(uniform(rng) * static_cast<double>(elsewhere.size())) % elsewhere.size()]].brain->genome();
                } else if (!archive.empty()) {
                    genome = archive[static_cast<std::size_t>(uniform(rng) * static_cast<double>(archive.size())) % archive.size()];
                } else {
                    genome = prey_ancestor(bison);
                }

                Prey& animal = add_prey(bison, spot_on(land), uniform(rng) * 2 * PI, std::make_shared<const Genome>(genome->mutated(settings.mutation_chance, rng)),
                                        (bison ? settings.bison_capacity : settings.hare_capacity) * .6);

                animal.age = (bison ? BISON_ADULT : HARE_ADULT) + uniform(rng) * .3 * animal.lifespan;
                animal.health = bison ? BISON_HEALTH * vigour(animal, weakest) : HARE_HEALTH;

                grow(animal);

                (bison ? tally(land).bison_arrived : tally(land).hares_arrived)++;
            }
        }
    }

    // Wolves the same way: a continent with fewer packs than that gets one, from the best of three
    // packs that died lately, as a pack wandering in from somewhere else.
    refound_wait -= DT;

    if (refound_wait > 0) {
        return;
    }

    refound_wait = REFOUND_WAIT;

    const double biggest = terrain.landmass_area(0);

    for (int land = 0; land < continents(); land++) {
        if (terrain.landmass_area(land) < ISLAND_SHARE * biggest) {
            continue;
        }

        std::set<int> here;

        for (const Wolf& wolf : wolves) {
            if (wolf.land == land) {
                here.insert(wolf.pack);
            }
        }

        if (static_cast<double>(here.size()) >= settings.min_packs) {
            continue;
        }

        std::shared_ptr<const Genome> genome;
        Colour colour;

        if (wolf_archive.empty()) {
            genome = wolf_ancestor();

            for (double& part : colour) {
                part = uniform(rng);
            }
        } else {
            std::size_t pick = static_cast<std::size_t>(uniform(rng) * static_cast<double>(wolf_archive.size())) % wolf_archive.size();

            for (int round = 1; round < 3; round++) {
                const std::size_t other = static_cast<std::size_t>(uniform(rng) * static_cast<double>(wolf_archive.size())) % wolf_archive.size();

                if (wolf_archive[other].gathered > wolf_archive[pick].gathered) {
                    pick = other;
                }
            }

            genome = std::make_shared<const Genome>(wolf_archive[pick].genome->mutated(settings.mutation_chance, rng));
            colour = drift(wolf_archive[pick].colour, COLOUR_DRIFT);
        }

        found_pack(std::move(genome), colour, START_WOLVES, START_PACK_ENERGY, spot_on(land), 5);

        tally(land).refounded++;
    }
}

void World::grow_grass(const Settings& settings)
{
    // Logistic: it grows back fastest at half grown, and grows back from bare ground too, more
    // slowly. Grass changes slowly next to everything else, so it only gets worked out every few
    // steps, for that many steps at once.
    constexpr int EVERY = 5;

    if (++grass_wait < EVERY) {
        return;
    }

    grass_wait = 0;

    const float most = static_cast<float>(settings.grass_max * GRASS_CELL * GRASS_CELL);
    const float rate = static_cast<float>(settings.grass_growth * DT * EVERY);
    const float seed = static_cast<float>(settings.grass_seed) * most;

    std::vector<double> standing(static_cast<std::size_t>(continents()), 0);

    for (std::size_t cell = 0; cell < grass.size(); cell++) {
        const int land = grass_land[cell];

        if (land < 0) {
            continue;
        }

        float& patch = grass[cell];

        patch = std::min(most, std::max(0.0f, patch + rate * (patch + seed) * (1 - patch / most)));

        standing[static_cast<std::size_t>(land)] += patch;
    }

    // Counted for every step it stood for, so that averages come out per step.
    for (int land = 0; land < continents(); land++) {
        tallies[static_cast<std::size_t>(land)].grass += standing[static_cast<std::size_t>(land)] * EVERY;
    }
}

void World::listen_list(const Settings& settings)
{
    wolf_callers.clear();

    for (const Wolf& wolf : wolves) {
        if (*std::max_element(wolf.calls.begin(), wolf.calls.end()) > .01f || wolf.noisy) {
            wolf_callers.push_back({wolf.body.pos, wolf.id, wolf.pack, wolf.calls, wolf.noisy ? static_cast<float>(settings.noise_volume) : 0.0f});
        }
    }

    prey_callers.clear();

    for (const Prey& animal : prey) {
        if (animal.call > .01 || animal.noisy) {
            Caller caller;

            caller.pos = animal.body.pos;
            caller.id = animal.id;
            caller.pack = animal.bison ? 1 : 0;
            caller.volume[0] = static_cast<float>(animal.call);
            caller.noise = animal.noisy ? static_cast<float>(settings.noise_volume) : 0.0f;

            prey_callers.push_back(caller);
        }
    }

    wolf_caller_grid.build(static_cast<int>(wolf_callers.size()), [this](int i) { return wolf_callers[static_cast<std::size_t>(i)].pos; });
    prey_caller_grid.build(static_cast<int>(prey_callers.size()), [this](int i) { return prey_callers[static_cast<std::size_t>(i)].pos; });
}

void World::record(const Settings&)
{
    for (const Wolf& wolf : wolves) {
        Tally& here = tally(wolf.land);

        double calling = 0;

        for (float call : wolf.calls) {
            calling += call / CHANNELS;
        }

        here.wolf_steps++;
        here.wolf_wet_steps += wolf.body.wet ? 1 : 0;
        here.wolf_calls[wolf.saw_bison ? 1 : 0] += calling;
        here.wolf_call_steps[wolf.saw_bison ? 1 : 0]++;
    }

    for (const Prey& animal : prey) {
        Tally& here = tally(animal.land);

        (animal.bison ? here.bison_steps : here.hare_steps)++;
        (animal.bison ? here.bison_wet_steps : here.hare_wet_steps) += animal.body.wet ? 1 : 0;

        auto& calls = animal.bison ? here.bison_calls : here.hare_calls;
        auto& steps = animal.bison ? here.bison_call_steps : here.hare_call_steps;

        calls[animal.saw_wolf ? 1 : 0] += animal.call;
        steps[animal.saw_wolf ? 1 : 0]++;
    }
}

void World::step(const Settings& settings)
{
    time += DT;
    weakest = settings.bison_weakest;

    auto last = std::chrono::steady_clock::now();
    int part = 0;

    const auto lap = [&]() {
        if (!profiling) {
            return;
        }

        const auto now = std::chrono::steady_clock::now();

        profile[static_cast<std::size_t>(part++)] += std::chrono::duration<double>(now - last).count();
        last = now;
    };

    index_everything();
    lap();

    // Everybody looks round and decides what to do, all at once, spread over the threads. This only
    // reads the world, and each animal only writes to itself.
    const int wolf_count = static_cast<int>(wolves.size());
    const int total = wolf_count + static_cast<int>(prey.size());

    next_job.store(0, std::memory_order_relaxed);

    threads.run([&](int) {
        for (int start = next_job.fetch_add(THINK_CHUNK, std::memory_order_relaxed); start < total; start = next_job.fetch_add(THINK_CHUNK, std::memory_order_relaxed)) {
            const int end = std::min(total, start + THINK_CHUNK);

            for (int i = start; i < end; i++) {
                if (i < wolf_count) {
                    think_wolf(wolves[static_cast<std::size_t>(i)], settings);
                } else {
                    think_prey(prey[static_cast<std::size_t>(i - wolf_count)], settings);
                }
            }
        }
    });
    lap();

    record(settings);
    lap();
    move(settings);
    lap();
    index_everything();
    lap();
    fight(settings);
    lap();
    eat(settings);
    lap();
    upkeep(settings);
    lap();
    lifecycle(settings);
    lap();
    rescue(settings);
    lap();
    grow_grass(settings);
    lap();
    listen_list(settings);
    lap();
}

}
