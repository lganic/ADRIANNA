#include "continent.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <set>
#include <utility>

namespace hail_mary {

// How far prey see wolves, all the way round, and how long they keep running after they last did.
constexpr double BISON_SIGHT = 30;
constexpr double HARE_SIGHT = 15;
constexpr double BISON_ALERT = 4;
constexpr double HARE_ALERT = 3;

// Bison keep together with the rest of the herd within this far.
constexpr double HERD_RANGE = 30;

// How long a bison faces up to a wolf for, once it turns on one.
constexpr double FIGHT_TIME = 1.5;

// Grazing and ambling speeds.
constexpr double BISON_GRAZE = 1;
constexpr double HARE_GRAZE = 1.5;

// Lifespans, picked evenly between these, in seconds.
constexpr double BISON_LIFE_MIN = 600;
constexpr double BISON_LIFE_MAX = 1200;
constexpr double HARE_LIFE_MIN = 300;
constexpr double HARE_LIFE_MAX = 600;

// Bison and hares only have young once they are this old.
constexpr double BISON_ADULT = 120;
constexpr double HARE_ADULT = 60;

// Grown bison come to the defence of any bison being bitten within this far.
constexpr double DEFENCE_RANGE = 15;

// How long wolves live, if nothing else gets them first, picked evenly between these, in seconds.
constexpr double WOLF_LIFE_MIN = 1200;
constexpr double WOLF_LIFE_MAX = 2400;

// How long a bite stays on the record of a bison, for working out who killed it.
constexpr double BITE_MEMORY = 10;
// Bites this recent count for how far round the bison the wolves were when it died.
constexpr double SURROUND_MEMORY = 3;

constexpr double WOLF_HEALING = 1; // Health per second, while the pack has food.
constexpr double STARVING = .5; // Health lost per second, while it doesn't.
constexpr double HARE_HEALING = 1;

// Energy left in a dead wolf. None, since otherwise the first thing that evolves is packs killing
// and eating each other's wolves, which don't run away, and nothing else ever gets started. Packs
// can still fight, over carcasses and ground.
constexpr double WOLF_CARCASS = 0;

constexpr double PUP_WAIT = 5; // Seconds between pups.

constexpr double INTAKE_MEMORY = 300; // Seconds that Pack::intake averages over.

constexpr int START_PACKS = 8;
constexpr int START_WOLVES = 3;
constexpr double START_ENERGY = 200;
constexpr double REFOUND_WAIT = 10;

constexpr int ARCHIVE_SIZE = 32;

// How far a daughter pack sets up from the middle of its mother's.
constexpr double DISPERSAL_MIN = 40;
constexpr double DISPERSAL_MAX = 100;

constexpr double COLOUR_DRIFT = .05;

// A packmate calling at least this loud, as a wolf hears it, counts for the approach stat. That is
// a full volume call from about 70 m away.
constexpr float LOUD_CALL = .25f;

Stats& operator+=(Stats& a, const Stats& b)
{
    a.bison_kills += b.bison_kills;
    a.weak_kills += b.weak_kills;
    a.hare_kills += b.hare_kills;
    a.old_bison += b.old_bison;
    a.wolves_killed += b.wolves_killed;
    a.wolves_gored += b.wolves_gored;
    a.wolves_starved += b.wolves_starved;
    a.wolves_old += b.wolves_old;
    a.pups += b.pups;
    a.daughters += b.daughters;
    a.refounded += b.refounded;
    a.immigrants += b.immigrants;

    for (int source = 0; source < SOURCES; source++) {
        a.eaten[source] += b.eaten[source];
    }

    a.kill_wolves += b.kill_wolves;
    a.kill_packs += b.kill_packs;
    a.surround += b.surround;
    a.surround_kills += b.surround_kills;
    a.bites += b.bites;
    a.front_bites += b.front_bites;

    for (int context = 0; context < CONTEXTS; context++) {
        for (int channel = 0; channel < CHANNELS; channel++) {
            a.calls[context][channel] += b.calls[context][channel];
        }

        a.context_steps[context] += b.context_steps[context];
    }

    a.approach += b.approach;
    a.approach_steps += b.approach_steps;
    a.packmate_distance += b.packmate_distance;
    a.packmate_steps += b.packmate_steps;
    a.noise_approach += b.noise_approach;
    a.noise_approach_steps += b.noise_approach_steps;

    for (int channel = 0; channel < CHANNELS; channel++) {
        a.channel_approach[channel] += b.channel_approach[channel];
        a.channel_approach_steps[channel] += b.channel_approach_steps[channel];
    }
    a.wolf_steps += b.wolf_steps;
    a.pack_steps += b.pack_steps;
    a.bison_steps += b.bison_steps;
    a.hare_steps += b.hare_steps;

    return a;
}

Stats operator-(Stats a, const Stats& b)
{
    a.bison_kills -= b.bison_kills;
    a.weak_kills -= b.weak_kills;
    a.hare_kills -= b.hare_kills;
    a.old_bison -= b.old_bison;
    a.wolves_killed -= b.wolves_killed;
    a.wolves_gored -= b.wolves_gored;
    a.wolves_starved -= b.wolves_starved;
    a.wolves_old -= b.wolves_old;
    a.pups -= b.pups;
    a.daughters -= b.daughters;
    a.refounded -= b.refounded;
    a.immigrants -= b.immigrants;

    for (int source = 0; source < SOURCES; source++) {
        a.eaten[source] -= b.eaten[source];
    }

    a.kill_wolves -= b.kill_wolves;
    a.kill_packs -= b.kill_packs;
    a.surround -= b.surround;
    a.surround_kills -= b.surround_kills;
    a.bites -= b.bites;
    a.front_bites -= b.front_bites;

    for (int context = 0; context < CONTEXTS; context++) {
        for (int channel = 0; channel < CHANNELS; channel++) {
            a.calls[context][channel] -= b.calls[context][channel];
        }

        a.context_steps[context] -= b.context_steps[context];
    }

    a.approach -= b.approach;
    a.approach_steps -= b.approach_steps;
    a.packmate_distance -= b.packmate_distance;
    a.packmate_steps -= b.packmate_steps;
    a.noise_approach -= b.noise_approach;
    a.noise_approach_steps -= b.noise_approach_steps;

    for (int channel = 0; channel < CHANNELS; channel++) {
        a.channel_approach[channel] -= b.channel_approach[channel];
        a.channel_approach_steps[channel] -= b.channel_approach_steps[channel];
    }
    a.wolf_steps -= b.wolf_steps;
    a.pack_steps -= b.pack_steps;
    a.bison_steps -= b.bison_steps;
    a.hare_steps -= b.hare_steps;

    return a;
}

double mean_call(const Stats& stats, std::initializer_list<Context> contexts, int channel)
{
    double sum = 0;
    uint64_t steps = 0;

    for (Context context : contexts) {
        sum += stats.calls[context][channel];
        steps += stats.context_steps[context];
    }

    return steps > 0 ? sum / static_cast<double>(steps) : 0;
}

double call_signal(const Stats& stats)
{
    double most = 0;

    for (int channel = 0; channel < CHANNELS; channel++) {
        const double bison = mean_call(stats, {AT_BISON, SEES_BISON}, channel);
        const double nothing = mean_call(stats, {ALONE, WITH_PACK}, channel);

        most = std::max(most, std::abs(bison - nothing));
    }

    return most;
}

// How strong a bison is for its age, from .3 to 1: calves start weak and grow in to it, and the old
// fade towards the end. Its most health, how fast it heals, and how much there is of it to eat all
// go with this, and it shows to wolves as how hurt it looks. So a lone wolf can bring down a calf
// or an old bison, which is how real wolves take most of theirs, but not one in its prime.
static double vigour(const Prey& animal)
{
    if (!animal.bison) {
        return 1;
    }

    constexpr double WEAKEST = .4;
    constexpr double OLD = .8; // Of its lifespan, after which it fades.

    const double growing = std::min(1.0, animal.age / BISON_ADULT);
    const double fading = std::max(0.0, (animal.age - OLD * animal.lifespan) / ((1 - OLD) * animal.lifespan));

    return WEAKEST + (1 - WEAKEST) * std::min(growing, 1 - std::min(1.0, fading));
}

// Calves are smaller, and so lighter.
static void grow(Prey& animal)
{
    if (!animal.bison) {
        return;
    }

    const double growing = std::min(1.0, animal.age / BISON_ADULT);
    const double scale = .65 + .35 * growing;

    animal.body.radius = BISON_RADIUS * scale;
    animal.body.mass = BISON_MASS * scale * scale;
}

static double falloff(double distance)
{
    const double scaled = distance / HEAR_SCALE;

    return 1 / (1 + scaled * scaled);
}

// From what reaches the ears, to what the brain gets: logarithmic, the way hearing works, so that a
// faraway call still registers next to a close one.
static float loudness(double heard)
{
    return static_cast<float>(clamp(std::log1p(heard / QUIETEST) / std::log1p(1 / QUIETEST), 0, 1));
}

// Turn towards dir, and slow down for sharp turns, the way an animal does.
static void steer_to(Body& body, Vec dir, double speed)
{
    if (length(dir) < 1e-9) {
        body.want_turn = 0;
        body.want_speed = speed;
        return;
    }

    const double off = wrap_angle(std::atan2(dir.y, dir.x) - body.heading);

    body.want_turn = clamp(off * 5, -body.top_turn, body.top_turn);
    body.want_speed = speed * clamp(std::cos(off), .25, 1);
}

static Vec rotate(Vec v, double angle)
{
    const double c = std::cos(angle);
    const double s = std::sin(angle);

    return {v.x * c - v.y * s, v.x * s + v.y * c};
}

Continent::Continent(int index, uint64_t seed, const Settings& settings)
    : index(index), rng(seed), wolf_grid(HALF_SIZE, 10), prey_grid(HALF_SIZE, 10), carcass_grid(HALF_SIZE, 10), body_grid(HALF_SIZE, 5)
{
    std::uniform_real_distribution<double> uniform(0, 1);
    std::normal_distribution<double> normal(0, 1);

    // Bison in three herds, and hares all over.
    const int herds = 3;
    const int bison = static_cast<int>(settings.bison_wanted);

    for (int herd = 0; herd < herds; herd++) {
        const Vec middle = random_spot();
        const double facing = uniform(rng) * 2 * PI;

        for (int i = herd; i < bison; i += herds) {
            add_prey(true, wrapped(middle + Vec{normal(rng) * 8, normal(rng) * 8}), facing + normal(rng) * .3);
        }
    }

    for (int i = 0; i < static_cast<int>(settings.hares_wanted); i++) {
        add_prey(false, random_spot(), uniform(rng) * 2 * PI);
    }

    // Grown, with their ages spread out, so that they don't all die of old age at once.
    for (Prey& animal : prey) {
        animal.age = (animal.bison ? BISON_ADULT : HARE_ADULT) + uniform(rng) * animal.lifespan * .7;
        animal.health = animal.bison ? BISON_HEALTH * vigour(animal) : animal.health;

        grow(animal);
    }

    for (int i = 0; i < START_PACKS; i++) {
        Colour colour;

        for (double& part : colour) {
            part = uniform(rng);
        }

        found_pack(std::make_shared<const Genome>(settings.founders == 1 ? Genome::ancestor(rng) : Genome::random(rng)), colour, START_WOLVES, START_ENERGY, random_spot(), 5);
    }
}

Vec Continent::random_spot()
{
    std::uniform_real_distribution<double> uniform(-HALF_SIZE, HALF_SIZE);

    const double x = uniform(rng);
    const double y = uniform(rng);

    return {x, y};
}

Colour Continent::drift(const Colour& colour, double amount)
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

Pack& Continent::found_pack(std::shared_ptr<const Genome> genome, const Colour& colour, int count, double energy, Vec around, double spread)
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

        add_wolf(pack, wrapped(around + heading_vector(angle) * distance), uniform(rng) * 2 * PI);
    }

    return pack;
}

void Continent::add_wolf(Pack& pack, Vec pos, double heading)
{
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

    std::uniform_real_distribution<double> uniform(WOLF_LIFE_MIN, WOLF_LIFE_MAX);

    wolf.lifespan = uniform(rng);

    wolves.push_back(std::move(wolf));

    pack.members++;
}

void Continent::add_prey(bool bison, Vec pos, double heading)
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
    animal.health = bison ? BISON_HEALTH : HARE_HEALTH;
    animal.lifespan = bison ? BISON_LIFE_MIN + uniform(rng) * (BISON_LIFE_MAX - BISON_LIFE_MIN) : HARE_LIFE_MIN + uniform(rng) * (HARE_LIFE_MAX - HARE_LIFE_MIN);
    animal.wander = heading;
    animal.health = bison ? BISON_HEALTH * vigour(animal) : HARE_HEALTH;

    grow(animal);

    prey.push_back(std::move(animal));
}

void Continent::archive(const Pack& pack)
{
    // The most recent ones, not the best ever. Keeping the best ever lets a few early packs that got
    // lucky hold on to every place, so that every new pack comes from them, and nothing improves
    // (the same as the record in projects/island_evolution).
    archived.push_back({pack.genome, pack.colour, pack.gathered});

    if (static_cast<int>(archived.size()) > ARCHIVE_SIZE) {
        archived.erase(archived.begin());
    }
}

bool Continent::best(Archived& out) const
{
    const Pack* top = nullptr;

    for (const auto& entry : packs) {
        if (!top || entry.second.gathered > top->gathered) {
            top = &entry.second;
        }
    }

    if (top) {
        out = {top->genome, top->colour, top->gathered};
        return true;
    }

    if (!archived.empty()) {
        out = *std::max_element(archived.begin(), archived.end(), [](const Archived& a, const Archived& b) { return a.gathered < b.gathered; });
        return true;
    }

    return false;
}

void Continent::immigrate(const Archived& migrant, const Settings& settings)
{
    const int count = std::min(2, std::max(1, static_cast<int>(settings.max_pack)));

    if (static_cast<int>(packs.size()) >= MAX_PACKS || static_cast<double>(wolves.size() + count) > settings.max_wolves) {
        return;
    }

    Pack& pack = found_pack(migrant.genome, migrant.colour, count, START_ENERGY, random_spot(), 5);

    pack.immigrant = true;

    stats.immigrants++;
}

int Continent::highest_generation() const
{
    int highest = 0;

    for (const auto& entry : packs) {
        highest = std::max(highest, entry.second.genome->generation);
    }

    return highest;
}

const Wolf* Continent::find_wolf(int id) const
{
    const auto at = std::lower_bound(wolves.begin(), wolves.end(), id, [](const Wolf& wolf, int value) { return wolf.id < value; });

    return at != wolves.end() && at->id == id ? &*at : nullptr;
}

void Continent::index_everything()
{
    wolf_grid.build(static_cast<int>(wolves.size()), [this](int i) { return wolves[i].body.pos; });
    prey_grid.build(static_cast<int>(prey.size()), [this](int i) { return prey[i].body.pos; });
    carcass_grid.build(static_cast<int>(carcasses.size()), [this](int i) { return carcasses[i].pos; });
}

namespace {

// The nearest thing of one kind that a wolf can see.
struct Seen {
    double gap = std::numeric_limits<double>::infinity();
    Vec direction; // In the wolf's own frame: ahead, and to the left.
    int index = -1; // Into whichever list it came from.

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
    Vec from; // Added up, weighted by how loud each one was.
};

}

void Continent::think(Wolf& wolf, const Settings& settings)
{
    // All from the wolf's own point of view, since it can only go the way it is facing.
    //
    // About itself:
    // * How fast it is going ahead, and to the left
    // * How hard it was turning
    // * Its health
    // * How much food the pack has in store, for each wolf in it
    // * How much it got hurt on the last step
    // * Whether a bison, a hare, or another pack's wolf is in reach of a bite, and whether it is eating
    // * A constant
    //
    // What it sees, out to the range of its sight and 135 degrees either side, with nothing behind.
    // For each of the nearest hare, the nearest bison, the nearest carcass, the nearest two
    // packmates, and the nearest wolf from another pack: how close it is, and which way. For the
    // bison, also which way it is facing, from 1 for straight at the wolf to -1 for straight away,
    // and how weak it is (hurt, young, or old). Then how many packmates it can see.
    //
    // That is what chasing and surrounding need. Muro et al. (2011) got wolf pack hunting, chasing
    // and encircling both, out of two rules that only look at the prey and the nearest packmates.
    //
    // Then what it hears. For each channel, from packmates and from strangers apart, since animals
    // know their own pack's voices: how loud it is, all added up, and which way it is coming from,
    // on average. Channel 0 also carries the noise of biting and feeding, which nobody chooses to
    // make, so wolves can come to go towards it because it means food, and a call on channel 0
    // then gets the same response. That is ritualisation: a signal that starts out as a cue.
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

    double chase_gap = std::numeric_limits<double>::infinity();
    double chase_bearing = 0;

    // Where something is, if the wolf can see it: how far off its edge is, and which way it is.
    const auto look = [&](Vec pos, double radius, double& gap, Vec& local) {
        const Vec offset = hail_mary::offset(body.pos, pos);

        gap = std::max(0.0, length(offset) - radius);

        if (gap > sight) {
            return false;
        }

        local = to_body(body.heading, unit(offset));

        return std::abs(std::atan2(local.y, local.x)) <= FIELD_OF_VIEW;
    };

    wolf_grid.visit(body.pos, sight + WOLF_RADIUS, [&](int i) {
        const Wolf& other = wolves[i];

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
        const Prey& animal = prey[i];

        double gap;
        Vec local;

        if (!look(animal.body.pos, animal.body.radius, gap, local)) {
            return;
        }

        (animal.bison ? bison : hare).consider(gap, local, i);

        // For the fixed rule (see Settings::scripted): the nearest prey of the kind it chases.
        const bool chased = animal.bison ? settings.scripted >= 2 : settings.scripted == 1 || settings.scripted == 3;

        if (chased && gap < chase_gap) {
            chase_gap = gap;
            chase_bearing = std::atan2(local.y, local.x);
        }
    });

    carcass_grid.visit(body.pos, sight + BISON_RADIUS, [&](int i) {
        double gap;
        Vec local;

        if (carcasses[i].energy > 0 && look(carcasses[i].pos, carcasses[i].radius, gap, local)) {
            carcass.consider(gap, local, i);
        }
    });

    std::array<Heard, CHANNELS> from_pack;
    std::array<Heard, CHANNELS> from_strangers;

    // Packmates' calls alone, as they would be heard if calls weren't muted, for the approach stat,
    // all together and each channel on its own. And packmates' noise on its own.
    Heard pack_calls;
    std::array<Heard, CHANNELS> pack_channels;
    Heard pack_noise;

    for (const Caller& caller : callers) {
        if (caller.wolf == wolf.id) {
            continue;
        }

        const Vec offset = hail_mary::offset(body.pos, caller.pos);
        const double fall = falloff(length(offset));
        const Vec toward = unit(offset);

        std::array<Heard, CHANNELS>& heard = caller.pack == wolf.pack ? from_pack : from_strangers;

        for (int channel = 0; channel < CHANNELS; channel++) {
            const double called = caller.volume[channel] * fall;
            const double volume = (settings.muted ? 0 : called) + (channel == 0 ? caller.noise * fall : 0);

            heard[channel].loud += volume;
            heard[channel].from += toward * volume;

            if (caller.pack == wolf.pack) {
                pack_calls.loud += called;
                pack_calls.from += toward * called;
                pack_channels[channel].loud += called;
                pack_channels[channel].from += toward * called;

                if (channel == 0) {
                    pack_noise.loud += caller.noise * fall;
                    pack_noise.from += toward * (caller.noise * fall);
                }
            }
        }
    }

    Inputs in;

    std::size_t at = 0;

    const auto put = [&](double value) { in[at++] = static_cast<float>(value); };

    const auto put_seen = [&](const Seen& seen) {
        if (!seen.any()) {
            put(0);
            put(0);
            put(0);
            return;
        }

        put(1 - seen.gap / sight);
        put(seen.direction.x);
        put(seen.direction.y);
    };

    const auto put_sound = [&](const Heard& heard) {
        if (heard.loud <= 1e-12) {
            put(0);
            put(0);
            put(0);
            return;
        }

        const Vec local = limit_length(to_body(body.heading, heard.from * (1 / heard.loud)));

        put(loudness(heard.loud));
        put(local.x);
        put(local.y);
    };

    const Vec velocity = to_body(body.heading, body.vel);

    put(clamp(velocity.x / WOLF_SPEED, -1, 1));
    put(clamp(velocity.y / WOLF_SPEED, -1, 1));
    put(wolf.turn);
    put(clamp(wolf.health / WOLF_HEALTH, 0, 1));
    put(std::tanh(pack.energy / (settings.pup_reserve * std::max(1, pack.members) + 1e-9)));
    put(clamp(wolf.pain, 0, 1));
    put(wolf.at_bison ? 1 : 0);
    put(wolf.at_hare ? 1 : 0);
    put(wolf.at_stranger ? 1 : 0);
    put(wolf.eating ? 1 : 0);
    put(1);

    put_seen(hare);
    put_seen(bison);

    if (bison.any()) {
        const Prey& animal = prey[static_cast<std::size_t>(bison.index)];

        put(dot(heading_vector(animal.body.heading), unit(offset(animal.body.pos, body.pos))));
        put(1 - clamp(animal.health / BISON_HEALTH, 0, 1));
    } else {
        put(0);
        put(0);
    }

    put_seen(carcass);
    put_seen(mate);
    put_seen(second_mate);
    put_seen(stranger);
    put(std::min(1.0, mates_seen / 5.0));

    for (const Heard& heard : from_pack) {
        put_sound(heard);
    }

    for (const Heard& heard : from_strangers) {
        put_sound(heard);
    }

    Outputs out;

    wolf.brain->think(in, out);

    const double speed = clamp(out[OUT_SPEED], -1, 1);

    wolf.body.want_speed = speed >= 0 ? speed * WOLF_SPEED : speed * .3 * WOLF_SPEED;
    wolf.turn = clamp(out[OUT_TURN], -1, 1);
    wolf.body.want_turn = wolf.turn * WOLF_TURN;
    wolf.biting = out[OUT_BITE] > 0;

    for (int channel = 0; channel < CHANNELS; channel++) {
        wolf.calls[channel] = static_cast<float>(clamp(out[OUT_CALL + channel], 0, 1));
    }

    // A fixed rule instead of the brain, to check what the bodies and prey allow: run at the
    // nearest prey in sight, biting once it is in reach, and otherwise amble along, silent.
    if (settings.scripted > 0) {
        const bool chasing = std::isfinite(chase_gap);

        wolf.turn = chasing ? clamp(chase_bearing * 3, -1, 1) : .1;
        wolf.body.want_turn = wolf.turn * WOLF_TURN;
        wolf.body.want_speed = chasing ? WOLF_SPEED : .4 * WOLF_SPEED;
        wolf.biting = wolf.at_bison || wolf.at_hare;
        wolf.calls.fill(0);
    }

    // What it was doing when it called, for working out whether calls mean anything.
    const bool sees_mate = mate.any();
    const bool sees_stranger = stranger.any();
    const bool sees_hare = hare.any();
    const bool sees_bison = bison.any();
    const bool sees_carcass = carcass.any();

    Context context = WITH_PACK;

    if (wolf.at_bison) {
        context = AT_BISON;
    } else if (wolf.eating) {
        context = EATING;
    } else if (sees_bison) {
        context = SEES_BISON;
    } else if (sees_hare) {
        context = SEES_HARE;
    } else if (sees_stranger) {
        context = SEES_STRANGER;
    } else if (!sees_mate) {
        context = ALONE;
    }

    stats.context_steps[context]++;

    for (int channel = 0; channel < CHANNELS; channel++) {
        stats.calls[context][channel] += wolf.calls[channel];
    }

    // Whether it goes towards a packmate's call, when it has nothing else to go on.
    if (!sees_mate && !sees_bison && !sees_hare && !sees_carcass) {
        if (loudness(pack_calls.loud) >= LOUD_CALL) {
            stats.approach += dot(body.vel, unit(pack_calls.from)) / WOLF_SPEED;
            stats.approach_steps++;
        }

        for (int channel = 0; channel < CHANNELS; channel++) {
            if (loudness(pack_channels[channel].loud) >= LOUD_CALL) {
                stats.channel_approach[channel] += dot(body.vel, unit(pack_channels[channel].from)) / WOLF_SPEED;
                stats.channel_approach_steps[channel]++;
            }
        }

        if (loudness(pack_noise.loud) >= LOUD_CALL) {
            stats.noise_approach += dot(body.vel, unit(pack_noise.from)) / WOLF_SPEED;
            stats.noise_approach_steps++;
        }
    }
}

void Continent::steer(Prey& animal, const Settings& settings)
{
    std::uniform_real_distribution<double> uniform(0, 1);
    std::normal_distribution<double> normal(0, 1);

    Body& body = animal.body;

    const double sight = animal.bison ? BISON_SIGHT : HARE_SIGHT;

    Vec threat;
    bool alarmed = false;

    wolf_grid.visit(body.pos, sight + WOLF_RADIUS, [&](int i) {
        const Vec away = offset(wolves[i].body.pos, body.pos);
        const double distance = length(away);

        if (distance > sight) {
            return;
        }

        threat += unit(away) * (1 / std::max(distance, 1.0));
        alarmed = true;
    });

    // Prey can hear wolves calling too, which is what calling costs, besides energy.
    if (settings.prey_hearing > 0) {
        const double threshold = falloff(settings.prey_hearing);

        for (const Caller& caller : callers) {
            double heard = 0;

            const double fall = falloff(length(offset(caller.pos, body.pos)));

            if (!settings.muted) {
                for (float volume : caller.volume) {
                    heard += volume * fall;
                }
            }

            heard += caller.noise * fall;

            if (heard >= threshold) {
                threat += unit(offset(caller.pos, body.pos)) * .05;
                alarmed = true;
            }
        }
    }

    const Wolf* biter = animal.bitten_by >= 0 ? find_wolf(animal.bitten_by) : nullptr;

    if (biter) {
        threat += unit(offset(biter->body.pos, body.pos));
        alarmed = true;
    }

    if (alarmed) {
        animal.alert = animal.bison ? BISON_ALERT : HARE_ALERT;

        if (length(threat) > 1e-9) {
            animal.flee = unit(threat);
        }
    }

    animal.alert -= DT;

    if (animal.bison) {
        // A healthy grown bison runs as fast as a wolf, so one wolf can't just run it down. Calves,
        // old ones, and the wounded are slower, which is why real wolves end up with so many of them.
        body.top_speed = BISON_SPEED * (.6 + .4 * clamp(animal.health / BISON_HEALTH, 0, 1));

        // A bitten bison sometimes turns on whoever bit it, and goes for it with its horns.
        if (biter && animal.fight <= 0 && uniform(rng) < settings.bison_fight * DT) {
            animal.fight = FIGHT_TIME;
            animal.fight_target = biter->id;
        }

        // Grown bison also go for anything biting one of the herd nearby, so a lone wolf can't just
        // pick off a calf. Muskoxen and bison both do this.
        if (!biter && animal.fight <= 0 && animal.age >= BISON_ADULT && settings.herd_defence > 0) {
            int defended = -1;

            prey_grid.visit(body.pos, DEFENCE_RANGE, [&](int i) {
                const Prey& other = prey[i];

                if (defended < 0 && other.bison && other.id != animal.id && other.bitten_by >= 0 && length(offset(body.pos, other.body.pos)) <= DEFENCE_RANGE) {
                    defended = other.bitten_by;
                }
            });

            if (defended >= 0 && find_wolf(defended) && uniform(rng) < settings.herd_defence * DT) {
                animal.fight = FIGHT_TIME;
                animal.fight_target = defended;
            }
        }

        if (animal.fight > 0) {
            animal.fight -= DT;

            if (const Wolf* target = find_wolf(animal.fight_target)) {
                steer_to(body, offset(body.pos, target->body.pos), .7 * BISON_SPEED);
                return;
            }

            animal.fight = 0;
        }

        Vec middle;
        int herd = 0;

        prey_grid.visit(body.pos, HERD_RANGE, [&](int i) {
            const Prey& other = prey[i];

            if (!other.bison || other.id == animal.id || length(offset(body.pos, other.body.pos)) > HERD_RANGE) {
                return;
            }

            middle += offset(body.pos, other.body.pos);
            herd++;
        });

        const Vec together = herd > 0 ? middle * (1.0 / herd / HERD_RANGE) : Vec{};

        if (animal.alert > 0) {
            steer_to(body, animal.flee + together * .4, BISON_SPEED);
        } else {
            animal.wander = wrap_angle(animal.wander + normal(rng) * .5 * std::sqrt(DT));
            steer_to(body, heading_vector(animal.wander) + together, BISON_GRAZE);
        }

        return;
    }

    if (animal.alert > 0) {
        // Hares zigzag when chased.
        animal.jink_wait -= DT;

        if (animal.jink_wait <= 0) {
            animal.jink = (uniform(rng) * 2 - 1) * .8;
            animal.jink_wait = .3 + uniform(rng) * .5;
        }

        steer_to(body, rotate(animal.flee, animal.jink), HARE_SPEED);
    } else {
        animal.wander = wrap_angle(animal.wander + normal(rng) * .8 * std::sqrt(DT));
        steer_to(body, heading_vector(animal.wander), HARE_GRAZE);
    }
}

void Continent::move()
{
    // Every body, wolves and then prey, the same way: turn, then ease towards the velocity it wants
    // along the way it is facing, then move. Animals don't skid sideways, so that eases away too.
    bodies.clear();

    for (Wolf& wolf : wolves) {
        bodies.push_back(&wolf.body);
    }

    for (Prey& animal : prey) {
        bodies.push_back(&animal.body);
    }

    for (Body* body : bodies) {
        body->heading = wrap_angle(body->heading + clamp(body->want_turn, -body->top_turn, body->top_turn) * DT);

        const double speed = clamp(body->want_speed, -.3 * body->top_speed, body->top_speed);
        const Vec wanted = heading_vector(body->heading) * speed;
        const double ease = 1 - std::exp(-DT / body->pickup);

        body->vel += (wanted - body->vel) * ease;
        body->pos = wrapped(body->pos + body->vel * DT);
    }

    // Bodies push each other apart, by mass, and stop running in to each other. So a bison can shove
    // a wolf out of the way, but can't get through a ring of them without a fight.
    const int count = static_cast<int>(bodies.size());

    for (int pass = 0; pass < 2; pass++) {
        body_grid.build(count, [this](int i) { return bodies[i]->pos; });

        for (int i = 0; i < count; i++) {
            Body& a = *bodies[i];

            body_grid.visit(a.pos, a.radius + BISON_RADIUS, [&](int j) {
                if (j <= i) {
                    return;
                }

                Body& b = *bodies[j];

                const Vec offset = hail_mary::offset(a.pos, b.pos);
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

        // Pushing can shove a body over an edge, and the next pass has to find it on the other side.
        for (Body* body : bodies) {
            body->pos = wrapped(body->pos);
        }
    }
}

void Continent::fight(const Settings& settings)
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
    }

    const double damage = settings.bite_damage * DT;

    // Everybody bites at once, so the order makes no difference, but it goes by id anyway.
    for (Wolf& wolf : wolves) {
        const Body& body = wolf.body;

        Wolf* wolf_target = nullptr;
        Prey* prey_target = nullptr;
        double nearest = std::numeric_limits<double>::infinity();

        wolf_grid.visit(body.pos, 2 * WOLF_RADIUS + BITE_REACH, [&](int i) {
            Wolf& other = wolves[i];

            if (other.id == wolf.id || other.pack == wolf.pack) {
                return;
            }

            const Vec offset = hail_mary::offset(body.pos, other.body.pos);
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
            Prey& animal = prey[i];

            const Vec offset = hail_mary::offset(body.pos, animal.body.pos);
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
        animal.bitten_by = wolf.id;

        if (!animal.bison) {
            continue;
        }

        const Vec from = unit(offset(animal.body.pos, body.pos));

        stats.bites++;

        if (std::abs(bearing(animal.body.heading, from)) <= HORN_ANGLE) {
            stats.front_bites++;
        }

        const auto previous = std::find_if(animal.bites.begin(), animal.bites.end(), [&](const Bite& bite) { return bite.wolf == wolf.id; });

        if (previous != animal.bites.end()) {
            previous->time = time;
            previous->from = from;
        } else {
            animal.bites.push_back({wolf.id, wolf.pack, time, from});
        }
    }

    // Horns: anything in front of a bison gets hurt, whether the bison means it or not.
    const double gore = settings.horn_damage * DT;

    for (Prey& animal : prey) {
        if (!animal.bison) {
            animal.health = std::min(HARE_HEALTH, animal.health + HARE_HEALING * DT);
            continue;
        }

        const Body& body = animal.body;

        wolf_grid.visit(body.pos, BISON_RADIUS + WOLF_RADIUS + HORN_REACH, [&](int i) {
            Wolf& wolf = wolves[i];

            const Vec offset = hail_mary::offset(body.pos, wolf.body.pos);
            const double gap = length(offset) - body.radius - wolf.body.radius;

            if (gap > HORN_REACH || std::abs(bearing(body.heading, offset)) > HORN_ANGLE) {
                return;
            }

            wolf.health -= gore;
            wolf.pain += gore;
            wolf.hurt_by = 2;

            wolf.noisy = true;
        });

        const double strength = vigour(animal);

        animal.health = std::min(BISON_HEALTH * strength, animal.health + settings.bison_healing * strength * DT);

        animal.bites.erase(std::remove_if(animal.bites.begin(), animal.bites.end(), [this](const Bite& bite) { return bite.time < time - BITE_MEMORY; }),
                           animal.bites.end());
    }
}

void Continent::eat(const Settings& settings)
{
    for (Wolf& wolf : wolves) {
        Carcass* nearest = nullptr;
        double best_gap = std::numeric_limits<double>::infinity();

        carcass_grid.visit(wolf.body.pos, WOLF_RADIUS + BISON_RADIUS + EAT_REACH, [&](int i) {
            Carcass& carcass = carcasses[i];

            if (carcass.energy <= 0) {
                return;
            }

            const double gap = length(offset(wolf.body.pos, carcass.pos)) - carcass.radius - wolf.body.radius;

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
        stats.eaten[nearest->source] += taken;
        wolf.eating = true;
        wolf.noisy = true;
    }
}

void Continent::upkeep(const Settings& settings)
{
    for (Wolf& wolf : wolves) {
        double calling = 0;

        for (float call : wolf.calls) {
            calling += call;
        }

        const double pace = length(wolf.body.vel) / WOLF_SPEED;
        const double spent = settings.living_energy + settings.running_energy * pace * pace + (wolf.biting ? settings.bite_energy : 0) + settings.call_energy * calling;

        packs.at(wolf.pack).energy -= spent * DT;
    }

    // A pack can only keep so much, which is enough for a daughter, and a full pack's reserve on
    // top. Without a limit, a pack that did well for a few hours could stop hunting for hours.
    const double most = settings.daughter_energy + settings.pup_reserve * std::max(1.0, settings.max_pack);

    for (auto& entry : packs) {
        entry.second.energy = clamp(entry.second.energy, 0, most);
    }

    for (Wolf& wolf : wolves) {
        wolf.starving = packs.at(wolf.pack).energy <= 0;

        if (wolf.starving) {
            wolf.health -= STARVING * DT;
        } else {
            wolf.health = std::min(WOLF_HEALTH, wolf.health + WOLF_HEALING * DT);
        }
    }
}

void Continent::kill(Prey& animal, const Settings& settings)
{
    Carcass carcass;

    carcass.pos = animal.body.pos;
    carcass.radius = animal.body.radius;

    if (!animal.bison) {
        stats.hare_kills++;

        carcass.energy = settings.hare_energy;
        carcass.source = HUNTED_HARE;
        carcasses.push_back(carcass);
        return;
    }

    stats.bison_kills++;

    if (vigour(animal) < .99) {
        stats.weak_kills++;
    }

    carcass.energy = settings.bison_energy * vigour(animal);
    carcass.source = HUNTED_BISON;
    carcasses.push_back(carcass);

    std::set<int> killers;
    std::set<int> killer_packs;

    Vec round;
    int recent = 0;

    for (const Bite& bite : animal.bites) {
        if (bite.time < time - BITE_MEMORY) {
            continue;
        }

        killers.insert(bite.wolf);
        killer_packs.insert(bite.pack);

        if (bite.time >= time - SURROUND_MEMORY) {
            round += bite.from;
            recent++;
        }
    }

    stats.kill_wolves += static_cast<double>(killers.size());
    stats.kill_packs += static_cast<double>(killer_packs.size());

    // How far round the bison the wolves biting it were: the length of the average direction to
    // them, which is 1 if they were all on the same side, and 0 if they were spread evenly round it.
    if (recent >= 2) {
        stats.surround += length(round) / recent;
        stats.surround_kills++;
    }
}

void Continent::lifecycle(const Settings& settings)
{
    std::uniform_real_distribution<double> uniform(0, 1);

    // Prey that were killed, or died of old age.
    for (Prey& animal : prey) {
        animal.age += DT;

        grow(animal);

        if (animal.health <= 0) {
            kill(animal, settings);
        } else if (animal.age >= animal.lifespan) {
            if (animal.bison) {
                stats.old_bison++;

                Carcass carcass;

                carcass.pos = animal.body.pos;
                carcass.radius = animal.body.radius;
                carcass.energy = settings.old_bison_energy;
                carcass.source = OLD_AGE;

                carcasses.push_back(carcass);
            }

            animal.health = 0;
        }
    }

    prey.erase(std::remove_if(prey.begin(), prey.end(), [](const Prey& animal) { return animal.health <= 0; }), prey.end());

    // Wolves that ran out of health, or time.
    for (Wolf& wolf : wolves) {
        wolf.age += DT;

        if (wolf.health > 0 && wolf.age < wolf.lifespan) {
            continue;
        }

        if (wolf.health > 0) {
            stats.wolves_old++;
            wolf.health = 0;
        } else if (wolf.starving) {
            stats.wolves_starved++;
        } else if (wolf.hurt_by == 2) {
            stats.wolves_gored++;
        } else if (wolf.hurt_by == 1) {
            stats.wolves_killed++;
        } else {
            stats.wolves_starved++;
        }

        if (WOLF_CARCASS > 0) {
            Carcass carcass;

            carcass.pos = wolf.body.pos;
            carcass.radius = WOLF_RADIUS;
            carcass.energy = WOLF_CARCASS;
            carcass.source = DEAD_WOLF;

            carcasses.push_back(carcass);
        }

        packs.at(wolf.pack).members--;
    }

    wolves.erase(std::remove_if(wolves.begin(), wolves.end(), [](const Wolf& wolf) { return wolf.health <= 0; }), wolves.end());

    for (auto entry = packs.begin(); entry != packs.end();) {
        if (entry->second.members <= 0) {
            archive(entry->second);
            entry = packs.erase(entry);
        } else {
            ++entry;
        }
    }

    // Packs grow, and then split. When there is only room for some, the ones that have been eating
    // best go first, rather than whichever is first in the list, which would hand every free place
    // to the oldest packs, and stop evolution dead once the continent is full.
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

    std::vector<std::pair<int, Vec>> new_pups; // Pack, and where.

    for (Pack* growing : by_intake) {
        Pack& pack = *growing;

        pack.pup_wait -= DT;

        const double crowd = static_cast<double>(wolves.size() + new_pups.size());

        if (pack.members < most) {
            if (pack.energy >= settings.pup_cost + settings.pup_reserve * pack.members && pack.pup_wait <= 0 && crowd + 1 <= settings.max_wolves) {
                // Next to a packmate picked at random.
                std::vector<const Wolf*> members;

                for (const Wolf& wolf : wolves) {
                    if (wolf.pack == pack.id) {
                        members.push_back(&wolf);
                    }
                }

                const Wolf& parent = *members[static_cast<std::size_t>(uniform(rng) * members.size()) % members.size()];
                const Vec spot = parent.body.pos - heading_vector(parent.body.heading) * (2 * WOLF_RADIUS + .2);

                new_pups.push_back({pack.id, wrapped(spot)});

                pack.energy -= settings.pup_cost;
                pack.pup_wait = PUP_WAIT;
                pack.members++; // Counted now, so that the pack can't have two at once.

                stats.pups++;
            }

            continue;
        }

        // Big enough packs split, and smaller ones (for comparison, see Settings::max_pack) have a
        // new one born, if there is room.
        const bool disperse = most >= 3;

        if (pack.energy < settings.daughter_energy || static_cast<int>(packs.size()) >= MAX_PACKS || (!disperse && crowd + founders > settings.max_wolves)) {
            continue;
        }

        pack.energy -= settings.daughter_energy / 2;
        pack.daughters++;
        stats.daughters++;

        auto genome = std::make_shared<const Genome>(pack.genome->mutated(settings.mutation_chance, rng));
        const Colour colour = drift(pack.colour, COLOUR_DRIFT);

        // Inserting in to a std::map doesn't move anything, so the packs this loop is going through
        // stay where they are. The daughter isn't in by_intake, so it waits until the next step.
        if (disperse) {
            // Full, and well fed: the two youngest leave to start a pack of their own, the way young
            // wolves disperse, with half of what it took, and the mother pack's genes with a few
            // changes, which stand in for mutation and for a mate from somewhere else both. This
            // happens even when the continent is full, so packs keep turning over.
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

            continue;
        }

        // A pair born to start a pack of their own, somewhere else.
        Vec middle;
        Vec first;
        int count = 0;

        for (const Wolf& wolf : wolves) {
            if (wolf.pack == pack.id) {
                if (count == 0) {
                    first = wolf.body.pos;
                }

                middle += offset(first, wolf.body.pos);
                count++;
            }
        }

        middle = first + middle * (1.0 / std::max(1, count));

        const double angle = uniform(rng) * 2 * PI;
        const double distance = DISPERSAL_MIN + uniform(rng) * (DISPERSAL_MAX - DISPERSAL_MIN);
        const Vec spot = wrapped(middle + heading_vector(angle) * distance);

        found_pack(std::move(genome), colour, founders, settings.daughter_energy / 2, spot, 5);
    }

    for (const auto& pup : new_pups) {
        Pack& pack = packs.at(pup.first);

        pack.members--; // add_wolf() counts it again.

        add_wolf(pack, pup.second, uniform(rng) * 2 * PI);
    }

    // Too few packs left: found one from the best that there has been.
    refound_wait -= DT;

    if (static_cast<int>(packs.size()) < MIN_PACKS && refound_wait <= 0 && static_cast<double>(wolves.size() + START_WOLVES) <= settings.max_wolves) {
        std::shared_ptr<const Genome> genome;
        Colour colour;

        if (archived.empty()) {
            genome = std::make_shared<const Genome>(settings.founders == 1 ? Genome::ancestor(rng) : Genome::random(rng));

            for (double& part : colour) {
                part = uniform(rng);
            }
        } else {
            // Whichever gathered the most out of three picked at random, so that the best do best,
            // but not every time.
            std::size_t pick = static_cast<std::size_t>(uniform(rng) * archived.size()) % archived.size();

            for (int round = 1; round < 3; round++) {
                const std::size_t other = static_cast<std::size_t>(uniform(rng) * archived.size()) % archived.size();

                if (archived[other].gathered > archived[pick].gathered) {
                    pick = other;
                }
            }

            genome = std::make_shared<const Genome>(archived[pick].genome->mutated(settings.mutation_chance, rng));
            colour = drift(archived[pick].colour, COLOUR_DRIFT);
        }

        found_pack(std::move(genome), colour, std::min(START_WOLVES, most), START_ENERGY, random_spot(), 5);

        refound_wait = REFOUND_WAIT;
        stats.refounded++;
    }

    // Carcasses rot.
    for (Carcass& carcass : carcasses) {
        carcass.age += DT;
    }

    const double lifetime = settings.carcass_lifetime;

    carcasses.erase(std::remove_if(carcasses.begin(), carcasses.end(), [lifetime](const Carcass& carcass) { return carcass.energy <= 1e-6 || carcass.age >= lifetime; }),
                    carcasses.end());
}

void Continent::breed_prey(const Settings& settings)
{
    // The herds are big, and on the move, so what the wolves take doesn't change how many there are
    // here, the way wolves don't decide how many caribou or wildebeest there are. Whenever there are
    // fewer than the plain holds, more wander in to make up for it, each missing animal on average
    // after the replacement time, grown, and joining a herd that is already here.
    //
    // Without this, wolves eat the prey down to a handful of newborns, which any wolf can catch on
    // its own, and then there are no herds, no grown bison, and nothing to work together on. And
    // with calves born to make up the numbers instead, wolves just farm the calves.
    std::uniform_real_distribution<double> uniform(0, 1);
    std::normal_distribution<double> normal(0, 1);

    for (const bool bison : {true, false}) {
        const double wanted = bison ? settings.bison_wanted : settings.hares_wanted;
        const double replacing = bison ? settings.bison_calving : settings.hare_breeding;

        std::vector<std::size_t> adults;
        int count = 0;

        for (std::size_t i = 0; i < prey.size(); i++) {
            if (prey[i].bison == bison) {
                count++;

                if (prey[i].age >= (bison ? BISON_ADULT : HARE_ADULT)) {
                    adults.push_back(i);
                }
            }
        }

        const int missing = static_cast<int>(wanted) - count;
        const double chance = replacing > 0 ? DT / replacing : 1;

        for (int i = 0; i < missing; i++) {
            if (uniform(rng) >= chance) {
                continue;
            }

            Vec spot = random_spot();
            double heading = uniform(rng) * 2 * PI;

            // Bison join a herd, a few meters from one of its grown members.
            if (bison && !adults.empty()) {
                const Prey& member = prey[adults[static_cast<std::size_t>(uniform(rng) * adults.size()) % adults.size()]];

                spot = wrapped(member.body.pos + Vec{normal(rng) * 4, normal(rng) * 4});
                heading = member.body.heading;
            }

            // Adding to prey can move it, so nothing from it is used after this.
            add_prey(bison, spot, heading);

            Prey& animal = prey.back();

            animal.age = (bison ? BISON_ADULT : HARE_ADULT) + uniform(rng) * .5 * animal.lifespan;
            animal.health = bison ? BISON_HEALTH * vigour(animal) : HARE_HEALTH;

            grow(animal);
        }
    }
}

void Continent::step(const Settings& settings)
{
    time += DT;

    index_everything();

    for (Wolf& wolf : wolves) {
        think(wolf, settings);
    }

    for (Prey& animal : prey) {
        steer(animal, settings);
    }

    move();

    index_everything();

    fight(settings);
    eat(settings);
    upkeep(settings);
    lifecycle(settings);
    breed_prey(settings);

    // What everybody hears on the next step.
    callers.clear();

    for (const Wolf& wolf : wolves) {
        if (*std::max_element(wolf.calls.begin(), wolf.calls.end()) > .01f || wolf.noisy) {
            callers.push_back({wolf.body.pos, wolf.id, wolf.pack, wolf.calls, wolf.noisy ? static_cast<float>(settings.noise_volume) : 0.0f});
        }
    }

    int bison = 0;

    for (const Prey& animal : prey) {
        bison += animal.bison ? 1 : 0;
    }

    // How spread out packs are. Wolves are in id order, and packs are small, so going through each
    // pack's members is cheap enough.
    std::map<int, std::vector<Vec>> members;

    for (const Wolf& wolf : wolves) {
        members[wolf.pack].push_back(wolf.body.pos);
    }

    for (const auto& entry : members) {
        const std::vector<Vec>& spots = entry.second;

        for (std::size_t i = 0; i < spots.size() && spots.size() > 1; i++) {
            double nearest = std::numeric_limits<double>::infinity();

            for (std::size_t j = 0; j < spots.size(); j++) {
                if (j != i) {
                    nearest = std::min(nearest, length(offset(spots[i], spots[j])));
                }
            }

            stats.packmate_distance += nearest;
            stats.packmate_steps++;
        }
    }

    stats.wolf_steps += static_cast<double>(wolves.size());
    stats.pack_steps += static_cast<double>(packs.size());
    stats.bison_steps += bison;
    stats.hare_steps += static_cast<double>(prey.size()) - bison;
}

}
