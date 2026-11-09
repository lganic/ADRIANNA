#include "memory_chase.hpp"

#include <algorithm>
#include <cmath>
#include <memory>
#include <utility>
#include <vector>

namespace memory_chase {

constexpr double PI = 3.14159265358979323846;

// How much of each new value goes in to the averages, which are shown in the fields and decide if
// the angry time grows.
constexpr double RECENT_RATE = .001;

// Radians per second. Constant, unlike chase_target, so that the angry spells are the only thing getting harder.
constexpr double TARGET_SPEED = .5;

// The warnings stay on for this long, and the mood changes the moment they turn off.
constexpr double WARNING_TIME = 1;

constexpr double START_ANGRY_TIME = 3;

// The calm spells are a random length from this up to however long the angry ones are. If they stayed
// short, the organism could get away with only remembering calm spells, and fleeing the rest of the time.
constexpr double MIN_CALM_TIME = 2;

// An angry dot only punishes the organism from up to this far away.
constexpr double ANGER_RANGE = 2;

// How much further from the dot it has to be, late in angry spells compared to late in calm ones,
// to count as remembering the mood.
constexpr double PASS_SEPARATION = 1;

static constexpr int to_steps(double seconds)
{
    return static_cast<int>(seconds / DT + .5);
}

constexpr int WARNING_STEPS = to_steps(WARNING_TIME);

World::World(double target_phase, int steps_left)
    : phys_obj({0, 0}, {0, 0}, BOX_SIZE, ORGANISM_RADIUS),
      target_phase(target_phase),
      spell_steps(steps_left),
      steps_left(steps_left) {}

bool World::warning() const
{
    return steps_left < WARNING_STEPS;
}

MemoryChase::MemoryChase(sim::Ui& ui, uint64_t seed)
    : angry_time(START_ANGRY_TIME),
      org(INPUT_SIZE, OUTPUT_SIZE, NUM_WORLDS, seed),
      rng(seed + 1),
      score_graph(ui.add_history_graph({"Score Graph", "Score"})),
      angry_time_graph(ui.add_history_graph({"How long the dot stays angry", "Seconds"})),
      late_angry_graph(ui.add_history_graph({"Distance late in an angry spell", "Distance"})),
      late_calm_graph(ui.add_history_graph({"Distance late in a calm spell", "Distance"})),
      angry_time_field(ui.add_field({"Angry time (s)"})),
      late_angry_field(ui.add_field({"Distance late in anger"})),
      late_calm_field(ui.add_field({"Distance late in calm"})),
      score_field(ui.add_field({"Recent score"})),
      wall_punish(ui.add_slider({"Wall punishment", 0, 20, 5, .5})),
      effort_punish(ui.add_slider({"Effort punishment", 0, 1, .1, .01})),
      anger_punish(ui.add_slider({"Anger punishment", 0, 10, 1, .1})),
      angry_growth(ui.add_slider({"Angry time growth (s per sim hour)", 0, 20, 1, .1}))
{
    for (int i = 0; i < NUM_WORLDS; i++) {
        // Spread out, so that the worlds aren't all in the same mood as each other.
        worlds.emplace_back(rng.uniform(0, 2 * PI), rng.randint(WARNING_STEPS + 1, to_steps(START_ANGRY_TIME + MIN_CALM_TIME)));
    }
}

double MemoryChase::time_step() const
{
    return DT;
}

void MemoryChase::setup_world(plotato::Graph& graph)
{
    graph.equal_aspect_framing = true;
    graph.set_bounds({-BOX_SIZE * 1.2, BOX_SIZE * 1.2, -BOX_SIZE * 1.2, BOX_SIZE * 1.2});
}

void MemoryChase::update_mood(World& world)
{
    if (world.steps_left > 0) {
        world.steps_left--;
        return;
    }

    world.angry = !world.angry;

    const double seconds = world.angry ? angry_time : rng.uniform(MIN_CALM_TIME, std::max(MIN_CALM_TIME, angry_time));

    world.spell_steps = std::max(WARNING_STEPS + 1, to_steps(seconds));
    world.steps_left = world.spell_steps - 1;
}

// How far off the target the organism gets near the end of a spell, right before the warning. This
// is the furthest it ever has to remember the last warning, so this is what shows if the memory held.
void MemoryChase::track_late_distance(const World& world)
{
    const int memory_steps = world.spell_steps - WARNING_STEPS;

    if (world.warning() || world.steps_left - WARNING_STEPS >= memory_steps / 4) {
        return;
    }

    double& recent = world.angry ? recent_late_angry : recent_late_calm;

    (world.angry ? late_angry_graph : late_calm_graph).add(world.dist);

    recent += RECENT_RATE * (world.dist - recent);
}

void MemoryChase::step(double)
{
    // Only gets harder while the organism is keeping up, so that it stops rising at the limit of what
    // the memory can do. A running total, so that the slider can be moved without the angry time jumping.
    if (recent_late_angry - recent_late_calm > PASS_SEPARATION) {
        angry_time += angry_growth.value() * DT / 3600;
    }

    // Make the input vectors
    // * Top distance raycast
    // * Bottom distance raycast
    // * Left distance raycast
    // * Right distance raycast
    // * Delta X to target
    // * Delta Y to target
    // * X Velocity
    // * Y Velocity
    // * The dot is about to get angry
    // * The dot is about to calm down
    std::vector<float> inputs;
    std::vector<float> scores;

    inputs.reserve(NUM_WORLDS * INPUT_SIZE);
    scores.reserve(NUM_WORLDS);

    for (World& world : worlds) {
        update_mood(world);

        world.target_phase += TARGET_SPEED * DT;

        world.target.x = TARGET_AMPLITUDE * std::cos(world.target_phase);
        world.target.y = TARGET_AMPLITUDE * std::sin(world.target_phase);

        const RaycastDistances distances = world.phys_obj.raycast();

        const double input_vec[INPUT_SIZE] = {
            distances.top / (2 * BOX_SIZE),
            distances.bottom / (2 * BOX_SIZE),
            distances.left / (2 * BOX_SIZE),
            distances.right / (2 * BOX_SIZE),
            (world.target.x - world.phys_obj.pos.x) / BOX_SIZE,
            (world.target.y - world.phys_obj.pos.y) / BOX_SIZE,
            world.phys_obj.vel.x / VEL_SCALE,
            world.phys_obj.vel.y / VEL_SCALE,
            world.warning() && !world.angry ? 1.0 : 0.0,
            world.warning() && world.angry ? 1.0 : 0.0,
        };

        // An organism going very fast would otherwise feed itself bigger and bigger updates.
        for (double input : input_vec) {
            inputs.push_back(static_cast<float>(std::min(1.0, std::max(-1.0, input))));
        }

        scores.push_back(static_cast<float>(world.score));
    }

    const cdn::Tensor actions = org.run(scores, cdn::Tensor({NUM_WORLDS, INPUT_SIZE}, std::move(inputs)));

    double total_score = 0;

    for (int i = 0; i < NUM_WORLDS; i++) {
        World& world = worlds[i];

        std::copy_n(actions.ptr() + i * OUTPUT_SIZE, OUTPUT_SIZE, world.output_vec.begin());

        const std::array<double, OUTPUT_SIZE>& output = world.output_vec;

        const bool hit_wall = world.phys_obj.sim(output[0], output[1], output[2], output[3], DT);

        // Get distance to target after moving, so the reward scores this action (for scoring)
        world.dist = std::hypot(world.target.x - world.phys_obj.pos.x, world.target.y - world.phys_obj.pos.y);

        world.score = 0;

        if (hit_wall) {
            world.score -= wall_punish.value();
        }

        if (world.angry) {
            world.score -= anger_punish.value() * std::max(0.0, ANGER_RANGE - world.dist);
        } else {
            world.score -= world.dist;
        }

        for (double thrust : output) {
            world.score -= std::max(0.0, thrust) * effort_punish.value();
        }

        total_score += world.score;

        track_late_distance(world);
    }

    total_score /= NUM_WORLDS;

    score_graph.add(total_score);
    angry_time_graph.add(angry_time);

    recent_score += RECENT_RATE * (total_score - recent_score);
}

void MemoryChase::draw(plotato::Graph& graph)
{
    constexpr double box = BOX_SIZE;

    const World& world = worlds.front();

    const plotato::Color calm_color(40, 120, 255);
    const plotato::Color angry_color(255, 0, 0);

    plotato::PlotStyle wall_style;
    wall_style.line_color = plotato::Color(0, 0, 255);

    plotato::PlotStyle thrust_style;
    thrust_style.line_color = plotato::Color(255, 0, 0);

    plotato::ShapeStyle organism_style;
    organism_style.fill = plotato::Color(0, 200, 0);
    organism_style.outline = organism_style.fill;

    plotato::ShapeStyle target_style;
    target_style.fill = world.angry ? angry_color : calm_color;
    target_style.outline = target_style.fill;

    const double x = world.phys_obj.pos.x;
    const double y = world.phys_obj.pos.y;

    graph.plot({-box, -box, box, box, -box}, {-box, box, box, -box, -box}, wall_style);

    graph.circle(x, y, ORGANISM_RADIUS, organism_style);

    // The warning is a ring around the dot, in the color that it is about to turn.
    if (world.warning()) {
        plotato::ShapeStyle warning_style;
        warning_style.fill = world.angry ? calm_color : angry_color;
        warning_style.outline = warning_style.fill;

        graph.circle(world.target.x, world.target.y, .18, warning_style);
    }

    graph.circle(world.target.x, world.target.y, .1, target_style);

    // Clamped, so these show what the thrusters really did.
    const double up = std::max(0.0, world.output_vec[0]);
    const double down = std::max(0.0, world.output_vec[1]);
    const double left = std::max(0.0, world.output_vec[2]);
    const double right = std::max(0.0, world.output_vec[3]);

    // Thrusters go on last, so that they show up on top of the organism.
    graph.plot({x, x}, {y, y + up}, thrust_style);
    graph.plot({x, x}, {y, y - down}, thrust_style);
    graph.plot({x, x - left}, {y, y}, thrust_style);
    graph.plot({x, x + right}, {y, y}, thrust_style);

    angry_time_field.set(angry_time);
    late_angry_field.set(recent_late_angry);
    late_calm_field.set(recent_late_calm);
    score_field.set(recent_score);
}

}

std::unique_ptr<sim::Project> sim::create_project(sim::Ui& ui, uint64_t seed)
{
    return std::make_unique<memory_chase::MemoryChase>(ui, seed);
}
