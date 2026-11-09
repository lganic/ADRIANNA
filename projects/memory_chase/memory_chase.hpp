#ifndef MEMORY_CHASE_HPP
#define MEMORY_CHASE_HPP

#include "memory_organism.hpp"
#include "physics.hpp"

#include <sim/project.hpp>

#include <array>
#include <cstdint>
#include <deque>

namespace memory_chase {

constexpr double BOX_SIZE = 2;
constexpr double ORGANISM_RADIUS = .2;
constexpr double TARGET_AMPLITUDE = 1;

constexpr double DT = 0.2;

constexpr double VEL_SCALE = 5;

constexpr int INPUT_SIZE = 10;
constexpr int OUTPUT_SIZE = 4;

// How many copies of the world get simulated. All of them train the one organism, and the first one
// is what goes on screen. In a single world, every update would come from whichever mood the dot
// happens to be in, which drags the network back and forth instead of teaching it the difference.
constexpr int NUM_WORLDS = 16;

// One world, and where its dot is in its mood swings.
struct World {
    World(double target_phase, int steps_left);

    PhysicsObject phys_obj;

    double target_phase;
    Point target;

    bool angry = false;
    int spell_steps; // How long the mood lasts, from one change to the next.
    int steps_left; // How many steps are left after this one, before the mood changes.

    std::array<double, OUTPUT_SIZE> output_vec{}; // What the organism asked for on the last step. Up, down, left, right.
    double dist = 0; // Distance to the target after the last step.
    double score = 0;

    // Whether the warning that the mood is about to change is on.
    bool warning() const;
};

// An organism chasing a dot, except that the dot sometimes gets angry, and then being close to it is
// what gets punished. It can't see the mood of the dot. All it gets is a warning for the second
// before each change, so it has to remember the last warning for as long as the mood lasts.
// The angry spells slowly get longer, to find out how long it can keep that up.
class MemoryChase : public sim::Project {
public:
    MemoryChase(sim::Ui& ui, uint64_t seed);

    double time_step() const override;
    void setup_world(plotato::Graph& graph) override;

    void step(double t) override;
    void draw(plotato::Graph& world) override;

    std::deque<World> worlds; // A deque, since physics objects can't move.

    double angry_time; // How long the angry spells last at the moment, in seconds.

    MemoryOrganism org;

private:
    void update_mood(World& world);
    void track_late_distance(const World& world);

    cdn::Rng rng;

    sim::HistoryGraph score_graph;
    sim::HistoryGraph angry_time_graph;
    sim::HistoryGraph late_angry_graph;
    sim::HistoryGraph late_calm_graph;

    sim::Field angry_time_field;
    sim::Field late_angry_field;
    sim::Field late_calm_field;
    sim::Field score_field;

    sim::Control wall_punish;
    sim::Control effort_punish;
    sim::Control anger_punish;
    sim::Control angry_growth;

    double recent_late_angry = 0;
    double recent_late_calm = 0;
    double recent_score = 0;
};

}

#endif // MEMORY_CHASE_HPP
