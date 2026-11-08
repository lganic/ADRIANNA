#ifndef SIM_PROJECT_HPP
#define SIM_PROJECT_HPP

#include <sim/scene.hpp>
#include <sim/ui.hpp>

#include <cstdint>
#include <memory>

namespace sim {

// What a project implements to get compiled in to the GUI. See projects/README.md
//
// A project gets constructed on the GUI thread, before the simulation thread starts. From then on
// everything below runs on the simulation thread, so nothing in here needs to be thread safe.
class Project {
public:
    virtual ~Project() = default;

    // How many simulated seconds one step covers.
    virtual double time_step() const = 0;

    // The part of the world that the world view shows.
    virtual Bounds world_bounds() const = 0;

    // Run one step. t is the simulated time at the end of it.
    virtual void step(double t) = 0;

    // Draw the world as it is right now. This only runs when the GUI is due a new frame, so it is
    // also the place for anything that is too slow to do on every step.
    virtual void draw(Scene& scene) = 0;
};

// Each project defines this, which is how the GUI gets hold of it.
std::unique_ptr<Project> create_project(Ui& ui, uint64_t seed);

}

#endif // SIM_PROJECT_HPP
