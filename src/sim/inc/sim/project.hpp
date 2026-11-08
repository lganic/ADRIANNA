#ifndef SIM_PROJECT_HPP
#define SIM_PROJECT_HPP

#include <sim/ui.hpp>

#include <Plotato/Plotato.hpp>

#include <cstdint>
#include <memory>

namespace sim {

// What a project implements to get compiled in to the GUI. See projects/README.md
//
// A project gets constructed on the GUI thread, before the simulation thread starts. From then on
// everything below runs on the simulation thread, so nothing in here needs to be thread safe.
// The one exception is setup_world(), which runs before the simulation thread starts.
class Project {
public:
    virtual ~Project() = default;

    // How many simulated seconds one step covers.
    virtual double time_step() const = 0;

    // Set up the graph that shows the world, with anything that stays the same between frames:
    // bounds, axes, titles, the legend, the style. Leave it alone to fit the graph to whatever got drawn.
    virtual void setup_world(plotato::Graph&) {}

    // Run one step. t is the simulated time at the end of it.
    virtual void step(double t) = 0;

    // Draw the world as it is right now, on a blank graph which goes on screen once this returns.
    // This only runs when the GUI is due a new frame, so it is also the place for anything too
    // slow to do on every step. Only add plots and text here. Anything else on this graph
    // doesn't carry over to the screen, so that goes in setup_world(). Don't call draw() on it.
    virtual void draw(plotato::Graph& world) = 0;
};

// Each project defines this, which is how the GUI gets hold of it.
std::unique_ptr<Project> create_project(Ui& ui, uint64_t seed);

}

#endif // SIM_PROJECT_HPP
