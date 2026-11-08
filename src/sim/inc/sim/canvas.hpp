#ifndef SIM_CANVAS_HPP
#define SIM_CANVAS_HPP

// How a project's drawing gets from the simulation thread on to the screen.
//
// A graph can't be changed while it is on screen, since a redraw could land halfway through and show
// a half drawn frame. So a project draws on an OffscreenCanvas, which only the simulation thread
// touches. The finished frame then moves across as a PlotContents, and gets swapped on to a
// Canvas by the GUI thread. Nothing gets copied along the way, it is all just pointers changing hands.

#include <Plotato/Plotato.hpp>

#include <memory>
#include <vector>

namespace sim {

// Everything that has been drawn on a graph.
struct PlotContents {
    std::vector<std::unique_ptr<plotato::PlotItem>> plots;
    std::vector<std::unique_ptr<plotato::Text>> text;
};

// A Plotato graph that can have everything on it swapped out in one go.
class Canvas : public plotato::Graph {
public:
    using plotato::Graph::Graph;

    void swap_contents(PlotContents& contents);
};

// A graph which never goes on screen. Has to be made on the GUI thread, but after that adding plots
// and text to it from any one thread is fine, since none of that goes anywhere near GTK.
class OffscreenCanvas : public Canvas {
public:
    OffscreenCanvas();
    ~OffscreenCanvas() override;
};

}

#endif // SIM_CANVAS_HPP
