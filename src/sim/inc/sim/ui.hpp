#ifndef SIM_UI_HPP
#define SIM_UI_HPP

// How a project puts things in the GUI. It asks the Ui for what it wants while it is being
// constructed, and keeps the handles that come back:
//
//   score_graph(ui.add_history_graph({"Score", "Average score"})),
//   punishment(ui.add_slider({"Punishment", 0, 20, 5, .5}))
//
//   score_graph.add(score);          // In step()
//   score -= punishment.value();
//
// All the handles are safe to use from the project without thinking about threads.

#include <sim/canvas.hpp>

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace sim {

// The tabs that a project can put things on.
enum class Page { World, Stats };

struct GraphOptions {
    std::string title{}; // Goes on the expander that the graph sits under.
    std::string y_label{};
    Page page = Page::Stats;
    std::string x_label = "Simulated hours";
    bool expanded = true;

    // Runs once on the graph that goes on screen, for anything else that stays the same between
    // frames, like bounds or a legend. Leave this out to fit the graph to the data.
    std::function<void(plotato::Graph&)> setup{};
};

struct FieldOptions {
    std::string label{};
    Page page = Page::World;
    std::string format = "%.3f";
};

struct ControlOptions {
    std::string label{};
    double min = 0;
    double max = 1;
    double value = 0; // Where it starts.
    double step = .01;
    Page page = Page::World;
};

// What sits behind each of the handles. All of this is owned by the Ui.
struct HistoryGraphState {
    GraphOptions options;

    double sum = 0;
    uint64_t count = 0;
    double last = 0;
};

struct LiveGraphState {
    GraphOptions options;

    std::unique_ptr<OffscreenCanvas> buffer{};
};

struct FieldState {
    FieldOptions options;

    double value = 0;
};

struct ControlState {
    ControlState(ControlOptions control_options) : options(std::move(control_options)), value(options.value) {}

    ControlOptions options;

    std::atomic<double> value; // The one thing here that the GUI thread writes to.
};

// A value over the whole run, like a learning curve. Stays a bounded size, however long the run gets.
class HistoryGraph {
public:
    explicit HistoryGraph(HistoryGraphState& state) : state(&state) {}

    // Add a sample. These get averaged into the points on the graph, so this is cheap enough to call every step.
    void add(double value)
    {
        state->sum += value;
        state->count++;
    }

private:
    HistoryGraphState* state;
};

// A graph that gets drawn from scratch every frame, like a snapshot of something happening right now.
class LiveGraph {
public:
    explicit LiveGraph(LiveGraphState& state) : state(&state) {}

    // Only for use from Project::draw(), and with the same rules as the world graph that gets passed in there.
    plotato::Graph& graph() { return *state->buffer; }

private:
    LiveGraphState* state;
};

// A labelled number for the GUI to show.
class Field {
public:
    explicit Field(FieldState& state) : state(&state) {}

    void set(double value) { state->value = value; }

private:
    FieldState* state;
};

// A number that the user can change while the simulation runs.
class Control {
public:
    explicit Control(ControlState& state) : state(&state) {}

    double value() const { return state->value.load(std::memory_order_relaxed); }

private:
    ControlState* state;
};

// Everything that a project has asked for. Only add to this while the project is being
// constructed, since the GUI builds its widgets from here straight afterwards.
class Ui {
public:
    HistoryGraph add_history_graph(GraphOptions options);
    LiveGraph add_live_graph(GraphOptions options);

    Field add_field(FieldOptions options);

    Control add_slider(ControlOptions options);
    Control add_entry(ControlOptions options);

    // The rest of this is for the framework, rather than for projects.

    // Make the graphs that projects draw on. Has to run on the GUI thread, after GTK is up.
    void create_buffers();

    enum class ItemType { HistoryGraph, LiveGraph, Field, Slider, Entry };

    struct Item {
        ItemType type;
        std::size_t index; // Into the list for that type.
    };

    std::vector<Item> items; // Everything in the order it got added, which is the order it shows up in.

    // Deques, since the handles point in to these, so they can't move around.
    std::deque<HistoryGraphState> history_graphs;
    std::deque<LiveGraphState> live_graphs;
    std::deque<FieldState> fields;
    std::deque<ControlState> controls;

    std::unique_ptr<OffscreenCanvas> world_buffer;
};

}

#endif // SIM_UI_HPP
