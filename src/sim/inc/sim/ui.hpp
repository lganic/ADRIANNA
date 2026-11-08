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

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace sim {

// The tabs that a project can put things on.
enum class Page { World, Stats };

struct Bounds {
    double min_x = 0;
    double max_x = 0;
    double min_y = 0;
    double max_y = 0;
};

struct GraphOptions {
    std::string title{}; // Goes on the expander that the graph sits under.
    std::string y_label{};
    Page page = Page::Stats;
    std::string x_label = "Simulated hours";
    bool expanded = true;
    std::optional<Bounds> bounds{}; // Leave this out to fit the graph to the data.
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

struct CurveGraphState {
    GraphOptions options;

    std::vector<double> x{};
    std::vector<double> y{};
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

// A whole curve that gets replaced each time, like a snapshot of something happening right now.
class CurveGraph {
public:
    explicit CurveGraph(CurveGraphState& state) : state(&state) {}

    // Only the latest one gets shown, so the place to call this is from Project::draw().
    void set(std::vector<double> x, std::vector<double> y)
    {
        state->x = std::move(x);
        state->y = std::move(y);
    }

private:
    CurveGraphState* state;
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
    CurveGraph add_curve_graph(GraphOptions options);

    Field add_field(FieldOptions options);

    Control add_slider(ControlOptions options);
    Control add_entry(ControlOptions options);

    // The rest of this is for the framework, rather than for projects.

    enum class ItemType { HistoryGraph, CurveGraph, Field, Slider, Entry };

    struct Item {
        ItemType type;
        std::size_t index; // Into the list for that type.
    };

    std::vector<Item> items; // Everything in the order it got added, which is the order it shows up in.

    // Deques, since the handles point in to these, so they can't move around.
    std::deque<HistoryGraphState> history_graphs;
    std::deque<CurveGraphState> curve_graphs;
    std::deque<FieldState> fields;
    std::deque<ControlState> controls;
};

}

#endif // SIM_UI_HPP
