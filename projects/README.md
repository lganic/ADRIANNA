# Projects

Every folder in here is a task that the GUI gets compiled for. Each one builds to its own executable.

| Project | What it does |
| --- | --- |
| `chase_target` | An organism chasing a target around a circle. The port of `tests/test_3`. |
| `find_food` | An agent with limited vision looking for food. |
| `thruster_chase` | The goal of `chase_target`, but steering with a forward/backward thruster on each leg. |
| `island_food` | The thruster organism going after food across a group of islands, with slow and costly water all around them. |
| `memory_chase` | `chase_target`, but the dot gets angry, and an LSTM has to remember the warning for as long as it lasts. |
| `island_evolution` | `island_food` with 40 islands and a population that lives off of energy, reproduces, and evolves its networks the way CoDeepNEAT does. |

## Compiling

From the repo root, this configures every project, then builds and runs one of them:

```
cmake --preset msys2-ucrt64
cmake --build --preset msys2-ucrt64 --target chase_target
cmake --build --preset msys2-ucrt64 --target find_food
build\find_food.exe
```

`dumb.bat find_food` does all of that for one project. With no argument it runs `chase_target`.

To only configure some projects, use `-DEVODEV_PROJECTS=find_food`. Executables have to run from the
repo root, since they load `static/` from there.

## Adding a project

Make a folder in here with a `CMakeLists.txt`. That is all it takes for it to get picked up.

```cmake
evodev_add_project(my_project
    TITLE "Goes on the window"
    SEED 125
    SOURCES my_project.cpp physics.cpp
    LIBRARIES box2d
)
```

Then implement `sim::Project` from `src/sim/inc/sim/project.hpp`, and define `sim::create_project`:

```cpp
class MyProject : public sim::Project {
public:
    MyProject(sim::Ui& ui, uint64_t seed)
        : score_graph(ui.add_history_graph({"Score", "Average score"})),
          speed(ui.add_slider({"Speed", 0, 10, 5, .1})) {}

    double time_step() const override { return .2; }

    void setup_world(plotato::Graph& graph) override { graph.set_bounds({-2, 2, -2, 2}); }

    void step(double t) override { score_graph.add(speed.value()); }
    void draw(plotato::Graph& world) override { world.circle(0, 0, 1); }

private:
    sim::HistoryGraph score_graph;
    sim::Control speed;
};

std::unique_ptr<sim::Project> sim::create_project(sim::Ui& ui, uint64_t seed)
{
    return std::make_unique<MyProject>(ui, seed);
}
```

A project brings its own physics. `sim::Organism` in `sim/models.hpp` is there for the learning.

## What a project can put in the GUI

All of these come from `sim::Ui`, and can only be added from the project's constructor. They show up
in the order they got added. The options say which page each one goes on (`sim::Page::World` or
`sim::Page::Stats`). See `src/sim/inc/sim/ui.hpp`.

| Call | Handle | Use it for |
| --- | --- | --- |
| `add_history_graph` | `.add(value)` | A value over the whole run. Cheap enough for every step. |
| `add_live_graph` | `.graph()` | A graph that gets drawn from scratch each frame. Use it from `draw()`. |
| `add_field` | `.set(value)` | A number to show. |
| `add_slider` | `.value()` | A number the user can drag. |
| `add_entry` | `.value()` | A number the user can type. |

Graphs always go under an expander. The handles are safe to use without thinking about threads.

## Drawing

Projects draw straight on Plotato graphs, so anything Plotato can draw is available.

| Where | Runs | What goes there |
| --- | --- | --- |
| `draw(world)` and `live_graph.graph()` | Every frame, on a blank graph | Plots and text: `plot`, `scatter`, `polygon`, `circle`, `add_text`, `add_item` |
| `setup_world(graph)` and `GraphOptions::setup` | Once, on the graph that is on screen | What stays the same: `set_bounds`, axes, titles, `legend`, `style` |

Keep to those. `draw` runs on the simulation thread, on a graph that isn't the one on screen, so
other changes to it (or calling its `draw()`) won't do what you'd want.
