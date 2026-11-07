#include <gtk/gtk.h>
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <string>
#include <Plotato/Plotato.hpp>
#include <sim/sim_thread.hpp>
#include "inc/util/widget_location.hpp"

// How often the GUI checks for a new frame. Faster than the simulation's frame rate, so none get missed.
constexpr guint FRAME_POLL_MS = 16;

// The status text and the learning curve don't need to keep up with the world view.
constexpr gint64 STATS_INTERVAL_US = 250000;

// How much space to leave around the box.
constexpr double WORLD_MARGIN = 1.3;

constexpr uint64_t SEED = 125;

// Everything that the GTK callbacks need. All of this belongs to the GUI thread, other than sim,
// which is the one thing that is safe to talk to from here.
struct App {
    sim::SimThread sim;

    plotato::Graph* world_graph = nullptr;
    plotato::Graph* stats_graph = nullptr;

    GtkLabel* status_label = nullptr;
    GtkToggleButton* fast_forward_button = nullptr;

    sim::SimFrame frame;
    sim::SimHistory history;

    gint64 stats_updated_at = 0;
    uint64_t stats_updated_steps = 0;
};

void on_window_close(GtkWidget* widget, gpointer user_data)
{
    gtk_main_quit();
}

static void draw_world(plotato::Graph& graph, const sim::SimFrame& frame)
{
    constexpr double box = sim::BOX_SIZE;

    const double x = frame.organism.x;
    const double y = frame.organism.y;

    plotato::PlotStyle wall_style;
    wall_style.line_color = plotato::Color(0, 0, 255);

    plotato::PlotStyle thrust_style;
    thrust_style.line_color = plotato::Color(255, 0, 0);

    plotato::ShapeStyle organism_style;
    organism_style.fill = plotato::Color(0, 200, 0);
    organism_style.outline = organism_style.fill;

    plotato::ShapeStyle target_style;
    target_style.fill = plotato::Color(255, 0, 0);
    target_style.outline = target_style.fill;

    graph.clear();

    graph.plot({-box, -box, box, box, -box}, {-box, box, box, -box, -box}, wall_style);

    graph.circle(x, y, .2, organism_style);
    graph.circle(frame.target.x, frame.target.y, .1, target_style);

    // Thrusters go on last, so that they show up on top of the organism.
    graph.plot({x, x}, {y, y + frame.thrust[0]}, thrust_style);
    graph.plot({x, x}, {y, y - frame.thrust[1]}, thrust_style);
    graph.plot({x, x - frame.thrust[2]}, {y, y}, thrust_style);
    graph.plot({x, x + frame.thrust[3]}, {y, y}, thrust_style);

    graph.draw();
}

static void draw_stats(plotato::Graph& graph, const sim::SimHistory& history)
{
    graph.clear();
    graph.plot(history.hours, history.avg_dist);
    graph.draw();
}

static void update_status(App& app, gint64 now)
{
    const double elapsed = (now - app.stats_updated_at) / 1e6;
    const double steps_per_second = (app.frame.steps - app.stats_updated_steps) / elapsed;

    char status[128];

    std::snprintf(status, sizeof(status), "%.2f hours    avg dist %.3f    %.0f steps/s",
                  app.frame.t / 3600, app.frame.avg_dist, steps_per_second);

    gtk_label_set_text(app.status_label, status);

    app.stats_updated_at = now;
    app.stats_updated_steps = app.frame.steps;
}

// The GUI side of the simulation. Picks up whatever the simulation thread has published since last time.
static gboolean on_frame_timer(gpointer user_data)
{
    App* app = static_cast<App*>(user_data);

    if (app->sim.take_frame(app->frame)) {
        draw_world(*app->world_graph, app->frame);
    }

    const gint64 now = g_get_monotonic_time();

    if (now - app->stats_updated_at >= STATS_INTERVAL_US) {

        update_status(*app, now);

        if (app->sim.take_history(app->history)) {
            draw_stats(*app->stats_graph, app->history);
        }
    }

    return G_SOURCE_CONTINUE;
}

static void on_fast_forward_toggled(GtkToggleButton* button, gpointer user_data)
{
    static_cast<App*>(user_data)->sim.set_fast_forward(gtk_toggle_button_get_active(button));
}

static gboolean on_key_press(GtkWidget* widget, GdkEventKey* event, gpointer user_data)
{
    App* app = static_cast<App*>(user_data);

    switch (gdk_keyval_to_lower(event->keyval)) {
    case GDK_KEY_f:
        gtk_toggle_button_set_active(app->fast_forward_button, TRUE);
        return TRUE;

    case GDK_KEY_s:
        gtk_toggle_button_set_active(app->fast_forward_button, FALSE);
        return TRUE;

    case GDK_KEY_v:
        app->sim.set_peek(true);
        return TRUE;

    default:
        return FALSE;
    }
}

static gboolean on_key_release(GtkWidget* widget, GdkEventKey* event, gpointer user_data)
{
    if (gdk_keyval_to_lower(event->keyval) != GDK_KEY_v) {
        return FALSE;
    }

    static_cast<App*>(user_data)->sim.set_peek(false);

    return TRUE;
}

// If the window loses focus while V is held, the key release never arrives here.
static gboolean on_focus_out(GtkWidget* widget, GdkEventFocus* event, gpointer user_data)
{
    static_cast<App*>(user_data)->sim.set_peek(false);

    return FALSE;
}

int main(int argc, char* argv[])
{
    gtk_init(&argc, &argv);

    // Override StackSwitcher button sizing
    GtkCssProvider* css = gtk_css_provider_new();

    gtk_css_provider_load_from_path(
        css,
        "static/style.css",
        nullptr
    );

    gtk_style_context_add_provider_for_screen(
        gdk_screen_get_default(),
        GTK_STYLE_PROVIDER(css),
        GTK_STYLE_PROVIDER_PRIORITY_APPLICATION
    );

    g_object_unref(css);

    GtkBuilder* builder = gtk_builder_new();
    if (!gtk_builder_add_from_file(builder, "static/gui.glade", NULL)) {
        g_error("Failed to load UI file");
        return 1;
    }

    GtkWidget* main_window = GTK_WIDGET(gtk_builder_get_object(builder, "main_window"));

    GtkWidget* world_area = GTK_WIDGET(find_drawing_area(builder, "graph-area"));
    GtkWidget* stats_area = GTK_WIDGET(find_drawing_area(builder, "stats-area"));

    plotato::FigureStyle world_style;
    world_style.dont_draw_version_text = true;

    plotato::FigureStyle stats_style;
    stats_style.default_margin = 25; // The default clips the last label on the bottom axis.

    plotato::Graph world_graph(world_area, world_style);
    plotato::Graph stats_graph(stats_area, stats_style);

    plotato::GraphBounds bounds{-sim::BOX_SIZE * 1.2, sim::BOX_SIZE * 1.2, -sim::BOX_SIZE * 1.2, sim::BOX_SIZE * 1.2};

    world_graph.equal_aspect_framing = true;
    world_graph.set_bounds(bounds);

    stats_graph.add_linear_axis(plotato::BOTTOM);
    stats_graph.add_linear_axis(plotato::LEFT);
    stats_graph.add_x_title("Simulated hours");
    stats_graph.add_y_title("Average distance to target");

    App app;

    app.world_graph = &world_graph;
    app.stats_graph = &stats_graph;
    app.status_label = find_label(builder, "status-label");
    app.fast_forward_button = GTK_TOGGLE_BUTTON(find_button(builder, "fast-forward-button"));
    app.stats_updated_at = g_get_monotonic_time();

    gtk_stack_set_visible_child_name(GTK_STACK(gtk_builder_get_object(builder, "detailstack")), "Stats");

    gtk_window_set_title(GTK_WINDOW(main_window), "EVO-Dev");

    g_signal_connect(main_window, "destroy", G_CALLBACK(on_window_close), nullptr);
    g_signal_connect(main_window, "key-press-event", G_CALLBACK(on_key_press), &app);
    g_signal_connect(main_window, "key-release-event", G_CALLBACK(on_key_release), &app);
    g_signal_connect(main_window, "focus-out-event", G_CALLBACK(on_focus_out), &app);
    g_signal_connect(app.fast_forward_button, "toggled", G_CALLBACK(on_fast_forward_toggled), &app);

    app.sim.start(SEED);

    const guint frame_timer = g_timeout_add(FRAME_POLL_MS, on_frame_timer, &app);

    gtk_widget_show_all(main_window);
    gtk_main();

    g_source_remove(frame_timer);
    app.sim.stop();

    g_object_unref(builder);
    return 0;
}
