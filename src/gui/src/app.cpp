#include <gui/app.hpp>

#include <gtk/gtk.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <memory>
#include <string>
#include <vector>
#include <Plotato/Plotato.hpp>
#include <sim/project.hpp>
#include <sim/sim_thread.hpp>
#include <util/widget_location.hpp>

namespace gui {

// How often the GUI checks for a new frame. Faster than the simulation's frame rate, so none get missed.
constexpr guint FRAME_POLL_MS = 16;

// The status text, fields and history graphs don't need to keep up with the world view.
constexpr gint64 STATS_INTERVAL_US = 250000;

// The least room an open graph gets. Past this a page scrolls, instead of squashing all its graphs.
constexpr int GRAPH_MIN_HEIGHT = 180;

constexpr int PAGE_SPACING = 6;

// A graph, and the expander that it sits under.
struct GraphView {
    GtkExpander* expander = nullptr;
    std::unique_ptr<sim::Canvas> graph;
};

// Everything that the GTK callbacks need. All of this belongs to the GUI thread, other than sim,
// which is the one thing that is safe to talk to from here.
struct App {
    App(sim::Project& project, sim::Ui& ui) : sim(project, ui), ui(ui) {}

    sim::SimThread sim;
    sim::Ui& ui;

    std::unique_ptr<sim::Canvas> world_graph;

    std::vector<GraphView> history_graphs; // In the same order as Ui::history_graphs.
    std::vector<GraphView> live_graphs; // In the same order as Ui::live_graphs.
    std::vector<GtkLabel*> field_labels; // In the same order as Ui::fields.

    bool graphs_opened = false; // Closed history graphs don't get updated, so opening one needs a redraw.

    GtkBox* world_page = nullptr;
    GtkBox* stats_page = nullptr;
    GtkSizeGroup* row_labels = nullptr;

    // Button viewport controls
    GtkButton* up_button;
    GtkButton* down_button;
    GtkButton* left_button;
    GtkButton* right_button;
    GtkButton* zoom_in_button;
    GtkButton* zoom_out_button;

    GtkLabel* status_label = nullptr;
    GtkToggleButton* fast_forward_button = nullptr;

    sim::SimFrame frame;
    sim::SimHistory history;

    gint64 stats_updated_at = 0;
    uint64_t stats_updated_steps = 0;

    guint frame_timer = 0;
};

static void on_window_close(GtkWidget*, gpointer user_data)
{
    // This has to stop here, since the widgets that it draws to are on their way out.
    g_source_remove(static_cast<App*>(user_data)->frame_timer);

    gtk_main_quit();
}

// Put a finished frame on screen. It all lands at once, so that a half drawn one never shows.
static void show_frame(sim::Canvas& canvas, sim::PlotContents& contents)
{
    canvas.swap_contents(contents);
    canvas.draw();
}

static void draw_live_graphs(App& app)
{
    for (std::size_t i = 0; i < app.live_graphs.size() && i < app.frame.live_graphs.size(); i++) {
        if (gtk_expander_get_expanded(app.live_graphs[i].expander)) {
            show_frame(*app.live_graphs[i].graph, app.frame.live_graphs[i]);
        }
    }
}

static void draw_history(App& app)
{
    for (std::size_t i = 0; i < app.history_graphs.size() && i < app.history.values.size(); i++) {

        GraphView& view = app.history_graphs[i];

        if (gtk_expander_get_expanded(view.expander)) {
            view.graph->clear();
            view.graph->plot(app.history.hours, app.history.values[i]);
            view.graph->draw();
        }
    }
}

static void update_status(App& app, gint64 now)
{
    const double elapsed = (now - app.stats_updated_at) / 1e6;
    const double steps_per_second = (app.frame.steps - app.stats_updated_steps) / elapsed;

    char status[128];

    std::snprintf(status, sizeof(status), "%.2f hours    %.0f steps/s", app.frame.t / 3600, steps_per_second);

    gtk_label_set_text(app.status_label, status);

    app.stats_updated_at = now;
    app.stats_updated_steps = app.frame.steps;
}

static void update_fields(App& app)
{
    std::size_t index = 0;

    for (const sim::FieldState& field : app.ui.fields) {

        if (index >= app.frame.fields.size()) {
            return; // Nothing has come through yet.
        }

        char text[64];

        std::snprintf(text, sizeof(text), field.options.format.c_str(), app.frame.fields[index]);

        gtk_label_set_text(app.field_labels[index], text);

        index++;
    }
}

// The GUI side of the simulation. Picks up whatever the simulation thread has published since last time.
static gboolean on_frame_timer(gpointer user_data)
{
    App* app = static_cast<App*>(user_data);

    if (app->sim.take_frame(app->frame)) {
        show_frame(*app->world_graph, app->frame.world);
        draw_live_graphs(*app);
    }

    const gint64 now = g_get_monotonic_time();

    if (now - app->stats_updated_at >= STATS_INTERVAL_US) {

        update_status(*app, now);
        update_fields(*app);

        if (app->sim.take_history(app->history) || app->graphs_opened) {
            draw_history(*app);

            app->graphs_opened = false;
        }
    }

    return G_SOURCE_CONTINUE;
}

static void on_fast_forward_toggled(GtkToggleButton* button, gpointer user_data)
{
    static_cast<App*>(user_data)->sim.set_fast_forward(gtk_toggle_button_get_active(button));
}

static gboolean on_key_press(GtkWidget*, GdkEventKey* event, gpointer user_data)
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

static gboolean on_key_release(GtkWidget*, GdkEventKey* event, gpointer user_data)
{
    if (gdk_keyval_to_lower(event->keyval) != GDK_KEY_v) {
        return FALSE;
    }

    static_cast<App*>(user_data)->sim.set_peek(false);

    return TRUE;
}

// If the window loses focus while V is held, the key release never arrives here.
static gboolean on_focus_out(GtkWidget*, GdkEventFocus*, gpointer user_data)
{
    static_cast<App*>(user_data)->sim.set_peek(false);

    return FALSE;
}

static void on_expander_toggled(GObject*, GParamSpec*, gpointer user_data)
{
    static_cast<App*>(user_data)->graphs_opened = true;
}

static void on_slider_changed(GtkRange* range, gpointer user_data)
{
    static_cast<sim::ControlState*>(user_data)->value = gtk_range_get_value(range);
}

static void on_entry_changed(GtkSpinButton* button, gpointer user_data)
{
    static_cast<sim::ControlState*>(user_data)->value = gtk_spin_button_get_value(button);
}

static GtkBox* page_box(App& app, sim::Page page)
{
    return page == sim::Page::World ? app.world_page : app.stats_page;
}

// A label on the left, with a widget filling the rest of the row.
static void add_row(App& app, sim::Page page, const std::string& text, GtkWidget* widget)
{
    GtkWidget* row = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, PAGE_SPACING);
    GtkWidget* label = gtk_label_new(text.c_str());

    gtk_label_set_xalign(GTK_LABEL(label), 0);
    gtk_size_group_add_widget(app.row_labels, label);

    gtk_box_pack_start(GTK_BOX(row), label, FALSE, FALSE, 0);
    gtk_box_pack_start(GTK_BOX(row), widget, TRUE, TRUE, 0);

    gtk_box_pack_start(page_box(app, page), row, FALSE, TRUE, 0);
}

static GraphView add_graph(App& app, const sim::GraphOptions& options)
{
    GtkWidget* expander = gtk_expander_new(options.title.c_str());
    GtkWidget* area = gtk_drawing_area_new();

    gtk_widget_set_size_request(area, -1, GRAPH_MIN_HEIGHT);
    gtk_widget_set_hexpand(area, TRUE);
    gtk_widget_set_vexpand(area, TRUE);

    gtk_container_add(GTK_CONTAINER(expander), area);

    gtk_expander_set_expanded(GTK_EXPANDER(expander), options.expanded);
    gtk_expander_set_resize_toplevel(GTK_EXPANDER(expander), TRUE);

    // Only take up the spare room while open, so that the graphs still open get it when one closes.
    g_object_bind_property(expander, "expanded", expander, "vexpand", G_BINDING_SYNC_CREATE);

    g_signal_connect(expander, "notify::expanded", G_CALLBACK(on_expander_toggled), &app);

    gtk_box_pack_start(page_box(app, options.page), expander, FALSE, TRUE, 0);

    plotato::FigureStyle style;
    style.default_margin = 25; // The default clips the last label on the bottom axis.

    GraphView view;

    view.expander = GTK_EXPANDER(expander);
    view.graph = std::make_unique<sim::Canvas>(area, style);

    view.graph->add_linear_axis(plotato::BOTTOM);
    view.graph->add_linear_axis(plotato::LEFT);
    view.graph->add_x_title(options.x_label);
    view.graph->add_y_title(options.y_label);

    if (options.setup) {
        options.setup(*view.graph);
    }

    return view;
}

static void add_field(App& app, const sim::FieldOptions& options)
{
    GtkWidget* value = gtk_label_new("-");

    gtk_label_set_xalign(GTK_LABEL(value), 0);

    app.field_labels.push_back(GTK_LABEL(value));

    add_row(app, options.page, options.label, value);
}

// Enough decimal places to show any value that the step can land on.
static int decimal_places(double step)
{
    return std::max(0, static_cast<int>(std::ceil(-std::log10(step))));
}

static void add_slider(App& app, sim::ControlState& control)
{
    const sim::ControlOptions& options = control.options;

    GtkWidget* slider = gtk_scale_new_with_range(GTK_ORIENTATION_HORIZONTAL, options.min, options.max, options.step);

    gtk_scale_set_digits(GTK_SCALE(slider), decimal_places(options.step));
    gtk_scale_set_value_pos(GTK_SCALE(slider), GTK_POS_RIGHT);
    gtk_range_set_value(GTK_RANGE(slider), options.value);

    g_signal_connect(slider, "value-changed", G_CALLBACK(on_slider_changed), &control);

    add_row(app, options.page, options.label, slider);
}

static void add_entry(App& app, sim::ControlState& control)
{
    const sim::ControlOptions& options = control.options;

    GtkWidget* entry = gtk_spin_button_new_with_range(options.min, options.max, options.step);

    gtk_spin_button_set_digits(GTK_SPIN_BUTTON(entry), decimal_places(options.step));
    gtk_spin_button_set_value(GTK_SPIN_BUTTON(entry), options.value);

    g_signal_connect(entry, "value-changed", G_CALLBACK(on_entry_changed), &control);

    add_row(app, options.page, options.label, entry);
}

// Build everything the project asked for, in the order that it asked for it.
static void build_project_widgets(App& app)
{
    for (const sim::Ui::Item& item : app.ui.items) {

        switch (item.type) {
        case sim::Ui::ItemType::HistoryGraph:
            app.history_graphs.push_back(add_graph(app, app.ui.history_graphs[item.index].options));
            break;

        case sim::Ui::ItemType::LiveGraph:
            app.live_graphs.push_back(add_graph(app, app.ui.live_graphs[item.index].options));
            break;

        case sim::Ui::ItemType::Field:
            add_field(app, app.ui.fields[item.index].options);
            break;

        case sim::Ui::ItemType::Slider:
            add_slider(app, app.ui.controls[item.index]);
            break;

        case sim::Ui::ItemType::Entry:
            add_entry(app, app.ui.controls[item.index]);
            break;
        }
    }
}

int run(int argc, char* argv[], const char* title, uint64_t seed)
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

    sim::Ui ui;

    const std::unique_ptr<sim::Project> project = sim::create_project(ui, seed);

    App app(*project, ui);

    app.world_page = find_box(builder, "world-page");
    app.stats_page = find_box(builder, "stats-page");
    app.row_labels = gtk_size_group_new(GTK_SIZE_GROUP_HORIZONTAL);
    app.status_label = find_label(builder, "status-label");
    app.fast_forward_button = GTK_TOGGLE_BUTTON(find_button(builder, "fast-forward-button"));
    app.stats_updated_at = g_get_monotonic_time();

    app.left_button = find_button(builder, "t-left");
    app.right_button = find_button(builder, "t-right");
    app.up_button = find_button(builder, "t-up");
    app.down_button = find_button(builder, "t-down");
    
    app.zoom_in_button = find_button(builder, "z-in");
    app.zoom_out_button = find_button(builder, "z-out");

    build_project_widgets(app);

    g_object_unref(app.row_labels);

    plotato::FigureStyle world_style;
    world_style.dont_draw_version_text = true;

    app.world_graph = std::make_unique<sim::Canvas>(world_area, world_style);

    project->setup_world(*app.world_graph);

    gtk_stack_set_visible_child_name(GTK_STACK(gtk_builder_get_object(builder, "detailstack")), "Stats");

    gtk_window_set_title(GTK_WINDOW(main_window), title);

    g_signal_connect(main_window, "destroy", G_CALLBACK(on_window_close), &app);
    g_signal_connect(main_window, "key-press-event", G_CALLBACK(on_key_press), &app);
    g_signal_connect(main_window, "key-release-event", G_CALLBACK(on_key_release), &app);
    g_signal_connect(main_window, "focus-out-event", G_CALLBACK(on_focus_out), &app);
    g_signal_connect(app.fast_forward_button, "toggled", G_CALLBACK(on_fast_forward_toggled), &app);

    app.sim.start();

    app.frame_timer = g_timeout_add(FRAME_POLL_MS, on_frame_timer, &app);

    gtk_widget_show_all(main_window);
    gtk_main();

    app.sim.stop();

    g_object_unref(builder);
    return 0;
}

}
