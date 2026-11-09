#include <gui/app.hpp>

#include <gtk/gtk.h>
#include <gdk/gdkkeysyms.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <exception>
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

// The size that a graph's export menu starts on.
constexpr int EXPORT_DEFAULT_WIDTH = 1200;
constexpr int EXPORT_DEFAULT_HEIGHT = 800;

constexpr int EXPORT_MIN_SIZE = 200; // Any smaller and the axes don't leave any room for the graph.
constexpr int EXPORT_MAX_SIZE = 16384; // Already about a gigabyte of image while it is being drawn.

// The menu for saving a graph to an image, and what it needs to do that.
struct GraphExportMenu {
    sim::Canvas* graph = nullptr;
    std::string title;

    GtkPopover* popover = nullptr;
    GtkSpinButton* width = nullptr;
    GtkSpinButton* height = nullptr;
};

// A graph, and the expander that it sits under.
struct GraphView {
    GtkExpander* expander = nullptr;
    std::unique_ptr<sim::Canvas> graph;
    std::unique_ptr<GraphExportMenu> export_menu;
};

enum KeyDirection {
    UP,
    DOWN,
    LEFT,
    RIGHT,
    ZOOMIN,
    ZOOMOUT
};

class ViewboxController {

    public:
        ViewboxController() = default;
        ViewboxController(plotato::GraphBounds initial_bounds): bounds(initial_bounds) {}

        void set_input(KeyDirection dir, bool pressed, bool keyboard) {
            if (keyboard)
                keys[dir] = pressed;
            else
                mouse[dir] = pressed;

            // Make keyboard input visually press the GTK button. (Purely decorative)
            if (keyboard) {
                if (pressed)
                    gtk_widget_set_state_flags(
                        buttons[dir], GTK_STATE_FLAG_ACTIVE, FALSE);
                else
                    gtk_widget_unset_state_flags(
                        buttons[dir], GTK_STATE_FLAG_ACTIVE);
            }
        }

        static gboolean on_timeout_static(gpointer user_data) {
            ViewboxController* self = static_cast<ViewboxController*>(user_data);
            return self->update_movement(); // Delegate to instance method
        }

        plotato::GraphBounds bounds;
        GtkWidget* buttons[6]; // Tracks each button.
    private:

        bool keys[6] = {}; // Key pressed for object.
        bool mouse[6] = {}; // Mouse pressed for object (via gui button)

        gint64 last_update = 0;

        double move_speed = 0.5;
        double zoom_speed = 0.5;

        bool active(KeyDirection dir) const {
            return keys[dir] || mouse[dir];
        }

        void translate(double x, double y, double z, double dt) {

            // Figure out the current ranges.
            double x_range = bounds.xmax - bounds.xmin;
            double y_range = bounds.ymax - bounds.ymin;

            double offset;

            if (x != 0) { // Nothing should happen at zero. Check if zero for optimization / stability purposes.
                // Apply X update.
                offset = x * x_range * move_speed * dt;
                bounds.xmin += offset;
                bounds.xmax += offset;
            }
            
            if (y != 0) { // Nothing should happen at zero. Check if zero for optimization / stability purposes.
                // Apply Y update.
                offset = y * y_range * move_speed * dt;
                bounds.ymin += offset;
                bounds.ymax += offset;
            }

            if (z != 0) { // Nothing should happen at zero. Check if zero for optimization / stability purposes.
                // Apply Zoom update
                std::pair<double, double> center = bounds.get_center();
                // Expand around center
                double x_inc_amount = z * x_range * zoom_speed * dt;
                double y_inc_amount = z * y_range * zoom_speed * dt;
        
                bounds.xmin = center.first - (x_range / 2) + x_inc_amount;
                bounds.xmax = center.first + (x_range / 2) - x_inc_amount;

                bounds.ymin = center.second - (y_range / 2) + y_inc_amount;
                bounds.ymax = center.second + (y_range / 2) - y_inc_amount;
            }
        }

        // Called approximately 60 times per second.
        gboolean update_movement() {

            gint64 now = g_get_monotonic_time();

            double dt = last_update == 0 ? 0.0 : (now - last_update) / 1000000.0;
            last_update = now;

            double x_del = 0;
            double y_del = 0;
            double z_del = 0;

            if (active(UP)) y_del += 1;
            if (active(DOWN)) y_del -= 1;
            if (active(LEFT)) x_del -= 1;
            if (active(RIGHT)) x_del += 1;
            if (active(ZOOMIN)) z_del += 1;
            if (active(ZOOMOUT)) z_del -= 1;

            translate(x_del, y_del, z_del, dt);

            return G_SOURCE_CONTINUE;
        }
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

    ViewboxController viewbox_controller;

    gint64 stats_updated_at = 0;
    uint64_t stats_updated_steps = 0;

    guint movement_timer = 0;
    guint frame_timer = 0;
};

static void on_window_close(GtkWidget*, gpointer user_data)
{

    App* app = static_cast<App*>(user_data);

    // This has to stop here, since the widgets that it draws to are on their way out.
    if (app->frame_timer)
        g_source_remove(app->frame_timer);

    if (app->movement_timer)
        g_source_remove(app->movement_timer);

    gtk_main_quit();
}

// Mouse input
struct ButtonBinding {
    ViewboxController* controller;
    KeyDirection direction;
};

gboolean on_translate_button_press(GtkWidget*, GdkEventButton* event,
                        gpointer data) {
    if (event->button != 1) return FALSE;

    auto* b = static_cast<ButtonBinding*>(data);
    b->controller->set_input(b->direction, true, false);
    return FALSE;
}

gboolean on_translate_button_release(GtkWidget*, GdkEventButton* event,
                          gpointer data) {
    if (event->button != 1) return FALSE;

    auto* b = static_cast<ButtonBinding*>(data);
    b->controller->set_input(b->direction, false, false);
    return FALSE;
}

// Prevent stuck movement if the window loses focus.
// gboolean on_focus_out(GtkWidget*, GdkEventFocus*,
//                       gpointer data) {
//     auto* c = static_cast<ViewboxController*>(data);

//     for (int i = 0; i < 6; ++i) {
//         c->set_input(static_cast<KeyDirection>(i), false, true);
//         c->set_input(static_cast<KeyDirection>(i), false, false);
//     }
//     return FALSE;
// }

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
        app->world_graph->set_bounds(app->viewbox_controller.bounds);
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

    case GDK_KEY_Up:
        app->viewbox_controller.set_input(UP, true, true);
        return TRUE;

    case GDK_KEY_Down:
        app->viewbox_controller.set_input(DOWN, true, true);
        return TRUE;

    case GDK_KEY_Left:
        app->viewbox_controller.set_input(LEFT, true, true);
        return TRUE;

    case GDK_KEY_Right:
        app->viewbox_controller.set_input(RIGHT, true, true);
        return TRUE;

    case GDK_KEY_plus:
        app->viewbox_controller.set_input(ZOOMIN, true, true);
        return TRUE;

    case GDK_KEY_minus:
        app->viewbox_controller.set_input(ZOOMOUT, true, true);
        return TRUE;

    default:
        return FALSE;
    }
}

static gboolean on_key_release(GtkWidget*, GdkEventKey* event, gpointer user_data)
{

    App* app = static_cast<App*>(user_data);

    switch (gdk_keyval_to_lower(event->keyval)) {

    case GDK_KEY_v:
        app->sim.set_peek(false);
        return TRUE;

    case GDK_KEY_Up:
        app->viewbox_controller.set_input(UP, false, true);
        return TRUE;

    case GDK_KEY_Down:
        app->viewbox_controller.set_input(DOWN, false, true);
        return TRUE;

    case GDK_KEY_Left:
        app->viewbox_controller.set_input(LEFT, false, true);
        return TRUE;

    case GDK_KEY_Right:
        app->viewbox_controller.set_input(RIGHT, false, true);
        return TRUE;

    case GDK_KEY_plus:
    case GDK_KEY_equal:
    case GDK_KEY_KP_Add:
        app->viewbox_controller.set_input(ZOOMIN, false, true);
        return TRUE;

    case GDK_KEY_minus:
    case GDK_KEY_KP_Subtract:
        app->viewbox_controller.set_input(ZOOMOUT, false, true);
        return TRUE;

    default:
        return FALSE;
    }
}

static gboolean on_focus_out(GtkWidget*, GdkEventFocus*, gpointer user_data)
{
    App* app = static_cast<App*>(user_data);

    app->sim.set_peek(false);

    // Prevent held input while focus is lost
    for (int i = 0; i < 6; ++i) {
        app->viewbox_controller.set_input(static_cast<KeyDirection>(i), false, true);
        app->viewbox_controller.set_input(static_cast<KeyDirection>(i), false, false);
    }
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

// The graph's title, with anything that can't go in a file name taken out.
static std::string export_file_name(const std::string& title)
{
    std::string name = title.empty() ? "graph" : title;

    for (std::size_t i = name.find_first_of("<>:\"/\\|?*"); i != std::string::npos; i = name.find_first_of("<>:\"/\\|?*", i)) {
        name[i] = '_';
    }

    return name + ".png";
}

// What gets written is a PNG no matter what the file is called, so make sure that the name says so.
static std::string with_png_extension(std::string path)
{
    if (path.size() < 4 || g_ascii_strcasecmp(path.c_str() + path.size() - 4, ".png") != 0) {
        path += ".png";
    }

    return path;
}

static void export_graph(GtkWindow* window, sim::Canvas& graph, const std::string& path, int width, int height)
{
    try {
        graph.export_image(path, width, height);
    }
    catch (const std::exception& error) {
        GtkWidget* dialog = gtk_message_dialog_new(
            window, GTK_DIALOG_MODAL, GTK_MESSAGE_ERROR, GTK_BUTTONS_CLOSE,
            "Couldn't save %s: %s", path.c_str(), error.what()
        );

        gtk_dialog_run(GTK_DIALOG(dialog));
        gtk_widget_destroy(dialog);
    }
}

static void on_export_save_clicked(GtkWidget* button, gpointer user_data)
{
    GraphExportMenu* menu = static_cast<GraphExportMenu*>(user_data);

    // A number that has been typed in doesn't count until it is committed.
    gtk_spin_button_update(menu->width);
    gtk_spin_button_update(menu->height);

    GtkWindow* window = GTK_WINDOW(gtk_widget_get_toplevel(button));

    gtk_popover_popdown(menu->popover);

    GtkFileChooserNative* native = gtk_file_chooser_native_new("Export graph", window, GTK_FILE_CHOOSER_ACTION_SAVE, "_Save", "_Cancel");
    GtkFileChooser* chooser = GTK_FILE_CHOOSER(native);

    GtkFileFilter* filter = gtk_file_filter_new();

    gtk_file_filter_set_name(filter, "PNG image");
    gtk_file_filter_add_pattern(filter, "*.png");

    gtk_file_chooser_add_filter(chooser, filter);
    gtk_file_chooser_set_do_overwrite_confirmation(chooser, TRUE);
    gtk_file_chooser_set_current_name(chooser, export_file_name(menu->title).c_str());

    if (gtk_native_dialog_run(GTK_NATIVE_DIALOG(native)) == GTK_RESPONSE_ACCEPT) {

        gchar* path = gtk_file_chooser_get_filename(chooser);

        export_graph(
            window, *menu->graph, with_png_extension(path),
            gtk_spin_button_get_value_as_int(menu->width),
            gtk_spin_button_get_value_as_int(menu->height)
        );

        g_free(path);
    }

    g_object_unref(native);
}

static GtkSpinButton* make_export_size_entry(int value)
{
    GtkWidget* entry = gtk_spin_button_new_with_range(EXPORT_MIN_SIZE, EXPORT_MAX_SIZE, 1);

    gtk_spin_button_set_numeric(GTK_SPIN_BUTTON(entry), TRUE);
    gtk_spin_button_set_value(GTK_SPIN_BUTTON(entry), value);

    return GTK_SPIN_BUTTON(entry);
}

static void attach_export_row(GtkGrid* grid, int row, const char* text, GtkSpinButton* entry)
{
    GtkWidget* label = gtk_label_new(text);

    gtk_label_set_xalign(GTK_LABEL(label), 0);

    gtk_grid_attach(grid, label, 0, row, 1, 1);
    gtk_grid_attach(grid, GTK_WIDGET(entry), 1, row, 1, 1);
}

// Put a button in the corner of a graph, for opening the menu that saves it to an image.
static std::unique_ptr<GraphExportMenu> add_export_menu(GtkOverlay* overlay, sim::Canvas& graph, const std::string& title)
{
    auto menu = std::make_unique<GraphExportMenu>();

    menu->graph = &graph;
    menu->title = title;
    menu->width = make_export_size_entry(EXPORT_DEFAULT_WIDTH);
    menu->height = make_export_size_entry(EXPORT_DEFAULT_HEIGHT);

    GtkWidget* save_button = gtk_button_new_with_label("Save to file...");

    g_signal_connect(save_button, "clicked", G_CALLBACK(on_export_save_clicked), menu.get());

    GtkWidget* grid = gtk_grid_new();

    gtk_grid_set_row_spacing(GTK_GRID(grid), PAGE_SPACING);
    gtk_grid_set_column_spacing(GTK_GRID(grid), PAGE_SPACING);
    gtk_container_set_border_width(GTK_CONTAINER(grid), PAGE_SPACING);

    attach_export_row(GTK_GRID(grid), 0, "Width", menu->width);
    attach_export_row(GTK_GRID(grid), 1, "Height", menu->height);
    gtk_grid_attach(GTK_GRID(grid), save_button, 0, 2, 2, 1);

    gtk_widget_show_all(grid); // Popovers aren't reached by the show_all on the window.

    GtkWidget* button = gtk_menu_button_new();

    menu->popover = GTK_POPOVER(gtk_popover_new(button));

    gtk_container_add(GTK_CONTAINER(menu->popover), grid);

    // Left to itself, the first time it opens the focus lands on the save button. Then nothing is
    // there to take what gets typed, and it all goes to the keyboard shortcuts on the window instead.
    g_signal_connect_swapped(menu->popover, "show", G_CALLBACK(gtk_widget_grab_focus), menu->width);

    gtk_menu_button_set_popover(GTK_MENU_BUTTON(button), GTK_WIDGET(menu->popover));
    gtk_button_set_image(GTK_BUTTON(button), gtk_image_new_from_icon_name("document-save-symbolic", GTK_ICON_SIZE_MENU));

    gtk_style_context_add_class(gtk_widget_get_style_context(button), "graph-export");
    gtk_widget_set_tooltip_text(button, "Export this graph to an image");
    gtk_widget_set_can_focus(button, FALSE);
    gtk_widget_set_halign(button, GTK_ALIGN_END);
    gtk_widget_set_valign(button, GTK_ALIGN_START);

    gtk_overlay_add_overlay(overlay, button);

    return menu;
}

static GraphView add_graph(App& app, const sim::GraphOptions& options)
{
    GtkWidget* expander = gtk_expander_new(options.title.c_str());
    GtkWidget* overlay = gtk_overlay_new();
    GtkWidget* area = gtk_drawing_area_new();

    gtk_widget_set_size_request(area, -1, GRAPH_MIN_HEIGHT);
    gtk_widget_set_hexpand(area, TRUE);
    gtk_widget_set_vexpand(area, TRUE);

    gtk_container_add(GTK_CONTAINER(overlay), area);
    gtk_container_add(GTK_CONTAINER(expander), overlay);

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

    view.export_menu = add_export_menu(GTK_OVERLAY(overlay), *view.graph, options.title);

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

    build_project_widgets(app);

    g_object_unref(app.row_labels);

    plotato::FigureStyle world_style;
    world_style.dont_draw_version_text = true;

    app.world_graph = std::make_unique<sim::Canvas>(world_area, world_style);

    project->setup_world(*app.world_graph);

    app.viewbox_controller = ViewboxController(app.world_graph->get_manual_bounds());

    app.left_button = find_button(builder, "t-left");
    app.right_button = find_button(builder, "t-right");
    app.up_button = find_button(builder, "t-up");
    app.down_button = find_button(builder, "t-down");

    app.zoom_in_button = find_button(builder, "z-in");
    app.zoom_out_button = find_button(builder, "z-out");

    app.viewbox_controller.buttons[0] = GTK_WIDGET(app.up_button);
    app.viewbox_controller.buttons[1] = GTK_WIDGET(app.down_button);
    app.viewbox_controller.buttons[2] = GTK_WIDGET(app.left_button);
    app.viewbox_controller.buttons[3] = GTK_WIDGET(app.right_button);
    app.viewbox_controller.buttons[4] = GTK_WIDGET(app.zoom_in_button);
    app.viewbox_controller.buttons[5] = GTK_WIDGET(app.zoom_out_button);

    gtk_widget_set_can_focus(GTK_WIDGET(app.up_button), FALSE);
    gtk_widget_set_can_focus(GTK_WIDGET(app.down_button), FALSE);
    gtk_widget_set_can_focus(GTK_WIDGET(app.left_button), FALSE);
    gtk_widget_set_can_focus(GTK_WIDGET(app.right_button), FALSE);
    gtk_widget_set_can_focus(GTK_WIDGET(app.zoom_in_button), FALSE);
    gtk_widget_set_can_focus(GTK_WIDGET(app.zoom_out_button), FALSE);

    // Bind each of the translate buttons
    // Left
    ButtonBinding left_binding {&app.viewbox_controller, KeyDirection::LEFT};
    g_signal_connect(app.left_button, "button-press-event", G_CALLBACK(on_translate_button_press), &left_binding);
    g_signal_connect(app.left_button, "button-release-event", G_CALLBACK(on_translate_button_release), &left_binding);

    // Right
    ButtonBinding right_binding {&app.viewbox_controller, KeyDirection::RIGHT};
    g_signal_connect(app.right_button, "button-press-event", G_CALLBACK(on_translate_button_press), &right_binding);
    g_signal_connect(app.right_button, "button-release-event", G_CALLBACK(on_translate_button_release), &right_binding);

    // Up
    ButtonBinding up_binding {&app.viewbox_controller, KeyDirection::UP};
    g_signal_connect(app.up_button, "button-press-event", G_CALLBACK(on_translate_button_press), &up_binding);
    g_signal_connect(app.up_button, "button-release-event", G_CALLBACK(on_translate_button_release), &up_binding);

    // Down
    ButtonBinding down_binding {&app.viewbox_controller, KeyDirection::DOWN};
    g_signal_connect(app.down_button, "button-press-event", G_CALLBACK(on_translate_button_press), &down_binding);
    g_signal_connect(app.down_button, "button-release-event", G_CALLBACK(on_translate_button_release), &down_binding);

    // Zoom In
    ButtonBinding zoom_in_binding {&app.viewbox_controller, KeyDirection::ZOOMIN};
    g_signal_connect(app.zoom_in_button, "button-press-event", G_CALLBACK(on_translate_button_press), &zoom_in_binding);
    g_signal_connect(app.zoom_in_button, "button-release-event", G_CALLBACK(on_translate_button_release), &zoom_in_binding);

    // Zoom Out
    ButtonBinding zoom_out_binding {&app.viewbox_controller, KeyDirection::ZOOMOUT};
    g_signal_connect(app.zoom_out_button, "button-press-event", G_CALLBACK(on_translate_button_press), &zoom_out_binding);
    g_signal_connect(app.zoom_out_button, "button-release-event", G_CALLBACK(on_translate_button_release), &zoom_out_binding);

    app.movement_timer = g_timeout_add(
        16,
        ViewboxController::on_timeout_static,
        &app.viewbox_controller
    );

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
