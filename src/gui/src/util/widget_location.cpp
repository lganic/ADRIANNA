/*
This file provides utilities for locating and caching widget references by ID.
*/

#include <gtk/gtk.h>
#include <string>
#include <unordered_map>

// Static cache of widget pointers
static std::unordered_map<std::string, GtkWidget*> widget_cache;

/*
Fetch and cache any GTK object by ID.
*/
GtkWidget* get_cached_widget(GtkBuilder* builder, const std::string& id)
{
    // Return cached widget if found
    auto it = widget_cache.find(id);
    if (it != widget_cache.end())
        return it->second;

    GtkWidget* widget = GTK_WIDGET(gtk_builder_get_object(builder, id.c_str()));
    if (!widget) {
        g_warning("Widget with ID '%s' not found", id.c_str());
        return nullptr;
    }

    // Cache the reference
    widget_cache[id] = widget;
    return widget;
}

/*
Find and return an entry widget by ID.
*/
GtkWidget* find_entry(GtkBuilder* builder, const char* entry_id)
{
    return get_cached_widget(builder, entry_id);
}

GtkWidget* find_entry(GtkBuilder* builder, const std::string& entry_id)
{
    return get_cached_widget(builder, entry_id);
}

/*
Find and return a label widget by ID.
*/
GtkLabel* find_label(GtkBuilder* builder, const char* label_id)
{
    return GTK_LABEL(get_cached_widget(builder, label_id));
}

GtkLabel* find_label(GtkBuilder* builder, const std::string& label_id)
{
    return GTK_LABEL(get_cached_widget(builder, label_id));
}

/*
Find and return a dropdown widget by ID.
*/
GtkComboBoxText* find_dropdown(GtkBuilder* builder, const char* dropdown_id)
{
    return GTK_COMBO_BOX_TEXT(get_cached_widget(builder, dropdown_id));
}

GtkComboBoxText* find_dropdown(GtkBuilder* builder, const std::string& dropdown_id)
{
    return GTK_COMBO_BOX_TEXT(get_cached_widget(builder, dropdown_id));
}

/*
Find and return a switch widget by ID.
*/
GtkSwitch* find_switch(GtkBuilder* builder, const char* switch_id)
{
    return GTK_SWITCH(get_cached_widget(builder, switch_id));
}

GtkSwitch* find_switch(GtkBuilder* builder, const std::string& switch_id)
{
    return GTK_SWITCH(get_cached_widget(builder, switch_id));
}

GtkSpinButton* find_spin_button(GtkBuilder* builder, const char* button_id)
{
    return GTK_SPIN_BUTTON(get_cached_widget(builder, button_id));
}

GtkSpinButton* find_spin_button(GtkBuilder* builder, const std::string& button_id)
{
    return GTK_SPIN_BUTTON(get_cached_widget(builder, button_id));
}

GtkButton* find_button(GtkBuilder* builder, const char* button_id)
{
    return GTK_BUTTON(get_cached_widget(builder, button_id));
}

GtkButton* find_button(GtkBuilder* builder, const std::string& button_id)
{
    return GTK_BUTTON(get_cached_widget(builder, button_id));
}

GtkImage* find_image(GtkBuilder* builder, const char* image_id)
{
    return GTK_IMAGE(get_cached_widget(builder, image_id));
}

GtkImage* find_image(GtkBuilder* builder, const std::string& image_id)
{
    return GTK_IMAGE(get_cached_widget(builder, image_id));
}

GtkBox* find_box(GtkBuilder* builder, const char* box_id)
{
    return GTK_BOX(get_cached_widget(builder, box_id));
}

GtkBox* find_box(GtkBuilder* builder, const std::string& box_id)
{
    return GTK_BOX(get_cached_widget(builder, box_id));
}

GtkSpinner* find_spinner(GtkBuilder* builder, const char* spinner_id)
{
    return GTK_SPINNER(get_cached_widget(builder, spinner_id));
}

GtkSpinner* find_spinner(GtkBuilder* builder, std::string& spinner_id)
{
    return GTK_SPINNER(get_cached_widget(builder, spinner_id));
}

GtkProgressBar* find_progress_bar(GtkBuilder* builder, const char* progress_bar_id)
{
    return GTK_PROGRESS_BAR(get_cached_widget(builder, progress_bar_id));
}

GtkProgressBar* find_progress_bar(GtkBuilder* builder, std::string& progress_bar_id)
{
    return GTK_PROGRESS_BAR(get_cached_widget(builder, progress_bar_id));
}

GtkDrawingArea* find_drawing_area(GtkBuilder* builder, const char* drawing_area_id)
{
    return GTK_DRAWING_AREA(get_cached_widget(builder, drawing_area_id));
}

GtkDrawingArea* find_drawing_area(GtkBuilder* builder, std::string& drawing_area_id)
{
    return GTK_DRAWING_AREA(get_cached_widget(builder, drawing_area_id));
}