#ifndef LOCATION_HPP
#define LOCATION_HPP

#include <gtk/gtk.h>

GtkWidget* get_cached_widget(GtkBuilder* builder, const std::string& id);
GtkWidget* find_entry(GtkBuilder* builder, const char* entry_id);
GtkWidget* find_entry(GtkBuilder* builder, const std::string& entry_id);
GtkLabel* find_label(GtkBuilder* builder, const char* label_id);
GtkLabel* find_label(GtkBuilder* builder, const std::string& label_id);
GtkComboBoxText* find_dropdown(GtkBuilder* builder, const char* dropdown_id);
GtkComboBoxText* find_dropdown(GtkBuilder* builder, const std::string& dropdown_id);
GtkSwitch* find_switch(GtkBuilder* builder, const char* switch_id);
GtkSwitch* find_switch(GtkBuilder* builder, const std::string& switch_id);
GtkSpinButton* find_spin_button(GtkBuilder* builder, const char* button_id);
GtkSpinButton* find_spin_button(GtkBuilder* builder, const std::string& button_id);
GtkButton* find_button(GtkBuilder* builder, const char* button_id);
GtkButton* find_button(GtkBuilder* builder, const std::string& button_id);
GtkImage* find_image(GtkBuilder* builder, const char* image_id);
GtkImage* find_image(GtkBuilder* builder, const std::string& image_id);
GtkBox* find_box(GtkBuilder* builder, const char* box_id);
GtkBox* find_box(GtkBuilder* builder, const std::string& box_id);
GtkSpinner* find_spinner(GtkBuilder* builder, const char* spinner_id);
GtkSpinner* find_spinner(GtkBuilder* builder, std::string& spinner_id);
GtkProgressBar* find_progress_bar(GtkBuilder* builder, const char* progress_bar_id);
GtkProgressBar* find_progress_bar(GtkBuilder* builder, std::string& progress_bar_id);
GtkDrawingArea* find_drawing_area(GtkBuilder* builder, const char * drawing_area_id);
GtkDrawingArea* find_drawing_area(GtkBuilder* builder, std::string& drawing_area_id);

#endif // LOCATION_HPP