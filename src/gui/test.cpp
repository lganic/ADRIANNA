#include <gtk/gtk.h>
#include <iostream>

void on_window_close(GtkWidget* widget, gpointer user_data)
{
    gtk_main_quit();
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

    gtk_window_set_title(GTK_WINDOW(main_window), "EVO-Dev");

    g_signal_connect(main_window, "destroy", G_CALLBACK(on_window_close), nullptr);

    gtk_widget_show_all(main_window);
    gtk_main();

    g_object_unref(builder);
    return 0;
}