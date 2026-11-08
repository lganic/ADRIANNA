#include <sim/canvas.hpp>

#include <mutex>

namespace sim {

void Canvas::swap_contents(PlotContents& contents)
{
    std::scoped_lock lock(data_mutex, text_mutex);

    current_plot_items.swap(contents.plots);
    text_items.swap(contents.text);
}

// Plotato needs a widget to attach a graph to, even if it is never going to be seen.
OffscreenCanvas::OffscreenCanvas() : Canvas(GTK_WIDGET(g_object_ref_sink(gtk_drawing_area_new()))) {}

OffscreenCanvas::~OffscreenCanvas()
{
    g_object_unref(area);
}

}
