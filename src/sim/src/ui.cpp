#include <sim/ui.hpp>

namespace sim {

HistoryGraph Ui::add_history_graph(GraphOptions options)
{
    items.push_back({ItemType::HistoryGraph, history_graphs.size()});

    history_graphs.push_back({std::move(options)});

    return HistoryGraph(history_graphs.back());
}

CurveGraph Ui::add_curve_graph(GraphOptions options)
{
    items.push_back({ItemType::CurveGraph, curve_graphs.size()});

    curve_graphs.push_back({std::move(options)});

    return CurveGraph(curve_graphs.back());
}

Field Ui::add_field(FieldOptions options)
{
    items.push_back({ItemType::Field, fields.size()});

    fields.push_back({std::move(options)});

    return Field(fields.back());
}

Control Ui::add_slider(ControlOptions options)
{
    items.push_back({ItemType::Slider, controls.size()});

    controls.emplace_back(std::move(options));

    return Control(controls.back());
}

Control Ui::add_entry(ControlOptions options)
{
    items.push_back({ItemType::Entry, controls.size()});

    controls.emplace_back(std::move(options));

    return Control(controls.back());
}

}
