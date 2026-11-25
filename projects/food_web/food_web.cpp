#include "food_web.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <thread>

namespace food_web {

constexpr double PER_HOUR = 3600 / DT;

constexpr double SAMPLE_EVERY = 60; // Seconds, for the population and phase graphs.
constexpr double SAMPLE_HOURS = 24; // How far back they go.

constexpr double GRASS_BLOCK = 25; // Meters across each square of grass that gets drawn.

// Where every slider starts, so that each number is only written down once, in world.hpp.
static const Settings DEFAULTS;

int FoodWeb::default_threads()
{
    // Cores rather than hardware threads, for the reasons in projects/island_evolution.
    return std::max(1, static_cast<int>(std::thread::hardware_concurrency() / 2));
}

FoodWeb::FoodWeb(sim::Ui& ui, uint64_t seed, int threads)
    : shown(ui.add_slider({"Continent shown in the population graphs", 1, 4, 1, 1})),
      grass_max(ui.add_slider({"Most grass, energy per square meter", .01, 1, DEFAULTS.grass_max, .01})),
      grass_growth(ui.add_slider({"How fast grass grows back, per second at its fastest", 0, .05, DEFAULTS.grass_growth, .0005})),
      bison_metabolism(ui.add_slider({"Energy per second a bison uses just living", 0, 1, DEFAULTS.bison_metabolism, .005})),
      bison_graze(ui.add_slider({"Grass a bison eats per second", 0, 5, DEFAULTS.bison_graze, .05})),
      bison_calf_cost(ui.add_slider({"Energy a calf costs its mother (of 80)", 1, 80, DEFAULTS.bison_calf_cost, 1})),
      bison_calving_wait(ui.add_slider({"Seconds between calves", 10, 2000, DEFAULTS.bison_calving_wait, 10})),
      hare_metabolism(ui.add_slider({"Energy per second a hare uses just living", 0, .2, DEFAULTS.hare_metabolism, .001})),
      hare_graze(ui.add_slider({"Grass a hare eats per second", 0, 1, DEFAULTS.hare_graze, .01})),
      hare_kit_cost(ui.add_slider({"Energy a leveret costs its mother (of 15)", .5, 15, DEFAULTS.hare_kit_cost, .5})),
      hare_breeding_wait(ui.add_slider({"Seconds between leverets", 5, 1000, DEFAULTS.hare_breeding_wait, 5})),
      breed_fill(ui.add_slider({"How full a bison or hare has to be to have young", 0, 1, DEFAULTS.breed_fill, .05})),
      bison_floor(ui.add_slider({"Fewest bison on a continent before one wanders in", 0, 50, DEFAULTS.bison_floor, 1})),
      hare_floor(ui.add_slider({"Fewest hares on a continent before one wanders in", 0, 50, DEFAULTS.hare_floor, 1})),
      min_packs(ui.add_slider({"Fewest packs on a continent before one wanders in", 0, 20, DEFAULTS.min_packs, 1})),
      swim_speed(ui.add_slider({"Swimming speed, as a share of top speed", .05, 1, DEFAULTS.swim_speed, .05})),
      wolf_swim_energy(ui.add_slider({"Extra energy per second a wolf uses swimming", 0, 2, DEFAULTS.wolf_swim_energy, .05})),
      bison_swim_energy(ui.add_slider({"Extra energy per second a bison uses swimming", 0, 2, DEFAULTS.bison_swim_energy, .05})),
      hare_swim_energy(ui.add_slider({"Extra energy per second a hare uses swimming", 0, .5, DEFAULTS.hare_swim_energy, .01})),
      bison_healing(ui.add_slider({"Bison health healed per second (of 80)", 0, 30, DEFAULTS.bison_healing, .5})),
      bison_weakest(ui.add_slider({"How strong newborn calves and the oldest bison are, next to one in its prime", .1, 1, DEFAULTS.bison_weakest, .05})),
      bite_damage(ui.add_slider({"Bite damage per second", 0, 30, DEFAULTS.bite_damage, .5})),
      horn_damage(ui.add_slider({"Horn damage per second (wolves have 30 health)", 0, 60, DEFAULTS.horn_damage, .5})),
      bison_energy(ui.add_slider({"Food in a hunted bison in its prime", 0, 1000, DEFAULTS.bison_energy, 10})),
      hare_energy(ui.add_slider({"Food in a hunted hare", 0, 200, DEFAULTS.hare_energy, 1})),
      living_energy(ui.add_slider({"Energy per second a wolf uses just living", 0, 2, DEFAULTS.living_energy, .01})),
      running_energy(ui.add_slider({"Extra energy per second a wolf uses at full speed", 0, 2, DEFAULTS.running_energy, .01})),
      call_energy(ui.add_slider({"Energy per second a wolf uses calling at full volume, per channel", 0, 1, DEFAULTS.call_energy, .005})),
      prey_call_energy(ui.add_slider({"Energy per second prey use calling at full volume", 0, 1, DEFAULTS.prey_call_energy, .005})),
      max_pack(ui.add_slider({"Most wolves in a pack", 3, 20, DEFAULTS.max_pack, 1})),
      daughter_energy(ui.add_slider({"Energy a full pack needs to split (half goes with the daughter)", 20, 2000, DEFAULTS.daughter_energy, 10})),
      max_wolves(ui.add_slider({"Most wolves in the world", 10, 2000, DEFAULTS.max_wolves, 10})),
      mutation_chance(ui.add_slider({"Chance each weight mutates, per child", 0, 1, DEFAULTS.mutation_chance, .01})),
      sight(ui.add_slider({"How far wolves can see (m)", 2, 120, DEFAULTS.sight, 1})),
      populations_graph(ui.add_live_graph({"Populations", "Bison, hares / 2, wolves on the continent shown, last 24 hours", sim::Page::World, "Simulated hours", true})),
      phase_graph(ui.add_live_graph({"Predator and prey", "Wolves against bison on the continent shown, last 24 hours", sim::Page::World, "Bison", true})),
      continent_field(ui.add_field({"Continent shown", sim::Page::World, "%.0f"})),
      bison_field(ui.add_field({"Bison there", sim::Page::World, "%.0f"})),
      hares_field(ui.add_field({"Hares there", sim::Page::World, "%.0f"})),
      wolves_field(ui.add_field({"Wolves there", sim::Page::World, "%.0f"})),
      packs_field(ui.add_field({"Packs in the world", sim::Page::World, "%.0f"})),
      grass_field(ui.add_field({"Grass standing there, share of the most", sim::Page::World, "%.2f"})),
      bison_generation_field(ui.add_field({"Highest bison generation", sim::Page::World, "%.0f"})),
      hare_generation_field(ui.add_field({"Highest hare generation", sim::Page::World, "%.0f"})),
      bison_graph(ui.add_history_graph({"Bison", "Bison alive, every continent"})),
      hares_graph(ui.add_history_graph({"Hares", "Hares alive, every continent"})),
      wolves_graph(ui.add_history_graph({"Wolves", "Wolves alive, every continent"})),
      packs_graph(ui.add_history_graph({"Packs", "Packs alive"})),
      grass_graph(ui.add_history_graph({"Grass", "Grass standing, as a share of the most there could be"})),
      bison_births_graph(ui.add_history_graph({"Bison Births", "Calves born per hour"})),
      bison_kills_graph(ui.add_history_graph({"Bison Kills", "Bison killed by wolves per hour"})),
      hare_births_graph(ui.add_history_graph({"Hare Births", "Leverets born per hour"})),
      hare_kills_graph(ui.add_history_graph({"Hare Kills", "Hares killed by wolves per hour"})),
      starved_graph(ui.add_history_graph({"Starved", "Bison and hares starved per hour"})),
      kill_wolves_graph(ui.add_history_graph({"Wolves Per Kill", "Wolves that bit each bison in its last 10 seconds"})),
      weak_graph(ui.add_history_graph({"Weak Kills", "Share of bison killed that were calves, old, or hurt"})),
      bison_alarm_graph(ui.add_history_graph({"Bison Alarm", "Bison call volume with a wolf in sight, minus without"})),
      hare_alarm_graph(ui.add_history_graph({"Hare Alarm", "Hare call volume with a wolf in sight, minus without"})),
      wolf_call_graph(ui.add_history_graph({"Wolf Calls", "Wolf call volume with a bison in sight, minus without"}))
{
    world = std::make_unique<World>(seed, read_settings(), threads);
}

void FoodWeb::setup_world(plotato::Graph& graph)
{
    graph.equal_aspect_framing = true;
    graph.set_bounds({-HALF_WIDTH * 1.02, HALF_WIDTH * 1.02, -HALF_HEIGHT * 1.02, HALF_HEIGHT * 1.02});
}

Settings FoodWeb::read_settings() const
{
    Settings s;

    s.grass_max = grass_max.value();
    s.grass_growth = grass_growth.value();
    s.bison_metabolism = bison_metabolism.value();
    s.bison_graze = bison_graze.value();
    s.bison_calf_cost = bison_calf_cost.value();
    s.bison_calving_wait = bison_calving_wait.value();
    s.hare_metabolism = hare_metabolism.value();
    s.hare_graze = hare_graze.value();
    s.hare_kit_cost = hare_kit_cost.value();
    s.hare_breeding_wait = hare_breeding_wait.value();
    s.breed_fill = breed_fill.value();
    s.bison_floor = bison_floor.value();
    s.hare_floor = hare_floor.value();
    s.min_packs = min_packs.value();
    s.swim_speed = swim_speed.value();
    s.wolf_swim_energy = wolf_swim_energy.value();
    s.bison_swim_energy = bison_swim_energy.value();
    s.hare_swim_energy = hare_swim_energy.value();
    s.bison_healing = bison_healing.value();
    s.bison_weakest = bison_weakest.value();
    s.bite_damage = bite_damage.value();
    s.horn_damage = horn_damage.value();
    s.bison_energy = bison_energy.value();
    s.hare_energy = hare_energy.value();
    s.living_energy = living_energy.value();
    s.running_energy = running_energy.value();
    s.call_energy = call_energy.value();
    s.prey_call_energy = prey_call_energy.value();
    s.max_pack = max_pack.value();
    s.daughter_energy = daughter_energy.value();
    s.max_wolves = max_wolves.value();
    s.mutation_chance = mutation_chance.value();
    s.sight = sight.value();

    return s;
}

void FoodWeb::sample()
{
    Sample now;

    now.hours = world->time / 3600;
    now.bison.assign(static_cast<std::size_t>(world->continents()), 0);
    now.hares.assign(static_cast<std::size_t>(world->continents()), 0);
    now.wolves.assign(static_cast<std::size_t>(world->continents()), 0);

    for (const Prey& animal : world->prey) {
        if (animal.land >= 0) {
            (animal.bison ? now.bison : now.hares)[static_cast<std::size_t>(animal.land)]++;
        }
    }

    for (const Wolf& wolf : world->wolves) {
        if (wolf.land >= 0) {
            now.wolves[static_cast<std::size_t>(wolf.land)]++;
        }
    }

    samples.push_back(std::move(now));

    while (!samples.empty() && samples.front().hours < samples.back().hours - SAMPLE_HOURS) {
        samples.pop_front();
    }
}

void FoodWeb::step(double)
{
    const Settings settings = read_settings();

    world->step(settings);

    sample_wait -= DT;

    if (sample_wait <= 0) {
        sample_wait = SAMPLE_EVERY;
        sample();
    }

    Tally total;

    for (const Tally& tally : world->tallies) {
        total += tally;
    }

    const Tally now = total - last_total;

    last_total = total;

    double bison = 0;
    double hares = 0;

    for (const Prey& animal : world->prey) {
        (animal.bison ? bison : hares) += 1;
    }

    bison_graph.add(bison);
    hares_graph.add(hares);
    wolves_graph.add(static_cast<double>(world->wolves.size()));
    packs_graph.add(static_cast<double>(world->packs.size()));

    double land_cells = 0;

    for (int land = 0; land < world->continents(); land++) {
        land_cells += world->terrain.landmass_area(land) / (GRASS_CELL * GRASS_CELL);
    }

    if (land_cells > 0) {
        grass_graph.add(now.grass / (land_cells * settings.grass_max * GRASS_CELL * GRASS_CELL));
    }

    bison_births_graph.add(static_cast<double>(now.bison_born) * PER_HOUR);
    bison_kills_graph.add(static_cast<double>(now.bison_killed) * PER_HOUR);
    hare_births_graph.add(static_cast<double>(now.hares_born) * PER_HOUR);
    hare_kills_graph.add(static_cast<double>(now.hares_killed) * PER_HOUR);
    starved_graph.add(static_cast<double>(now.bison_starved + now.hares_starved) * PER_HOUR);

    if (now.bison_killed > 0) {
        kill_wolves_graph.add(now.kill_wolves / static_cast<double>(now.bison_killed));
        weak_graph.add(static_cast<double>(now.weak_killed) / static_cast<double>(now.bison_killed));
    }

    const auto difference = [](const std::array<double, 2>& calls, const std::array<double, 2>& steps) {
        return (steps[1] > 0 ? calls[1] / steps[1] : 0) - (steps[0] > 0 ? calls[0] / steps[0] : 0);
    };

    if (now.bison_call_steps[1] > 0) {
        bison_alarm_graph.add(difference(now.bison_calls, now.bison_call_steps));
    }

    if (now.hare_call_steps[1] > 0) {
        hare_alarm_graph.add(difference(now.hare_calls, now.hare_call_steps));
    }

    if (now.wolf_call_steps[1] > 0) {
        wolf_call_graph.add(difference(now.wolf_calls, now.wolf_call_steps));
    }
}

static plotato::Color colour_of(const Colour& colour)
{
    const auto part = [](double value) { return static_cast<uint8_t>(40 + 200 * clamp(value, 0, 1)); };

    return plotato::Color(part(colour[0]), part(colour[1]), part(colour[2]));
}

static plotato::ShapeStyle filled(plotato::Color fill, plotato::Color outline = plotato::Color(0, 0, 0, 0), double width = 0)
{
    plotato::ShapeStyle style;

    style.fill = fill;
    style.outline = outline;
    style.outline_width = width;

    return style;
}

static plotato::MarkerStyle marker(plotato::Color fill, double size, plotato::Color outline = plotato::Color(0, 0, 0, 0), double width = 0)
{
    plotato::MarkerStyle style;

    style.fill = fill;
    style.outline = outline;
    style.outline_width = width;
    style.size = size;

    return style;
}

void FoodWeb::draw(plotato::Graph& graph)
{
    const Settings settings = read_settings();

    graph.polygon({-HALF_WIDTH, -HALF_WIDTH, HALF_WIDTH, HALF_WIDTH}, {-HALF_HEIGHT, HALF_HEIGHT, HALF_HEIGHT, -HALF_HEIGHT}, filled(plotato::Color(150, 195, 230)));

    for (const LandRun& run : world->terrain.runs()) {
        graph.polygon({run.left, run.left, run.right, run.right}, {run.bottom, run.top, run.top, run.bottom}, filled(plotato::Color(222, 205, 160)));
    }

    // Grass, in squares, greener the more there is.
    const double most = settings.grass_max * GRASS_CELL * GRASS_CELL;
    const int per_block = static_cast<int>(GRASS_BLOCK / GRASS_CELL);

    for (int row = 0; row < world->grass_rows; row += per_block) {
        for (int column = 0; column < world->grass_columns; column += per_block) {
            double total = 0;
            int land = 0;

            for (int r = row; r < std::min(world->grass_rows, row + per_block); r++) {
                for (int c = column; c < std::min(world->grass_columns, column + per_block); c++) {
                    const Vec middle = {-HALF_WIDTH + (c + .5) * GRASS_CELL, -HALF_HEIGHT + (r + .5) * GRASS_CELL};

                    if (world->terrain.is_land(middle)) {
                        total += world->grass[static_cast<std::size_t>(r * world->grass_columns + c)];
                        land++;
                    }
                }
            }

            if (land * 2 < per_block * per_block) {
                continue;
            }

            const double share = most > 0 ? total / (land * most) : 0;
            const double left = -HALF_WIDTH + column * GRASS_CELL;
            const double bottom = -HALF_HEIGHT + row * GRASS_CELL;
            const double size = per_block * GRASS_CELL;

            graph.polygon({left, left, left + size, left + size}, {bottom, bottom + size, bottom + size, bottom},
                          filled(plotato::Color(70, 150, 50, static_cast<uint8_t>(200 * clamp(share, 0, 1)))));
        }
    }

    plotato::PlotStyle wall;
    wall.line_color = plotato::Color(60, 60, 80);
    wall.line_width = 2;

    graph.plot({-HALF_WIDTH, -HALF_WIDTH, HALF_WIDTH, HALF_WIDTH, -HALF_WIDTH}, {-HALF_HEIGHT, HALF_HEIGHT, HALF_HEIGHT, -HALF_HEIGHT, -HALF_HEIGHT}, wall);

    std::vector<double> x;
    std::vector<double> y;

    const auto scatter = [&](const plotato::MarkerStyle& style) {
        if (!x.empty()) {
            graph.scatter(x, y, style);
        }

        x.clear();
        y.clear();
    };

    for (const Carcass& carcass : world->carcasses) {
        x.push_back(carcass.pos.x);
        y.push_back(carcass.pos.y);
    }

    scatter(marker(plotato::Color(140, 20, 20), 4));

    for (const Prey& animal : world->prey) {
        if (!animal.bison) {
            x.push_back(animal.body.pos.x);
            y.push_back(animal.body.pos.y);
        }
    }

    scatter(marker(plotato::Color(120, 110, 100), 3));

    for (const Prey& animal : world->prey) {
        if (animal.bison) {
            x.push_back(animal.body.pos.x);
            y.push_back(animal.body.pos.y);
        }
    }

    scatter(marker(plotato::Color(100, 65, 35), 5, plotato::Color(40, 25, 10), 1));

    // Wolves, one scatter per pack, coloured by pack.
    for (const auto& entry : world->packs) {
        for (const Wolf& wolf : world->wolves) {
            if (wolf.pack == entry.first) {
                x.push_back(wolf.body.pos.x);
                y.push_back(wolf.body.pos.y);
            }
        }

        scatter(marker(colour_of(entry.second.colour), 6, plotato::Color(0, 0, 0), 1));
    }

    // Rings round anything calling loudly, and round wolves that are biting.
    for (const Wolf& wolf : world->wolves) {
        const float loudest = *std::max_element(wolf.calls.begin(), wolf.calls.end());

        if (loudest > .3f) {
            graph.circle(wolf.body.pos.x, wolf.body.pos.y, 2 + 4 * loudest, filled(plotato::Color(0, 0, 0, 0), plotato::Color(40, 90, 230), 1.5));
        }

        if (wolf.biting && (wolf.at_bison || wolf.at_hare)) {
            graph.circle(wolf.body.pos.x, wolf.body.pos.y, 1.5, filled(plotato::Color(220, 0, 0, 160)));
        }
    }

    for (const Prey& animal : world->prey) {
        if (animal.call > .3) {
            graph.circle(animal.body.pos.x, animal.body.pos.y, 2 + 4 * animal.call, filled(plotato::Color(0, 0, 0, 0), plotato::Color(240, 160, 0), 1.5));
        }
    }

    plotato::FigureTextStyle caption;
    caption.font_size = 12;
    caption.absolute = false;
    caption.anchor = plotato::Anchor::BOTTOM_LEFT;
    caption.background_color = plotato::Color(0, 0, 0, 0);
    caption.outline_color = plotato::Color(0, 0, 0, 0);
    caption.outline_width = 0;

    char text[120];

    std::snprintf(text, sizeof(text), "%.1f hours", world->time / 3600);
    graph.add_text(-HALF_WIDTH, HALF_HEIGHT * 1.005, text, caption);

    // The continent picked for the graphs and fields.
    const int index = std::min(world->continents() - 1, std::max(0, static_cast<int>(std::lround(shown.value())) - 1));

    if (!samples.empty() && index >= 0) {
        std::vector<double> hours;
        std::vector<double> bison;
        std::vector<double> hares;
        std::vector<double> wolves;

        for (const Sample& sample : samples) {
            hours.push_back(sample.hours);
            bison.push_back(sample.bison[static_cast<std::size_t>(index)]);
            hares.push_back(sample.hares[static_cast<std::size_t>(index)] / 2.0);
            wolves.push_back(sample.wolves[static_cast<std::size_t>(index)]);
        }

        plotato::PlotStyle bison_line;
        bison_line.line_color = plotato::Color(120, 75, 40);
        plotato::PlotStyle hare_line;
        hare_line.line_color = plotato::Color(150, 150, 150);
        plotato::PlotStyle wolf_line;
        wolf_line.line_color = plotato::Color(40, 90, 230);

        plotato::Graph& populations = populations_graph.graph();

        populations.plot(hours, bison, bison_line);
        populations.plot(hours, hares, hare_line);
        populations.plot(hours, wolves, wolf_line);

        phase_graph.graph().plot(bison, wolves, wolf_line);

        const Sample& last = samples.back();

        continent_field.set(index + 1);
        bison_field.set(last.bison[static_cast<std::size_t>(index)]);
        hares_field.set(last.hares[static_cast<std::size_t>(index)]);
        wolves_field.set(last.wolves[static_cast<std::size_t>(index)]);

        const std::vector<double> standing = world->grass_by_continent();
        const double cells = world->terrain.landmass_area(index) / (GRASS_CELL * GRASS_CELL);

        grass_field.set(cells > 0 ? standing[static_cast<std::size_t>(index)] / (cells * most) : 0);
    }

    packs_field.set(static_cast<double>(world->packs.size()));
    bison_generation_field.set(world->highest_generation(true));
    hare_generation_field.set(world->highest_generation(false));
}

}

std::unique_ptr<sim::Project> sim::create_project(sim::Ui& ui, uint64_t seed)
{
    return std::make_unique<food_web::FoodWeb>(ui, seed);
}
