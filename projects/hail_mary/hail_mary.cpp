#include "hail_mary.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>
#include <thread>

namespace hail_mary {

constexpr double PER_HOUR = 3600 / DT;

// The colour of each channel, for the rings drawn round wolves that are calling.
const plotato::Color CHANNEL_COLOURS[CHANNELS] = {
    plotato::Color(220, 40, 40),
    plotato::Color(40, 90, 230),
    plotato::Color(240, 160, 0),
};

int HailMary::default_threads()
{
    // Cores rather than hardware threads, for the reasons in projects/island_evolution.
    return std::max(1, static_cast<int>(std::thread::hardware_concurrency() / 2));
}

HailMary::HailMary(sim::Ui& ui, uint64_t seed, int continent_count, int thread_count)
    : threads(std::max(1, std::min(thread_count, continent_count))),
      shown(ui.add_slider({"Continent shown", 1, static_cast<double>(std::max(1, continent_count)), 1, 1})),
      mute(ui.add_slider({"Calls muted on: 0 none, 1 every other continent, 2 all", 0, 2, 0, 1})),
      migration(ui.add_slider({"Minutes between migrations (0 for never)", 0, 240, 30, 5})),
      bison_wanted(ui.add_slider({"Bison on a continent", 0, 100, 30, 1})),
      hares_wanted(ui.add_slider({"Hares on a continent", 0, 200, 30, 1})),
      bison_calving(ui.add_slider({"Seconds before a bison that died is replaced, on average", 0, 600, 30, 5})),
      hare_breeding(ui.add_slider({"Seconds before a hare that died is replaced, on average", 0, 600, 20, 5})),
      bison_healing(ui.add_slider({"Bison health healed per second (of 80)", 0, 30, 8, .5})),
      bite_damage(ui.add_slider({"Bite damage per second", 0, 30, 6, .5})),
      horn_damage(ui.add_slider({"Horn damage per second (wolves have 30 health)", 0, 60, 8, .5})),
      bison_fight(ui.add_slider({"Chance per second that a bitten bison turns on the biter", 0, 5, .3, .05})),
      herd_defence(ui.add_slider({"Chance per second that a grown bison goes for a wolf biting the herd nearby", 0, 5, .5, .05})),
      bison_energy(ui.add_slider({"Energy in a hunted bison", 0, 1000, 240, 10})),
      hare_energy(ui.add_slider({"Energy in a hunted hare", 0, 200, 12, 1})),
      old_bison_energy(ui.add_slider({"Energy in a bison that died of old age", 0, 1000, 60, 10})),
      carcass_lifetime(ui.add_slider({"Seconds before a carcass rots away", 5, 600, 120, 5})),
      eat_rate(ui.add_slider({"Energy a wolf eats per second", .5, 50, 10, .5})),
      living_energy(ui.add_slider({"Energy per second a wolf uses just living", 0, 2, .12, .01})),
      running_energy(ui.add_slider({"Extra energy per second at full speed", 0, 2, .25, .01})),
      call_energy(ui.add_slider({"Energy per second calling at full volume, per channel", 0, 1, .05, .005})),
      bite_energy(ui.add_slider({"Energy per second biting", 0, 1, .1, .01})),
      pup_cost(ui.add_slider({"Energy a pup costs", 0, 500, 40, 5})),
      pup_reserve(ui.add_slider({"Energy a pack keeps in store per wolf before having a pup", 0, 500, 60, 5})),
      max_pack(ui.add_slider({"Most wolves in a pack (1 for no packs)", 1, 12, 6, 1})),
      daughter_energy(ui.add_slider({"Energy a full pack needs to found another (half goes with it)", 20, 2000, 400, 10})),
      max_wolves(ui.add_slider({"Most wolves on a continent", 4, 200, 60, 1})),
      mutation_chance(ui.add_slider({"Chance each weight mutates, per daughter pack", 0, 1, 1, .01})),
      sight(ui.add_slider({"How far wolves can see (m)", 2, 120, 30, 1})),
      noise_volume(ui.add_slider({"How loud biting and feeding sound, on channel 0 (a call is 0 to 1)", 0, 1, .3, .05})),
      prey_hearing(ui.add_slider({"How far prey hear a full volume call from (m, 0 for not at all)", 0, 100, 20, 1})),
      founders(ui.add_slider({"First packs: 0 random brains, 1 plain predators (see Genome::ancestor)", 0, 1, 1, 1})),
      scripted(ui.add_slider({"Test: wolves ignore their brains and chase 0 nothing, 1 hares, 2 bison, 3 either", 0, 3, 0, 1})),
      continent_field(ui.add_field({"Continent shown", sim::Page::World, "%.0f"})),
      muted_field(ui.add_field({"Calls muted here (1 for yes)", sim::Page::World, "%.0f"})),
      wolves_field(ui.add_field({"Wolves here", sim::Page::World, "%.0f"})),
      packs_field(ui.add_field({"Packs here", sim::Page::World, "%.0f"})),
      bison_field(ui.add_field({"Bison here", sim::Page::World, "%.0f"})),
      hares_field(ui.add_field({"Hares here", sim::Page::World, "%.0f"})),
      generation_field(ui.add_field({"Highest generation here", sim::Page::World, "%.0f"})),
      bison_kills_field(ui.add_field({"Bison killed here", sim::Page::World, "%.0f"})),
      hare_kills_field(ui.add_field({"Hares killed here", sim::Page::World, "%.0f"})),
      migrations_field(ui.add_field({"Migrations everywhere", sim::Page::World, "%.0f"})),
      wolves_graph(ui.add_history_graph({"Wolves", "Wolves alive, every continent"})),
      packs_graph(ui.add_history_graph({"Packs", "Packs alive, every continent"})),
      bison_graph(ui.add_history_graph({"Bison", "Bison alive, every continent"})),
      hares_graph(ui.add_history_graph({"Hares", "Hares alive, every continent"})),
      bison_kills_heard_graph(ui.add_history_graph({"Bison Kills", "Bison killed per hour per continent, where calls are heard"})),
      bison_kills_muted_graph(ui.add_history_graph({"Bison Kills Muted", "Bison killed per hour per continent, where calls are muted"})),
      hare_kills_graph(ui.add_history_graph({"Hare Kills", "Hares killed per hour per continent"})),
      kill_wolves_graph(ui.add_history_graph({"Wolves Per Kill", "Wolves that bit each bison in its last 10 seconds"})),
      surround_graph(ui.add_history_graph({"Surround", "How one-sided the last bites were (1 one side, 0 all round)"})),
      front_graph(ui.add_history_graph({"Front Bites", "Fraction of bites on bison from in front of its horns"})),
      bison_share_graph(ui.add_history_graph({"Bison Share", "Fraction of food from hunted bison"})),
      hare_share_graph(ui.add_history_graph({"Hare Share", "Fraction of food from hunted hares"})),
      scavenged_graph(ui.add_history_graph({"Scavenged", "Fraction of food from carcasses nobody killed"})),
      calling_graph(ui.add_history_graph({"Calling", "Average call volume, over every channel"})),
      signal_graph(ui.add_history_graph({"Signal", "Call volume near a bison, minus with nothing in sight"})),
      approach_heard_graph(ui.add_history_graph({"Approach", "Speed towards a packmate's call, out of sight of anything"})),
      approach_muted_graph(ui.add_history_graph({"Approach Muted", "The same, where calls are muted"})),
      wolves_killed_graph(ui.add_history_graph({"Wolves Killed", "Wolves killed by other packs, per hour per continent"})),
      gored_graph(ui.add_history_graph({"Gored", "Wolves gored by bison, per hour per continent"})),
      daughters_graph(ui.add_history_graph({"Daughters", "Daughter packs founded per hour per continent"})),
      generation_graph(ui.add_history_graph({"Generation", "Average generation of the packs alive"})),
      mutation_graph(ui.add_history_graph({"Mutation", "Average mutation size"}))
{
    const Settings settings = read_settings();

    std::mt19937_64 seeds(seed);

    for (int i = 0; i < std::max(1, continent_count); i++) {
        continents.push_back(std::make_unique<Continent>(i, seeds(), settings));
    }

    last_stats.resize(continents.size());

    migration_wait = migration.value() * 60;
}

void HailMary::setup_world(plotato::Graph& graph)
{
    graph.equal_aspect_framing = true;
    graph.set_bounds({-HALF_SIZE * 1.05, HALF_SIZE * 1.05, -HALF_SIZE * 1.1, HALF_SIZE * 1.1});
}

Settings HailMary::read_settings() const
{
    Settings settings;

    settings.bison_wanted = bison_wanted.value();
    settings.hares_wanted = hares_wanted.value();
    settings.bison_calving = bison_calving.value();
    settings.hare_breeding = hare_breeding.value();
    settings.bison_healing = bison_healing.value();
    settings.bite_damage = bite_damage.value();
    settings.horn_damage = horn_damage.value();
    settings.bison_fight = bison_fight.value();
    settings.herd_defence = herd_defence.value();
    settings.bison_energy = bison_energy.value();
    settings.hare_energy = hare_energy.value();
    settings.old_bison_energy = old_bison_energy.value();
    settings.carcass_lifetime = carcass_lifetime.value();
    settings.eat_rate = eat_rate.value();

    settings.living_energy = living_energy.value();
    settings.running_energy = running_energy.value();
    settings.call_energy = call_energy.value();
    settings.bite_energy = bite_energy.value();
    settings.pup_cost = pup_cost.value();
    settings.pup_reserve = pup_reserve.value();
    settings.max_pack = max_pack.value();
    settings.daughter_energy = daughter_energy.value();
    settings.max_wolves = max_wolves.value();
    settings.mutation_chance = mutation_chance.value();

    settings.sight = sight.value();
    settings.noise_volume = noise_volume.value();
    settings.prey_hearing = prey_hearing.value();
    settings.scripted = static_cast<int>(std::lround(scripted.value()));
    settings.founders = static_cast<int>(std::lround(founders.value()));

    return settings;
}

bool HailMary::muted(int continent) const
{
    const int mode = static_cast<int>(std::lround(mute.value()));

    return mode >= 2 || (mode == 1 && continent % 2 == 1);
}

void HailMary::step(double)
{
    const Settings settings = read_settings();

    std::vector<Settings> each(continents.size(), settings);

    for (std::size_t i = 0; i < continents.size(); i++) {
        each[i].muted = muted(static_cast<int>(i));
    }

    next_continent.store(0, std::memory_order_relaxed);

    const int count = static_cast<int>(continents.size());

    threads.run([&](int) {
        for (int i = next_continent.fetch_add(1, std::memory_order_relaxed); i < count; i = next_continent.fetch_add(1, std::memory_order_relaxed)) {
            continents[static_cast<std::size_t>(i)]->step(each[static_cast<std::size_t>(i)]);
        }
    });

    time += DT;

    migrate(settings);
    record(settings);
}

void HailMary::migrate(const Settings& settings)
{
    const double minutes = migration.value();

    if (minutes <= 0 || continents.size() < 2) {
        migration_wait = minutes * 60;
        return;
    }

    migration_wait -= DT;

    if (migration_wait > 0) {
        return;
    }

    migration_wait = minutes * 60;

    // Round the ring, each continent's best pack sends a pair to the next one. All of them are
    // picked before any of them arrive, so the order makes no difference.
    std::vector<Archived> leaving(continents.size());
    std::vector<bool> any(continents.size(), false);

    for (std::size_t i = 0; i < continents.size(); i++) {
        any[i] = continents[i]->best(leaving[i]);
    }

    for (std::size_t i = 0; i < continents.size(); i++) {
        if (any[i]) {
            continents[(i + 1) % continents.size()]->immigrate(leaving[i], settings);
            migrations++;
        }
    }
}

Stats total(const HailMary& world, bool heard)
{
    Stats sum;

    for (std::size_t i = 0; i < world.continents.size(); i++) {
        if (world.muted(static_cast<int>(i)) != heard) {
            sum += world.continents[i]->stats;
        }
    }

    return sum;
}

void HailMary::record(const Settings&)
{
    // Each continent on its own, so that moving the mute slider part way through doesn't mix up
    // which continent's counts get taken away from which.
    Stats now_heard;
    Stats now_muted;

    for (std::size_t i = 0; i < continents.size(); i++) {
        const Stats& stats = continents[i]->stats;

        (muted(static_cast<int>(i)) ? now_muted : now_heard) += stats - last_stats[i];

        last_stats[i] = stats;
    }

    Stats both = now_heard;
    both += now_muted;

    int heard_count = 0;

    for (std::size_t i = 0; i < continents.size(); i++) {
        heard_count += muted(static_cast<int>(i)) ? 0 : 1;
    }

    const int muted_count = static_cast<int>(continents.size()) - heard_count;
    const double continent_count = static_cast<double>(continents.size());

    double wolves = 0;
    double packs = 0;
    double bison = 0;
    double hares = 0;
    double generation = 0;
    double mutation = 0;
    double calling = 0;
    double wolf_count = 0;

    for (std::size_t i = 0; i < continents.size(); i++) {
        const Continent& continent = *continents[i];

        wolves += static_cast<double>(continent.wolves.size());
        packs += static_cast<double>(continent.packs.size());

        for (const Prey& animal : continent.prey) {
            (animal.bison ? bison : hares) += 1;
        }

        for (const auto& entry : continent.packs) {
            generation += entry.second.genome->generation;
            mutation += entry.second.genome->mutation_size;
        }

        if (!muted(static_cast<int>(i))) {
            for (const Wolf& wolf : continent.wolves) {
                for (float call : wolf.calls) {
                    calling += call / CHANNELS;
                }

                wolf_count++;
            }
        }
    }

    wolves_graph.add(wolves);
    packs_graph.add(packs);
    bison_graph.add(bison);
    hares_graph.add(hares);

    if (heard_count > 0) {
        bison_kills_heard_graph.add(static_cast<double>(now_heard.bison_kills) * PER_HOUR / heard_count);
    }

    if (muted_count > 0) {
        bison_kills_muted_graph.add(static_cast<double>(now_muted.bison_kills) * PER_HOUR / muted_count);
    }

    hare_kills_graph.add(static_cast<double>(both.hare_kills) * PER_HOUR / continent_count);

    if (both.bison_kills > 0) {
        kill_wolves_graph.add(both.kill_wolves / static_cast<double>(both.bison_kills));
    }

    if (both.surround_kills > 0) {
        surround_graph.add(both.surround / static_cast<double>(both.surround_kills));
    }

    if (both.bites > 0) {
        front_graph.add(static_cast<double>(both.front_bites) / static_cast<double>(both.bites));
    }

    double eaten = 0;

    for (double amount : both.eaten) {
        eaten += amount;
    }

    if (eaten > 0) {
        bison_share_graph.add(both.eaten[HUNTED_BISON] / eaten);
        hare_share_graph.add(both.eaten[HUNTED_HARE] / eaten);
        scavenged_graph.add((both.eaten[OLD_AGE] + both.eaten[DEAD_WOLF]) / eaten);
    }

    if (wolf_count > 0) {
        calling_graph.add(calling / wolf_count);
    }

    if (heard_count > 0) {
        signal_graph.add(call_signal(now_heard));
    }

    if (now_heard.approach_steps > 0) {
        approach_heard_graph.add(now_heard.approach / static_cast<double>(now_heard.approach_steps));
    }

    if (now_muted.approach_steps > 0) {
        approach_muted_graph.add(now_muted.approach / static_cast<double>(now_muted.approach_steps));
    }

    wolves_killed_graph.add(static_cast<double>(both.wolves_killed) * PER_HOUR / continent_count);
    gored_graph.add(static_cast<double>(both.wolves_gored) * PER_HOUR / continent_count);
    daughters_graph.add(static_cast<double>(both.daughters) * PER_HOUR / continent_count);

    if (packs > 0) {
        generation_graph.add(generation / packs);
        mutation_graph.add(mutation / packs);
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

static void draw_bar(plotato::Graph& graph, Vec pos, double above, double fraction, plotato::Color colour)
{
    const double width = 2.4;
    const double height = .4;
    const double left = pos.x - width / 2;
    const double bottom = pos.y + above;

    graph.polygon({left, left, left + width, left + width}, {bottom, bottom + height, bottom + height, bottom}, filled(plotato::Color(60, 60, 60, 150)));

    const double right = left + width * clamp(fraction, 0, 1);

    if (right > left) {
        graph.polygon({left, left, right, right}, {bottom, bottom + height, bottom + height, bottom}, filled(colour));
    }
}

// A circle with a line out to its nose.
static void draw_animal(plotato::Graph& graph, const Body& body, plotato::Color fill)
{
    graph.circle(body.pos.x, body.pos.y, body.radius, filled(fill, plotato::Color(0, 0, 0), 1));

    const Vec nose = body.pos + heading_vector(body.heading) * (body.radius * 1.4);

    plotato::PlotStyle line;
    line.line_color = plotato::Color(0, 0, 0);
    line.line_width = 1.5;

    graph.plot({body.pos.x, nose.x}, {body.pos.y, nose.y}, line);
}

void HailMary::draw(plotato::Graph& graph)
{
    const int index = std::min(static_cast<int>(continents.size()) - 1, std::max(0, static_cast<int>(std::lround(shown.value())) - 1));
    const Continent& continent = *continents[static_cast<std::size_t>(index)];

    const double edge = HALF_SIZE;

    graph.polygon({-edge, -edge, edge, edge}, {-edge, edge, edge, -edge}, filled(plotato::Color(206, 222, 160)));

    // Not a wall: the map wraps round.
    plotato::PlotStyle wall;
    wall.line_color = plotato::Color(150, 165, 120);
    wall.line_width = 1;

    graph.plot({-edge, -edge, edge, edge, -edge}, {-edge, edge, edge, -edge, -edge}, wall);

    for (const Carcass& carcass : continent.carcasses) {
        const double left = clamp(carcass.energy / 100, .25, 1);

        graph.circle(carcass.pos.x, carcass.pos.y, carcass.radius, filled(plotato::Color(140, 20, 20, static_cast<uint8_t>(255 * left))));
    }

    for (const Prey& animal : continent.prey) {
        if (animal.bison) {
            draw_animal(graph, animal.body, plotato::Color(110, 75, 45));

            if (animal.health < BISON_HEALTH * .99) {
                draw_bar(graph, animal.body.pos, animal.body.radius + .6, animal.health / BISON_HEALTH, plotato::Color(210, 40, 40));
            }
        } else {
            draw_animal(graph, animal.body, plotato::Color(185, 170, 150));
        }
    }

    for (const Wolf& wolf : continent.wolves) {
        const Pack& pack = continent.packs.at(wolf.pack);
        const Body& body = wolf.body;

        draw_animal(graph, body, colour_of(pack.colour));

        if (wolf.biting) {
            const Vec forward = heading_vector(body.heading);
            const Vec side = {-forward.y * .5 * body.radius, forward.x * .5 * body.radius};
            const Vec mouth = body.pos + forward * body.radius;
            const Vec tip = mouth + forward * (.7 * body.radius);

            graph.polygon({mouth.x + side.x, tip.x, mouth.x - side.x}, {mouth.y + side.y, tip.y, mouth.y - side.y}, filled(plotato::Color(220, 0, 0)));
        }

        // A ring for whichever channel it calls loudest on, bigger the louder.
        const auto loudest = std::max_element(wolf.calls.begin(), wolf.calls.end());

        if (*loudest > .2f) {
            plotato::ShapeStyle ring = filled(plotato::Color(0, 0, 0, 0), CHANNEL_COLOURS[loudest - wolf.calls.begin()], 1.5);

            graph.circle(body.pos.x, body.pos.y, body.radius * (1.6 + 3 * *loudest), ring);
        }

        if (wolf.health < WOLF_HEALTH * .99) {
            draw_bar(graph, body.pos, body.radius + .6, wolf.health / WOLF_HEALTH, plotato::Color(210, 40, 40));
        }
    }

    plotato::FigureTextStyle caption;
    caption.font_size = 12;
    caption.absolute = false;
    caption.anchor = plotato::Anchor::BOTTOM_LEFT;
    caption.background_color = plotato::Color(0, 0, 0, 0);
    caption.outline_color = plotato::Color(0, 0, 0, 0);
    caption.outline_width = 0;

    char text[160];

    std::snprintf(text, sizeof(text), "Continent %d of %zu%s, %.1f hours", index + 1, continents.size(), muted(index) ? ", calls muted" : "", time / 3600);

    graph.add_text(-edge, edge * 1.02, text, caption);

    int bison = 0;

    for (const Prey& animal : continent.prey) {
        bison += animal.bison ? 1 : 0;
    }

    continent_field.set(index + 1);
    muted_field.set(muted(index) ? 1 : 0);
    wolves_field.set(static_cast<double>(continent.wolves.size()));
    packs_field.set(static_cast<double>(continent.packs.size()));
    bison_field.set(bison);
    hares_field.set(static_cast<double>(continent.prey.size()) - bison);
    generation_field.set(continent.highest_generation());
    bison_kills_field.set(static_cast<double>(continent.stats.bison_kills));
    hare_kills_field.set(static_cast<double>(continent.stats.hare_kills));
    migrations_field.set(static_cast<double>(migrations));
}

}

std::unique_ptr<sim::Project> sim::create_project(sim::Ui& ui, uint64_t seed)
{
    return std::make_unique<hail_mary::HailMary>(ui, seed);
}
