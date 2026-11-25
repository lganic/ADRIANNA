// Runs food_web with no window, for experiments, and prints what happened as CSV.
//
//   food_web_headless --seed 1 --hours 48 --report 15 --set "Most grass=.2" --at 24 "wolves can see=20"
//
// --seed N           Which run. Default 1.
// --hours H          How long to run, in simulated hours. Default 4.
// --threads N        Default one per core. Makes no difference to what happens.
// --report M         Print a row for every continent, and one for the whole world, every M simulated minutes.
// --set "label=v"    Set a slider before starting, by any part of its label that only matches one.
// --at H "label=v"   Set a slider once the run gets to H hours.
// --draw N           Draw everything every N steps, without a window, to check that drawing works.
// --profile          At the end, print how long each part of a step took.
//
// Counts of animals and grass are averages over the report. Everything per hour is over the report too.

#include "food_web.hpp"

#include <sim/ui.hpp>

#include <gtk/gtk.h>

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

using namespace food_web;

namespace {

struct Change {
    double hours = 0;
    std::string label;
    double value = 0;
};

std::string lower(std::string text)
{
    std::transform(text.begin(), text.end(), text.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return text;
}

bool parse_setting(const std::string& text, std::string& label, double& value)
{
    const std::size_t equals = text.rfind('=');

    if (equals == std::string::npos) {
        return false;
    }

    label = text.substr(0, equals);
    value = std::atof(text.c_str() + equals + 1);

    return true;
}

void set_control(sim::Ui& ui, const std::string& label, double value)
{
    sim::ControlState* found = nullptr;
    int matches = 0;

    for (sim::ControlState& control : ui.controls) {
        if (lower(control.options.label).find(lower(label)) != std::string::npos) {
            found = &control;
            matches++;
        }
    }

    if (matches != 1) {
        std::fprintf(stderr, "\"%s\" matches %d sliders, not 1\n", label.c_str(), matches);
        std::exit(1);
    }

    found->value.store(value);

    std::fprintf(stderr, "set \"%s\" to %g\n", found->options.label.c_str(), value);
}

double ratio(double a, double b) { return b > 0 ? a / b : 0; }

void print_header()
{
    std::printf("hours,continent,bison,hares,wolves,grass,bison_born_h,bison_killed_h,bison_starved_h,bison_old_h,bison_arrived_h,"
                "hares_born_h,hares_killed_h,hares_starved_h,hares_arrived_h,wolf_food_per_wolf_h,share_bison,share_hare,share_scavenged,"
                "wolves_per_kill,weak_share,surround,pups_h,daughters_h,wolves_starved_h,wolves_gored_h,wolves_killed_h,wolves_old_h,refounded,"
                "bison_call_wolf,bison_call_none,hare_call_wolf,hare_call_none,wolf_call_bison,wolf_call_none,wolves_wet,bison_wet,hares_wet,calf_share,old_share,hungry_share\n");
}

void print_row(double hours, const std::string& name, const Tally& d, double span_hours, double grass_cells, double most)
{
    const double steps = span_hours * 3600 / DT;
    const double food = d.wolf_food_bison + d.wolf_food_hare + d.wolf_food_scavenged;
    const auto per_hour = [&](double count) { return span_hours > 0 ? count / span_hours : 0; };

    std::printf("%.2f,%s,%.1f,%.1f,%.1f,%.3f,%.1f,%.1f,%.1f,%.1f,%.1f,%.1f,%.1f,%.1f,%.1f,%.1f,%.3f,%.3f,%.3f,%.2f,%.3f,%.3f,%.1f,%.1f,%.1f,%.1f,%.1f,%.1f,%llu,"
                "%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f\n",
                hours, name.c_str(), d.bison_steps / steps, d.hare_steps / steps, d.wolf_steps / steps, ratio(d.grass / steps, grass_cells * most),
                per_hour(static_cast<double>(d.bison_born)), per_hour(static_cast<double>(d.bison_killed)), per_hour(static_cast<double>(d.bison_starved)),
                per_hour(static_cast<double>(d.bison_old)), per_hour(static_cast<double>(d.bison_arrived)), per_hour(static_cast<double>(d.hares_born)),
                per_hour(static_cast<double>(d.hares_killed)), per_hour(static_cast<double>(d.hares_starved)), per_hour(static_cast<double>(d.hares_arrived)),
                ratio(food, d.wolf_steps * DT / 3600), ratio(d.wolf_food_bison, food), ratio(d.wolf_food_hare, food), ratio(d.wolf_food_scavenged, food),
                ratio(d.kill_wolves, static_cast<double>(d.bison_killed)), ratio(static_cast<double>(d.weak_killed), static_cast<double>(d.bison_killed)),
                ratio(d.surround, static_cast<double>(d.surround_kills)), per_hour(static_cast<double>(d.pups)), per_hour(static_cast<double>(d.daughters)),
                per_hour(static_cast<double>(d.wolves_starved)), per_hour(static_cast<double>(d.wolves_gored)), per_hour(static_cast<double>(d.wolves_killed)),
                per_hour(static_cast<double>(d.wolves_old)), static_cast<unsigned long long>(d.refounded), ratio(d.bison_calls[1], d.bison_call_steps[1]),
                ratio(d.bison_calls[0], d.bison_call_steps[0]), ratio(d.hare_calls[1], d.hare_call_steps[1]), ratio(d.hare_calls[0], d.hare_call_steps[0]),
                ratio(d.wolf_calls[1], d.wolf_call_steps[1]), ratio(d.wolf_calls[0], d.wolf_call_steps[0]), ratio(d.wolf_wet_steps, d.wolf_steps),
                ratio(d.bison_wet_steps, d.bison_steps), ratio(d.hare_wet_steps, d.hare_steps),
                ratio(static_cast<double>(d.calves_killed), static_cast<double>(d.bison_killed)), ratio(static_cast<double>(d.old_killed), static_cast<double>(d.bison_killed)),
                ratio(static_cast<double>(d.hungry_killed), static_cast<double>(d.bison_killed)));
}

}

int main(int argc, char** argv)
{
    uint64_t seed = 1;
    double hours = 4;
    int thread_count = FoodWeb::default_threads();
    double report_minutes = 30;
    bool profile = false;
    long draw_every = 0;

    std::vector<Change> before;
    std::vector<Change> later;

    for (int i = 1; i < argc; i++) {
        const std::string arg = argv[i];

        const auto next = [&]() -> std::string {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "%s needs a value\n", arg.c_str());
                std::exit(1);
            }

            return argv[++i];
        };

        Change change;

        if (arg == "--seed") {
            seed = std::strtoull(next().c_str(), nullptr, 10);
        } else if (arg == "--hours") {
            hours = std::atof(next().c_str());
        } else if (arg == "--threads") {
            thread_count = std::atoi(next().c_str());
        } else if (arg == "--draw") {
            draw_every = std::atol(next().c_str());
        } else if (arg == "--profile") {
            profile = true;
        } else if (arg == "--report") {
            report_minutes = std::atof(next().c_str());
        } else if (arg == "--set") {
            if (!parse_setting(next(), change.label, change.value)) {
                std::fprintf(stderr, "--set needs label=value\n");
                return 1;
            }

            before.push_back(change);
        } else if (arg == "--at") {
            change.hours = std::atof(next().c_str());

            if (!parse_setting(next(), change.label, change.value)) {
                std::fprintf(stderr, "--at needs hours and label=value\n");
                return 1;
            }

            later.push_back(change);
        } else {
            std::fprintf(stderr, "Unknown argument %s\n", arg.c_str());
            return 1;
        }
    }

    sim::Ui ui;

    // Settings that matter from the very start (the founders, and how much grass there is to begin
    // with) come from the defaults, since the sliders only exist once the project has made them.
    FoodWeb project(ui, seed, thread_count);

    for (const Change& change : before) {
        set_control(ui, change.label, change.value);
    }

    World& world = *project.world;

    world.profiling = profile;

    // Drawing every so often on graphs that never go on screen, to check that drawing works.
    if (draw_every > 0) {
        gtk_init(nullptr, nullptr);
        ui.create_buffers();
    }

    for (int land = 0; land < world.continents(); land++) {
        std::fprintf(stderr, "landmass %d: %.0f square meters\n", land, world.terrain.landmass_area(land));
    }

    const long steps_per_report = static_cast<long>(report_minutes * 60 / DT + .5);
    const long total_steps = static_cast<long>(hours * 3600 / DT + .5);

    std::vector<Tally> last(world.tallies.size());

    std::vector<double> cells(world.tallies.size(), 0);

    for (int land = 0; land < world.continents(); land++) {
        cells[static_cast<std::size_t>(land)] = world.terrain.landmass_area(land) / (GRASS_CELL * GRASS_CELL);
    }

    print_header();

    const auto started = std::chrono::steady_clock::now();

    long since_report = 0;

    for (long step = 1; step <= total_steps; step++) {
        const double now = step * DT / 3600;

        for (Change& change : later) {
            if (change.hours >= 0 && now >= change.hours) {
                set_control(ui, change.label, change.value);
                change.hours = -1;
            }
        }

        project.step(now * 3600);

        if (draw_every > 0 && step % draw_every == 0) {
            project.draw(*ui.world_buffer);

            // Thrown away, the way the window would have taken them.
            sim::PlotContents frame;
            ui.world_buffer->swap_contents(frame);

            for (sim::LiveGraphState& graph : ui.live_graphs) {
                sim::PlotContents live;
                graph.buffer->swap_contents(live);
            }
        }

        since_report++;

        if (since_report < steps_per_report && step != total_steps) {
            continue;
        }

        const double span = since_report * DT / 3600;

        since_report = 0;

        const Settings settings = project.read_settings();
        const double most = settings.grass_max * GRASS_CELL * GRASS_CELL;

        Tally everything;
        double all_cells = 0;

        for (std::size_t i = 0; i < world.tallies.size(); i++) {
            const Tally delta = world.tallies[i] - last[i];

            last[i] = world.tallies[i];
            everything += delta;
            all_cells += cells[i];

            // Islands and the sea are only in the world total.
            if (static_cast<int>(i) < world.continents() && cells[i] >= .25 * cells[0]) {
                print_row(now, std::to_string(i), delta, span, cells[i], most);
            }
        }

        print_row(now, "world", everything, span, all_cells, most);

        std::fflush(stdout);
    }

    double fingerprint = 0;

    for (const Wolf& wolf : world.wolves) {
        fingerprint += wolf.body.pos.x * 1.3 + wolf.body.pos.y * .7 + wolf.health;
    }

    for (const Prey& animal : world.prey) {
        fingerprint += animal.body.pos.x * .3 - animal.body.pos.y * 1.1 + animal.energy;
    }

    const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();

    if (profile) {
        const char* parts[] = {"index", "think", "record", "move", "index again", "fight", "eat", "upkeep", "lifecycle", "rescue", "grass", "listen"};

        for (std::size_t i = 0; i < world.profile.size(); i++) {
            std::fprintf(stderr, "%12s %6.2f s %5.1f%%\n", parts[i], world.profile[i], 100 * world.profile[i] / seconds);
        }
    }

    std::fprintf(stderr, "fingerprint %.6f, %.1f s for %.2f simulated hours (%.0fx real time), %d landmasses\n", fingerprint, seconds, hours,
                 hours * 3600 / std::max(1e-9, seconds), world.continents());

    return 0;
}
