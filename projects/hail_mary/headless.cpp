// Runs hail_mary with no window, for experiments, and prints what happened as CSV.
//
//   hail_mary_headless --seed 1 --hours 12 --report 30 --set "Calls muted=1" --at 10 "Calls muted=2"
//
// --seed N           Which run. Default 1.
// --hours H          How long to run, in simulated hours. Default 4.
// --continents N     Default 8.
// --threads N        Default one per core. Makes no difference to what happens.
// --report M         Print a row for every continent every M simulated minutes. Default 30.
// --set "label=v"    Set a slider before starting. The label only has to be part of the slider's
//                    label, as long as it only matches one.
// --at H "label=v"   Set a slider once the run gets to H hours.
// --calls H1 H2      At the end, print how loud each channel was in each situation, between hours H1 and H2.
// --trace H S C F    From hour H, record S seconds of continent C to the JSON file F, every other
//                    step: every wolf, bison, hare, and carcass, and who is biting and calling.
// --draw N           Draw the world every N steps, on a graph that never goes on screen, to check
//                    that drawing works. Cycles through the continents. Needs GTK, but no window.
//
// Every row is for one continent over the last report, and then one for all continents where calls
// were heard to start with, and one for all where they were muted to start with, which are what to
// compare. Those two stay the same continents if the mute slider moves later on (with --at), and
// their muted column says how many of them are muted now.

#include "hail_mary.hpp"

#include <sim/ui.hpp>

#include <gtk/gtk.h>

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <string>
#include <vector>

using namespace hail_mary;

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

void set_control(sim::Ui& ui, const std::string& label, double value, bool announce = true)
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

    if (!announce) {
        return;
    }

    std::fprintf(stderr, "set \"%s\" to %g\n", found->options.label.c_str(), value);
}

double per_hour(double count, double hours) { return hours > 0 ? count / hours : 0; }

double ratio(double a, double b) { return b > 0 ? a / b : 0; }

void print_header()
{
    std::printf("hours,continent,muted,wolves,packs,bison,hares,bison_kills_h,hare_kills_h,wolves_per_kill,packs_per_kill,surround,front_bites,"
                "share_bison,share_hare,share_scavenged,call_volume,signal,approach,approach_steps,wolves_killed_h,gored_h,starved_h,daughters_h,refounded_h,"
                "generation,mutation,food_per_wolf_h,pack_size,weak_share,old_h,packmate_distance,approach_ch0,approach_ch1,approach_ch2,approach_noise\n");
}

// One row, from what changed over the last report, and what things look like right now.
void print_row(double hours, const char* name, int muted, const Stats& delta, double span_hours, int continents, double generation, double mutation)
{
    const double steps = span_hours * 3600 / DT;
    const double count = std::max(1, continents);

    double eaten = 0;

    for (double amount : delta.eaten) {
        eaten += amount;
    }

    double calls = 0;
    double context_steps = 0;

    for (int context = 0; context < CONTEXTS; context++) {
        for (int channel = 0; channel < CHANNELS; channel++) {
            calls += delta.calls[context][channel];
        }

        context_steps += static_cast<double>(delta.context_steps[context]);
    }

    std::printf("%.2f,%s,%d,%.1f,%.1f,%.1f,%.1f,%.2f,%.2f,%.2f,%.2f,%.3f,%.3f,%.3f,%.3f,%.3f,%.4f,%.4f,%.4f,%llu,%.2f,%.2f,%.2f,%.2f,%.2f,%.1f,%.4f,%.1f,%.2f,%.3f,%.2f,%.1f,%.4f,%.4f,%.4f,%.4f\n",
                hours, name, muted,
                delta.wolf_steps / steps / count, delta.pack_steps / steps / count, delta.bison_steps / steps / count, delta.hare_steps / steps / count,
                per_hour(static_cast<double>(delta.bison_kills), span_hours) / count, per_hour(static_cast<double>(delta.hare_kills), span_hours) / count,
                ratio(delta.kill_wolves, static_cast<double>(delta.bison_kills)), ratio(delta.kill_packs, static_cast<double>(delta.bison_kills)),
                ratio(delta.surround, static_cast<double>(delta.surround_kills)), ratio(static_cast<double>(delta.front_bites), static_cast<double>(delta.bites)),
                ratio(delta.eaten[HUNTED_BISON], eaten), ratio(delta.eaten[HUNTED_HARE], eaten), ratio(delta.eaten[OLD_AGE] + delta.eaten[DEAD_WOLF], eaten),
                ratio(calls, context_steps * CHANNELS), call_signal(delta), ratio(delta.approach, static_cast<double>(delta.approach_steps)),
                static_cast<unsigned long long>(delta.approach_steps),
                per_hour(static_cast<double>(delta.wolves_killed), span_hours) / count, per_hour(static_cast<double>(delta.wolves_gored), span_hours) / count,
                per_hour(static_cast<double>(delta.wolves_starved), span_hours) / count, per_hour(static_cast<double>(delta.daughters), span_hours) / count,
                per_hour(static_cast<double>(delta.refounded), span_hours) / count, generation, mutation,
                ratio(eaten, delta.wolf_steps * DT / 3600), ratio(delta.wolf_steps, delta.pack_steps),
                ratio(static_cast<double>(delta.weak_kills), static_cast<double>(delta.bison_kills)),
                per_hour(static_cast<double>(delta.wolves_old), span_hours) / count,
                ratio(delta.packmate_distance, static_cast<double>(delta.packmate_steps)),
                ratio(delta.channel_approach[0], static_cast<double>(delta.channel_approach_steps[0])),
                ratio(delta.channel_approach[1], static_cast<double>(delta.channel_approach_steps[1])),
                ratio(delta.channel_approach[2], static_cast<double>(delta.channel_approach_steps[2])),
                ratio(delta.noise_approach, static_cast<double>(delta.noise_approach_steps)));
}

const char* CONTEXT_NAMES[CONTEXTS] = {"at_bison", "eating", "sees_bison", "sees_hare", "sees_stranger", "alone", "with_pack"};

void print_calls(const char* group, const Stats& delta)
{
    for (int context = 0; context < CONTEXTS; context++) {
        std::printf("calls,%s,%s,%llu", group, CONTEXT_NAMES[context], static_cast<unsigned long long>(delta.context_steps[context]));

        for (int channel = 0; channel < CHANNELS; channel++) {
            std::printf(",%.4f", ratio(delta.calls[context][channel], static_cast<double>(delta.context_steps[context])));
        }

        std::printf("\n");
    }
}

}

int main(int argc, char** argv)
{
    uint64_t seed = 1;
    double hours = 4;
    int continent_count = DEFAULT_CONTINENTS;
    int thread_count = HailMary::default_threads();
    double report_minutes = 30;
    double call_from = -1;
    double call_to = -1;
    long draw_every = 0;

    double trace_from = -1;
    double trace_seconds = 0;
    int trace_continent = 0;
    std::string trace_path;

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
        } else if (arg == "--continents") {
            continent_count = std::atoi(next().c_str());
        } else if (arg == "--threads") {
            thread_count = std::atoi(next().c_str());
        } else if (arg == "--report") {
            report_minutes = std::atof(next().c_str());
        } else if (arg == "--trace") {
            trace_from = std::atof(next().c_str());
            trace_seconds = std::atof(next().c_str());
            trace_continent = std::atoi(next().c_str());
            trace_path = next();
        } else if (arg == "--draw") {
            draw_every = std::atol(next().c_str());
        } else if (arg == "--calls") {
            call_from = std::atof(next().c_str());
            call_to = std::atof(next().c_str());
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

    // Sliders only exist once the project has made them, so --set takes effect just after it is
    // made, before the first step. The one thing that misses is how many bison and hares there are
    // to begin with, which always starts from the defaults and then grows or shrinks from there.
    HailMary world(ui, seed, continent_count, thread_count);

    for (const Change& change : before) {
        set_control(ui, change.label, change.value);
    }

    if (draw_every > 0) {
        gtk_init(nullptr, nullptr);
        ui.create_buffers();
    }

    long draws = 0;

    std::ofstream trace;
    bool tracing = false;
    const long trace_start = static_cast<long>(trace_from * 3600 / DT + .5);
    const long trace_end = trace_start + static_cast<long>(trace_seconds / DT + .5);

    const double steps_per_report = report_minutes * 60 / DT;
    const long total_steps = static_cast<long>(hours * 3600 / DT + .5);
    const long call_start = static_cast<long>(call_from * 3600 / DT + .5);
    const long call_end = std::min(total_steps, static_cast<long>(call_to * 3600 / DT + .5));

    std::vector<Stats> last(world.continents.size());
    std::vector<Stats> at_call_start(world.continents.size());
    std::vector<Stats> at_call_end(world.continents.size());

    // Which group each continent is in, for good.
    std::vector<bool> started_muted(world.continents.size());

    for (std::size_t i = 0; i < world.continents.size(); i++) {
        started_muted[i] = world.muted(static_cast<int>(i));
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

        if (call_from >= 0 && step == call_start + 1) {
            for (std::size_t i = 0; i < world.continents.size(); i++) {
                at_call_start[i] = world.continents[i]->stats;
            }
        }

        world.step(now * 3600);

        if (call_from >= 0 && step == call_end) {
            for (std::size_t i = 0; i < world.continents.size(); i++) {
                at_call_end[i] = world.continents[i]->stats;
            }
        }

        if (trace_from >= 0 && step >= trace_start && step <= trace_end && (step - trace_start) % 2 == 0) {
            const Continent& continent = *world.continents.at(static_cast<std::size_t>(trace_continent));

            if (!tracing) {
                trace.open(trace_path);
                trace << "{\"size\":" << 2 * HALF_SIZE << ",\"dt\":" << 2 * DT << ",\"continent\":" << trace_continent
                      << ",\"hours\":" << now << ",\"muted\":" << (world.muted(trace_continent) ? "true" : "false") << ",\"frames\":[";
                tracing = true;
            } else {
                trace << ",";
            }

            char buffer[256];

            trace << "{\"w\":[";

            for (std::size_t i = 0; i < continent.wolves.size(); i++) {
                const Wolf& wolf = continent.wolves[i];
                const Colour& colour = continent.packs.at(wolf.pack).colour;

                std::snprintf(buffer, sizeof(buffer), "%s[%.1f,%.1f,%.2f,%d,%d,%.2f,%.2f,%.2f,%.0f,%.2f,%.2f,%.2f]", i ? "," : "", wolf.body.pos.x, wolf.body.pos.y,
                              wolf.body.heading, wolf.pack, wolf.biting ? 1 : 0, wolf.calls[0], wolf.calls[1], wolf.calls[2], wolf.health, colour[0], colour[1], colour[2]);
                trace << buffer;
            }

            trace << "],\"b\":[";

            bool first = true;

            for (const Prey& animal : continent.prey) {
                if (!animal.bison) {
                    continue;
                }

                std::snprintf(buffer, sizeof(buffer), "%s[%.1f,%.1f,%.2f,%.2f,%.0f]", first ? "" : ",", animal.body.pos.x, animal.body.pos.y, animal.body.heading,
                              animal.body.radius, animal.health);
                trace << buffer;
                first = false;
            }

            trace << "],\"h\":[";

            first = true;

            for (const Prey& animal : continent.prey) {
                if (animal.bison) {
                    continue;
                }

                std::snprintf(buffer, sizeof(buffer), "%s[%.1f,%.1f,%.2f]", first ? "" : ",", animal.body.pos.x, animal.body.pos.y, animal.body.heading);
                trace << buffer;
                first = false;
            }

            trace << "],\"c\":[";

            for (std::size_t i = 0; i < continent.carcasses.size(); i++) {
                const Carcass& carcass = continent.carcasses[i];

                std::snprintf(buffer, sizeof(buffer), "%s[%.1f,%.1f,%.1f,%.0f]", i ? "," : "", carcass.pos.x, carcass.pos.y, carcass.radius, carcass.energy);
                trace << buffer;
            }

            trace << "]}";

            if (step + 2 > trace_end) {
                trace << "]}\n";
                trace.close();
                std::fprintf(stderr, "wrote %s\n", trace_path.c_str());
            }
        }

        if (draw_every > 0 && step % draw_every == 0) {
            set_control(ui, "Continent shown", static_cast<double>(1 + draws % static_cast<long>(world.continents.size())), false);
            world.draw(*ui.world_buffer);
            draws++;

            // Thrown away, the way the window would have taken it.
            sim::PlotContents frame;
            ui.world_buffer->swap_contents(frame);
        }

        since_report++;

        if (since_report < steps_per_report && step != total_steps) {
            continue;
        }

        const double span = since_report * DT / 3600;

        since_report = 0;

        Stats heard;
        Stats muted;
        int heard_count = 0;
        int muted_count = 0;
        int heard_now_muted = 0;
        int muted_now_muted = 0;
        double heard_generation = 0;
        double muted_generation = 0;
        double heard_mutation = 0;
        double muted_mutation = 0;

        for (std::size_t i = 0; i < world.continents.size(); i++) {
            const Continent& continent = *world.continents[i];
            const Stats delta = continent.stats - last[i];
            const bool is_muted = started_muted[i];
            const bool now_muted = world.muted(static_cast<int>(i));

            last[i] = continent.stats;

            double generation = 0;
            double mutation = 0;

            for (const auto& entry : continent.packs) {
                generation += entry.second.genome->generation;
                mutation += entry.second.genome->mutation_size;
            }

            const double packs = std::max<std::size_t>(1, continent.packs.size());

            print_row(now, std::to_string(i).c_str(), now_muted ? 1 : 0, delta, span, 1, generation / packs, mutation / packs);

            (is_muted ? muted : heard) += delta;
            (is_muted ? muted_count : heard_count)++;
            (is_muted ? muted_generation : heard_generation) += generation / packs;
            (is_muted ? muted_mutation : heard_mutation) += mutation / packs;
            (is_muted ? muted_now_muted : heard_now_muted) += now_muted ? 1 : 0;
        }

        if (heard_count > 0) {
            print_row(now, "heard", heard_now_muted, heard, span, heard_count, heard_generation / heard_count, heard_mutation / heard_count);
        }

        if (muted_count > 0) {
            print_row(now, "muted", muted_now_muted, muted, span, muted_count, muted_generation / muted_count, muted_mutation / muted_count);
        }

        std::fflush(stdout);
    }

    if (call_from >= 0) {
        Stats heard;
        Stats muted;

        for (std::size_t i = 0; i < world.continents.size(); i++) {
            (started_muted[i] ? muted : heard) += at_call_end[i] - at_call_start[i];
        }

        print_calls("heard", heard);
        print_calls("muted", muted);
    }

    // Enough of the state to tell two runs apart, for checking that thread count changes nothing.
    double fingerprint = 0;

    for (const auto& continent : world.continents) {
        for (const Wolf& wolf : continent->wolves) {
            fingerprint += wolf.body.pos.x * 1.3 + wolf.body.pos.y * .7 + wolf.health;
        }

        for (const Prey& animal : continent->prey) {
            fingerprint += animal.body.pos.x * .3 - animal.body.pos.y * 1.1;
        }

        fingerprint += static_cast<double>(continent->stats.bison_kills * 7 + continent->stats.hare_kills * 3 + continent->stats.daughters * 11);
    }

    const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();

    std::fprintf(stderr, "fingerprint %.6f, %.1f s for %.2f simulated hours (%.0fx real time)\n", fingerprint, seconds, hours, hours * 3600 / std::max(1e-9, seconds));

    return 0;
}
