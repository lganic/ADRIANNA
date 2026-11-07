// Assembly of genotypes into graphs, parallel evaluation, and the CoDeepNEAT and
// DeepNEAT generation loops. Evaluation jobs get their seeds before any thread starts
// and write only to their own slot, so results do not depend on the thread count.

#include "codeepneat/codeepneat.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>
#include <set>
#include <sstream>
#include <stdexcept>
#include <thread>

namespace cdn {

// ============================================================================
// AssembledGenome I/O
// ============================================================================

void AssembledGenome::write(std::ostream& os) const {
  os.precision(17);
  os << "assembled " << fitness << ' ' << generation << ' ' << modules.size() << '\n';
  blueprint.write(os);
  for (const auto& kv : modules) {
    os << "module_for_species " << kv.first << '\n';
    kv.second.write(os);
  }
}

AssembledGenome AssembledGenome::read(std::istream& is) {
  AssembledGenome a;
  std::string word;
  size_t n = 0;
  is >> word;
  if (word != "assembled") throw std::runtime_error("AssembledGenome::read: bad header '" + word + "'");
  is >> a.fitness >> a.generation >> n;
  a.blueprint = Genome::read(is);
  for (size_t k = 0; k < n; ++k) {
    int sid = -1;
    is >> word >> sid;
    if (word != "module_for_species") throw std::runtime_error("AssembledGenome::read: expected module_for_species");
    a.modules[sid] = Genome::read(is);
  }
  return a;
}

void AssembledGenome::save(const std::string& path) const {
  std::ofstream f(path);
  if (!f) throw std::runtime_error("AssembledGenome::save: cannot open " + path);
  write(f);
}

AssembledGenome AssembledGenome::load(const std::string& path) {
  std::ifstream f(path);
  if (!f) throw std::runtime_error("AssembledGenome::load: cannot open " + path);
  return read(f);
}

std::string AssembledGenome::describe() const {
  std::ostringstream os;
  os << "Assembled network (fitness " << fitness << ", generation " << generation << ")\n";
  os << blueprint.describe();
  for (const auto& kv : modules) os << "species " << kv.first << " -> " << kv.second.describe();
  return os.str();
}

// ============================================================================
// Assembly
// ============================================================================

BuildOptions default_build_options(const HyperparamValues& globals) {
  BuildOptions o;
  o.weight_init = globals.s("weight_init", o.weight_init);
  return o;
}

namespace {

bool is_recurrent_type(const std::string& t, const std::vector<std::string>& types) {
  return std::find(types.begin(), types.end(), t) != types.end();
}

std::vector<int> mapped_preds(const Genome& g, int node, const std::map<int, int>& id_map) {
  std::vector<int> r;
  for (int p : g.enabled_predecessors(node)) {
    auto it = id_map.find(p);
    if (it != id_map.end()) r.push_back(it->second);
  }
  return r;
}

/// Adds the layer nodes of `module` behind `in_id`; returns the graph id of the
/// module's output.
int expand_module(NetworkGraph& g, const Genome& module, const std::string& prefix, int in_id, NodeOrigin origin,
                  const std::vector<std::string>& rec_types) {
  if (!module.has_input_output_path()) return in_id;  // degenerate module = identity
  std::map<int, int> m2g{{0, in_id}};
  for (int mid : module.active_nodes()) {
    const NodeGene& ng = module.nodes.at(mid);
    std::vector<int> preds = mapped_preds(module, mid, m2g);
    if (preds.empty()) continue;
    int id = g.add_layer(prefix + "/n" + std::to_string(mid), ng.hp.s("type", "dense"), ng.hp, preds,
                         ng.hp.s("merge", "concat"));
    NodeOrigin o = origin;
    o.role = "layer";
    o.module_node = mid;
    g.node(id).origin = o;
    m2g[mid] = id;
  }
  std::vector<int> outs = mapped_preds(module, module.output_id(), m2g);
  if (outs.empty()) return in_id;
  int out_id = g.add_identity(prefix + "/out", outs, module.out_merge);
  NodeOrigin o = origin;
  o.role = "module_out";
  g.node(out_id).origin = o;
  for (const auto& kv : module.cell_skips) {
    const ConnGene& s = kv.second;
    if (!s.enabled || !m2g.count(s.from) || !m2g.count(s.to) || s.from == 0 || s.to == 0) continue;
    if (!is_recurrent_type(module.nodes.at(s.from).hp.s("type"), rec_types) ||
        !is_recurrent_type(module.nodes.at(s.to).hp.s("type"), rec_types))
      continue;
    g.add_cell_skip(m2g[s.from], m2g[s.to]);
  }
  return out_id;
}

}  // namespace

NetworkGraph assemble_network(const AssembledGenome& ag, const IOSpec& io, const std::vector<std::string>& rec_types) {
  const Genome& bp = ag.blueprint;
  if (bp.num_inputs != static_cast<int>(io.inputs.size()))
    throw std::invalid_argument("assemble_network: genome has " + std::to_string(bp.num_inputs) +
                                " inputs but IOSpec has " + std::to_string(io.inputs.size()));
  if (io.outputs.empty()) throw std::invalid_argument("assemble_network: IOSpec has no outputs");
  NetworkGraph g;
  std::map<int, int> id_map;
  for (int k = 0; k < bp.num_inputs; ++k) id_map[k] = g.add_input("input:" + io.inputs[k].name, io.inputs[k].shape);

  if (bp.kind == GenomeKind::DeepNEAT || bp.kind == GenomeKind::Module) {
    for (int nid : bp.active_nodes()) {
      const NodeGene& ng = bp.nodes.at(nid);
      std::vector<int> preds = mapped_preds(bp, nid, id_map);
      if (preds.empty()) continue;
      int id = g.add_layer("n" + std::to_string(nid), ng.hp.s("type", "dense"), ng.hp, preds,
                           ng.hp.s("merge", "concat"));
      g.node(id).origin.module_node = nid;
      id_map[nid] = id;
    }
    for (const auto& kv : bp.cell_skips) {
      const ConnGene& s = kv.second;
      if (!s.enabled || !id_map.count(s.from) || !id_map.count(s.to) || bp.is_input(s.from) || bp.is_input(s.to))
        continue;
      if (!is_recurrent_type(bp.nodes.at(s.from).hp.s("type"), rec_types) ||
          !is_recurrent_type(bp.nodes.at(s.to).hp.s("type"), rec_types))
        continue;
      g.add_cell_skip(id_map[s.from], id_map[s.to]);
    }
  } else {
    for (int nid : bp.active_nodes()) {
      const NodeGene& bn = bp.nodes.at(nid);
      std::vector<int> preds = mapped_preds(bp, nid, id_map);
      if (preds.empty()) continue;
      auto mit = ag.modules.find(bn.species);
      if (mit == ag.modules.end())
        throw std::invalid_argument("assemble_network: no module for species " + std::to_string(bn.species));
      const Genome& module = mit->second;
      std::string prefix = "bp" + std::to_string(nid) + "/m" + std::to_string(module.id);
      NodeOrigin origin;
      origin.blueprint_node = nid;
      origin.species = bn.species;
      origin.module_id = module.id;
      int in_id = g.add_identity(prefix + "/in", preds, bn.hp.s("merge", "concat"));
      NodeOrigin oi = origin;
      oi.role = "module_in";
      g.node(in_id).origin = oi;
      id_map[nid] = expand_module(g, module, prefix, in_id, origin, rec_types);
    }
  }
  std::vector<int> body_in = mapped_preds(bp, bp.output_id(), id_map);
  if (body_in.empty()) throw std::invalid_argument("assemble_network: genome output is not connected");
  int body = g.add_identity("body", body_in, bp.out_merge);
  g.node(body).origin.role = "body";
  for (const auto& o : io.outputs) g.add_output("head:" + o.name, body, o.size, o.activation, o.init_scale);
  return g;
}

// ============================================================================
// Shared evaluation machinery
// ============================================================================

namespace {

struct Job {
  AssembledGenome genome;
  uint64_t seed = 0;
  EvaluationRecord rec;
};

void run_jobs(std::vector<Job>& jobs, const Evaluator& eval, int threads, const IOSpec& io,
              const std::vector<std::string>& rec_types,
              const std::function<BuildOptions(const HyperparamValues&)>& build_options, double failed,
              int generation) {
  std::atomic<size_t> next{0};
  auto worker = [&]() {
    for (;;) {
      size_t i = next.fetch_add(1);
      if (i >= jobs.size()) break;
      Job& job = jobs[i];
      job.rec.index = static_cast<int>(i);
      try {
        NetworkGraph graph = assemble_network(job.genome, io, rec_types);
        BuildOptions opt = build_options ? build_options(job.genome.globals()) : BuildOptions();
        Rng rng(job.seed);
        Network net(graph, opt, rng.next_u64());
        job.rec.num_parameters = net.num_parameters();
        EvalContext ctx{net, job.genome.globals(), rng, job.genome, generation, static_cast<int>(i)};
        double f = eval(ctx);
        if (!std::isfinite(f)) {
          job.rec.error = "non-finite fitness";
          f = failed;
        }
        job.rec.fitness = f;
      } catch (const std::exception& e) {
        job.rec.fitness = failed;
        job.rec.error = e.what();
      }
    }
  };
  int n = std::max(1, std::min<int>(threads, static_cast<int>(jobs.size())));
  if (n == 1) {
    worker();
  } else {
    std::vector<std::thread> pool;
    for (int t = 0; t < n; ++t) pool.emplace_back(worker);
    for (auto& t : pool) t.join();
  }
}

/// Fitness = mean of the samples. Genomes that were not part of any evaluated
/// network keep their previous fitness (elites), or else receive the mean of
/// their species (or of the population).
void assign_fitness(Population& pop) {
  double total = 0.0;
  int n = 0;
  std::map<int, std::pair<double, int>> by_species;
  for (auto& g : pop.genomes()) {
    if (g.fitness_samples.empty()) continue;
    double s = 0.0;
    for (double f : g.fitness_samples) s += f;
    g.fitness = s / g.fitness_samples.size();
    g.evaluated = true;
    total += g.fitness;
    ++n;
    auto& e = by_species[g.species];
    e.first += g.fitness;
    e.second += 1;
  }
  for (auto& g : pop.genomes()) {
    if (g.evaluated || g.has_prior_fitness) continue;
    auto it = by_species.find(g.species);
    if (it != by_species.end())
      g.fitness = it->second.first / it->second.second;
    else
      g.fitness = n ? total / n : 0.0;
  }
}

void fill_stats(GenerationStats& st, const std::vector<Job>& jobs) {
  st.evaluations = static_cast<int>(jobs.size());
  if (jobs.empty()) return;
  double sum = 0.0, params = 0.0;
  st.best_fitness = -std::numeric_limits<double>::infinity();
  st.worst_fitness = std::numeric_limits<double>::infinity();
  for (const auto& j : jobs) {
    st.best_fitness = std::max(st.best_fitness, j.rec.fitness);
    st.worst_fitness = std::min(st.worst_fitness, j.rec.fitness);
    sum += j.rec.fitness;
    params += static_cast<double>(j.rec.num_parameters);
    if (!j.rec.error.empty()) ++st.failures;
    st.records.push_back(j.rec);
  }
  st.mean_fitness = sum / jobs.size();
  st.mean_parameters = params / jobs.size();
}

}  // namespace

std::string GenerationStats::to_string() const {
  std::ostringstream os;
  os.setf(std::ios::fixed);
  os.precision(3);
  os << "gen " << generation << " | best " << best_fitness << " mean " << mean_fitness << " worst " << worst_fitness
     << " | best-ever " << best_ever << " | species m=" << module_species << " b=" << blueprint_species
     << " | params " << static_cast<long>(mean_parameters) << " | " << evaluations << " evals";
  if (failures) os << " (" << failures << " failed)";
  os.precision(1);
  os << " | " << seconds << "s";
  return os.str();
}

// ============================================================================
// CoDeepNEAT
// ============================================================================

CoDeepNEATConfig::CoDeepNEATConfig() {
  modules.size = 45;
  modules.speciation.target_species = 4;
  blueprints.size = 25;
  blueprints.speciation.target_species = 3;
  blueprints.mutation.node_hyperparams = 0.1;
}

CoDeepNEAT::CoDeepNEAT(CoDeepNEATConfig config) : cfg_(std::move(config)), rng_(cfg_.seed) {
  if (cfg_.io.inputs.empty() || cfg_.io.outputs.empty())
    throw std::invalid_argument("CoDeepNEAT: IOSpec needs at least one input and one output");
  if (cfg_.networks_per_generation < 1) throw std::invalid_argument("CoDeepNEAT: networks_per_generation < 1");
  GenomeOperators mod_ops(GenomeKind::Module, cfg_.module_node_space, HyperparamSpace(), cfg_.modules.mutation,
                          cfg_.modules.speciation, cfg_.recurrent_layer_types);
  modules_ = Population(cfg_.modules, mod_ops, 1);
  modules_.initialize(rng_);
  GenomeOperators bp_ops(GenomeKind::Blueprint, cfg_.blueprint_node_space, cfg_.global_space,
                         cfg_.blueprints.mutation, cfg_.blueprints.speciation, cfg_.recurrent_layer_types);
  blueprints_ = Population(cfg_.blueprints, bp_ops, static_cast<int>(cfg_.io.inputs.size()));
  blueprints_.initialize(rng_, modules_.species_ids());
}

const AssembledGenome& CoDeepNEAT::best() const {
  if (!has_best_) throw std::logic_error("CoDeepNEAT::best: no generation evaluated yet");
  return best_;
}

AssembledGenome CoDeepNEAT::sample_assembly(size_t blueprint_index, Rng& rng) const {
  AssembledGenome ag;
  ag.blueprint = blueprints_.genomes().at(blueprint_index);
  ag.generation = generation_;
  for (int nid : ag.blueprint.active_nodes()) {
    int sid = ag.blueprint.nodes.at(nid).species;
    if (ag.modules.count(sid)) continue;  // same species -> same module
    const Species* s = modules_.find_species(sid);
    if (!s || s->members.empty())
      throw std::logic_error("CoDeepNEAT: blueprint points to missing module species " + std::to_string(sid));
    ag.modules[sid] = modules_.genomes()[rng.choice(s->members)];
  }
  return ag;
}

NetworkGraph CoDeepNEAT::graph(const AssembledGenome& g) const {
  return assemble_network(g, cfg_.io, cfg_.recurrent_layer_types);
}

std::unique_ptr<Network> CoDeepNEAT::build(const AssembledGenome& g, uint64_t seed) const {
  BuildOptions opt = cfg_.build_options ? cfg_.build_options(g.globals()) : BuildOptions();
  return std::unique_ptr<Network>(new Network(graph(g), opt, seed));
}

void CoDeepNEAT::fix_species_pointers() {
  std::vector<int> ids = modules_.species_ids();
  for (auto& bp : blueprints_.genomes())
    for (auto& kv : bp.nodes)
      if (std::find(ids.begin(), ids.end(), kv.second.species) == ids.end()) kv.second.species = rng_.choice(ids);
}

GenerationStats CoDeepNEAT::step(const Evaluator& evaluator) {
  auto t0 = std::chrono::steady_clock::now();
  auto& bps = blueprints_.genomes();
  auto& mods = modules_.genomes();
  for (auto& g : bps) g.reset_evaluation();
  for (auto& g : mods) g.reset_evaluation();

  // 1. Assemble networks: blueprints round-robin in a shuffled order.
  std::vector<size_t> perm(bps.size());
  for (size_t i = 0; i < perm.size(); ++i) perm[i] = i;
  rng_.shuffle(perm);
  std::vector<Job> jobs(cfg_.networks_per_generation);
  for (size_t i = 0; i < jobs.size(); ++i) {
    jobs[i].genome = sample_assembly(perm[i % perm.size()], rng_);
    jobs[i].seed = rng_.next_u64();
    jobs[i].rec.blueprint_id = jobs[i].genome.blueprint.id;
    for (const auto& kv : jobs[i].genome.modules) jobs[i].rec.modules[kv.first] = kv.second.id;
  }

  // 2. Train / evaluate.
  run_jobs(jobs, evaluator, cfg_.num_threads, cfg_.io, cfg_.recurrent_layer_types, cfg_.build_options,
           cfg_.failed_fitness, generation_);

  // 3. Attribute fitness back to blueprints and modules (average).
  std::map<int, size_t> bp_index, mod_index;
  for (size_t i = 0; i < bps.size(); ++i) bp_index[bps[i].id] = i;
  for (size_t i = 0; i < mods.size(); ++i) mod_index[mods[i].id] = i;
  for (auto& job : jobs) {
    double f = job.rec.fitness;
    bps[bp_index.at(job.rec.blueprint_id)].fitness_samples.push_back(f);
    std::set<int> seen;
    for (const auto& kv : job.rec.modules)
      if (seen.insert(kv.second).second) mods[mod_index.at(kv.second)].fitness_samples.push_back(f);
    job.genome.fitness = f;
    if (!has_best_ || f > best_.fitness) {
      best_ = job.genome;
      best_.fitness = f;
      best_.generation = generation_;
      has_best_ = true;
    }
  }
  assign_fitness(blueprints_);
  assign_fitness(modules_);
  if (best_.generation == generation_) {
    // Refresh the champion's genome copies with their attributed (average) fitness.
    best_.blueprint.fitness = bps[bp_index.at(best_.blueprint.id)].fitness;
    for (auto& kv : best_.modules) kv.second.fitness = mods[mod_index.at(kv.second.id)].fitness;
  }

  GenerationStats st;
  st.generation = generation_;
  fill_stats(st, jobs);
  st.best_ever = best_.fitness;
  st.module_species = static_cast<int>(modules_.species().size());
  st.blueprint_species = static_cast<int>(blueprints_.species().size());

  // 4. Breed the next generation of modules, then blueprints (whose species
  //    pointers must refer to the new module species).
  modules_.reproduce(generation_, rng_);
  blueprints_.reproduce(generation_, rng_, modules_.species_ids());
  fix_species_pointers();
  ++generation_;

  st.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  if (cfg_.verbose) std::cout << st.to_string() << std::endl;
  return st;
}

std::vector<GenerationStats> CoDeepNEAT::run(int generations, const Evaluator& evaluator,
                                             const std::function<bool(const GenerationStats&)>& on_generation) {
  std::vector<GenerationStats> all;
  for (int g = 0; g < generations; ++g) {
    all.push_back(step(evaluator));
    if (on_generation && !on_generation(all.back())) break;
  }
  return all;
}

// ============================================================================
// DeepNEAT
// ============================================================================

DeepNEAT::DeepNEAT(DeepNEATConfig config) : cfg_(std::move(config)), rng_(cfg_.seed) {
  if (cfg_.io.inputs.empty() || cfg_.io.outputs.empty())
    throw std::invalid_argument("DeepNEAT: IOSpec needs at least one input and one output");
  GenomeOperators ops(GenomeKind::DeepNEAT, cfg_.node_space, cfg_.global_space, cfg_.population.mutation,
                      cfg_.population.speciation, cfg_.recurrent_layer_types);
  pop_ = Population(cfg_.population, ops, static_cast<int>(cfg_.io.inputs.size()));
  pop_.initialize(rng_);
}

const AssembledGenome& DeepNEAT::best() const {
  if (!has_best_) throw std::logic_error("DeepNEAT::best: no generation evaluated yet");
  return best_;
}

NetworkGraph DeepNEAT::graph(const AssembledGenome& g) const {
  return assemble_network(g, cfg_.io, cfg_.recurrent_layer_types);
}

std::unique_ptr<Network> DeepNEAT::build(const AssembledGenome& g, uint64_t seed) const {
  BuildOptions opt = cfg_.build_options ? cfg_.build_options(g.globals()) : BuildOptions();
  return std::unique_ptr<Network>(new Network(graph(g), opt, seed));
}

GenerationStats DeepNEAT::step(const Evaluator& evaluator) {
  auto t0 = std::chrono::steady_clock::now();
  auto& gs = pop_.genomes();
  for (auto& g : gs) g.reset_evaluation();
  int reps = std::max(1, cfg_.evaluations_per_genome);
  std::vector<Job> jobs;
  std::vector<size_t> owner;
  for (size_t i = 0; i < gs.size(); ++i)
    for (int r = 0; r < reps; ++r) {
      Job j;
      j.genome.blueprint = gs[i];
      j.genome.generation = generation_;
      j.seed = rng_.next_u64();
      j.rec.blueprint_id = gs[i].id;
      jobs.push_back(std::move(j));
      owner.push_back(i);
    }
  run_jobs(jobs, evaluator, cfg_.num_threads, cfg_.io, cfg_.recurrent_layer_types, cfg_.build_options,
           cfg_.failed_fitness, generation_);
  for (size_t k = 0; k < jobs.size(); ++k) gs[owner[k]].fitness_samples.push_back(jobs[k].rec.fitness);
  assign_fitness(pop_);
  for (const auto& g : gs) {
    if (!has_best_ || g.fitness > best_.fitness) {
      best_ = AssembledGenome();
      best_.blueprint = g;
      best_.fitness = g.fitness;
      best_.generation = generation_;
      has_best_ = true;
    }
  }
  GenerationStats st;
  st.generation = generation_;
  fill_stats(st, jobs);
  st.best_ever = best_.fitness;
  st.module_species = static_cast<int>(pop_.species().size());
  st.blueprint_species = 0;
  pop_.reproduce(generation_, rng_);
  ++generation_;
  st.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  if (cfg_.verbose) std::cout << st.to_string() << std::endl;
  return st;
}

std::vector<GenerationStats> DeepNEAT::run(int generations, const Evaluator& evaluator,
                                           const std::function<bool(const GenerationStats&)>& on_generation) {
  std::vector<GenerationStats> all;
  for (int g = 0; g < generations; ++g) {
    all.push_back(step(evaluator));
    if (on_generation && !on_generation(all.back())) break;
  }
  return all;
}

}  // namespace cdn
