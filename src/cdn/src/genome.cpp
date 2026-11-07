// Genome queries and text I/O, innovation tracking, and the NEAT operators
// (mutation, crossover, compatibility distance). All operators keep the connection
// genes acyclic, so re-enabling a gene or crossing over can never create a cycle.

#include "codeepneat/genome.hpp"

#include <algorithm>
#include <cmath>
#include <istream>
#include <limits>
#include <ostream>
#include <queue>
#include <set>
#include <sstream>
#include <stdexcept>

namespace cdn {

const char* to_string(GenomeKind k) {
  switch (k) {
    case GenomeKind::Module: return "module";
    case GenomeKind::Blueprint: return "blueprint";
    case GenomeKind::DeepNEAT: return "deepneat";
  }
  return "?";
}

static GenomeKind kind_from_string(const std::string& s) {
  if (s == "module") return GenomeKind::Module;
  if (s == "blueprint") return GenomeKind::Blueprint;
  if (s == "deepneat") return GenomeKind::DeepNEAT;
  throw std::runtime_error("unknown genome kind '" + s + "'");
}

// ============================================================================
// Genome queries
// ============================================================================

const ConnGene* Genome::find_conn(int from, int to) const {
  for (const auto& kv : conns)
    if (kv.second.from == from && kv.second.to == to) return &kv.second;
  return nullptr;
}
ConnGene* Genome::find_conn(int from, int to) {
  for (auto& kv : conns)
    if (kv.second.from == from && kv.second.to == to) return &kv.second;
  return nullptr;
}
const ConnGene* Genome::find_cell_skip(int from, int to) const {
  for (const auto& kv : cell_skips)
    if (kv.second.from == from && kv.second.to == to) return &kv.second;
  return nullptr;
}
ConnGene* Genome::find_cell_skip(int from, int to) {
  for (auto& kv : cell_skips)
    if (kv.second.from == from && kv.second.to == to) return &kv.second;
  return nullptr;
}

bool Genome::would_create_cycle(int from, int to) const {
  if (from == to) return true;
  if (is_input(from) || is_output(to)) return false;
  // Is `from` reachable from `to`?
  std::set<int> seen{to};
  std::vector<int> stack{to};
  while (!stack.empty()) {
    int v = stack.back();
    stack.pop_back();
    for (const auto& kv : conns) {
      if (kv.second.from != v) continue;
      int w = kv.second.to;
      if (w == from) return true;
      if (seen.insert(w).second) stack.push_back(w);
    }
  }
  return false;
}

static std::set<int> forward_reach(const Genome& g) {
  std::set<int> seen;
  std::vector<int> stack;
  for (int i = 0; i < g.num_inputs; ++i) {
    seen.insert(i);
    stack.push_back(i);
  }
  while (!stack.empty()) {
    int v = stack.back();
    stack.pop_back();
    for (const auto& kv : g.conns)
      if (kv.second.enabled && kv.second.from == v && seen.insert(kv.second.to).second) stack.push_back(kv.second.to);
  }
  return seen;
}

static std::set<int> backward_reach(const Genome& g) {
  std::set<int> seen{g.output_id()};
  std::vector<int> stack{g.output_id()};
  while (!stack.empty()) {
    int v = stack.back();
    stack.pop_back();
    for (const auto& kv : g.conns)
      if (kv.second.enabled && kv.second.to == v && seen.insert(kv.second.from).second) stack.push_back(kv.second.from);
  }
  return seen;
}

bool Genome::has_input_output_path() const { return forward_reach(*this).count(output_id()) != 0; }

std::vector<int> Genome::active_nodes() const {
  std::set<int> f = forward_reach(*this), b = backward_reach(*this);
  std::set<int> active;
  for (const auto& kv : nodes)
    if (f.count(kv.first) && b.count(kv.first)) active.insert(kv.first);
  std::map<int, int> indeg;
  for (int v : active) indeg[v] = 0;
  for (const auto& kv : conns) {
    const auto& c = kv.second;
    if (c.enabled && active.count(c.to) && active.count(c.from)) ++indeg[c.to];
  }
  std::priority_queue<int, std::vector<int>, std::greater<int>> q;
  for (const auto& kv : indeg)
    if (kv.second == 0) q.push(kv.first);
  std::vector<int> order;
  while (!q.empty()) {
    int v = q.top();
    q.pop();
    order.push_back(v);
    for (const auto& kv : conns) {
      const auto& c = kv.second;
      if (c.enabled && c.from == v && active.count(c.to))
        if (--indeg[c.to] == 0) q.push(c.to);
    }
  }
  if (order.size() != active.size()) throw std::logic_error("genome " + std::to_string(id) + " contains a cycle");
  return order;
}

std::vector<int> Genome::enabled_predecessors(int node) const {
  std::vector<int> r;
  for (const auto& kv : conns)
    if (kv.second.enabled && kv.second.to == node) r.push_back(kv.second.from);
  return r;
}

size_t Genome::enabled_conn_count() const {
  size_t n = 0;
  for (const auto& kv : conns) n += kv.second.enabled ? 1 : 0;
  return n;
}

bool Genome::validate(std::string* why) const {
  auto fail = [&](const std::string& m) {
    if (why) *why = m;
    return false;
  };
  if (num_inputs < 1) return fail("num_inputs < 1");
  for (const auto& kv : nodes) {
    if (kv.first != kv.second.id) return fail("node key/id mismatch");
    if (kv.first <= num_inputs) return fail("hidden node id collides with virtual node");
    if (kind == GenomeKind::Blueprint && kv.second.species < 0) return fail("blueprint node without species");
  }
  std::set<std::pair<int, int>> pairs;
  for (const auto& kv : conns) {
    const auto& c = kv.second;
    if (kv.first != c.innovation) return fail("conn key/innovation mismatch");
    if (!(is_input(c.from) || is_hidden(c.from))) return fail("conn from invalid node " + std::to_string(c.from));
    if (!(is_output(c.to) || is_hidden(c.to))) return fail("conn to invalid node " + std::to_string(c.to));
    if (!pairs.insert({c.from, c.to}).second) return fail("duplicate connection");
  }
  // Acyclic over all genes.
  std::map<int, int> indeg;
  for (int i = 0; i <= num_inputs; ++i) indeg[i] = 0;
  for (const auto& kv : nodes) indeg[kv.first] = 0;
  for (const auto& kv : conns) ++indeg[kv.second.to];
  std::vector<int> q;
  for (const auto& kv : indeg)
    if (kv.second == 0) q.push_back(kv.first);
  size_t seen = 0;
  while (!q.empty()) {
    int v = q.back();
    q.pop_back();
    ++seen;
    for (const auto& kv : conns)
      if (kv.second.from == v && --indeg[kv.second.to] == 0) q.push_back(kv.second.to);
  }
  if (seen != indeg.size()) return fail("cycle in connection genes");
  std::set<std::pair<int, int>> spairs;
  for (const auto& kv : cell_skips) {
    const auto& c = kv.second;
    if (!is_hidden(c.from) || !is_hidden(c.to) || c.from == c.to) return fail("invalid cell skip");
    if (!spairs.insert({c.from, c.to}).second) return fail("duplicate cell skip");
  }
  if (out_merge != "concat" && out_merge != "sum") return fail("invalid out_merge");
  return true;
}

void Genome::reset_evaluation() {
  evaluated = false;
  fitness_samples.clear();
}

std::string Genome::describe() const {
  std::ostringstream os;
  os << to_string(kind) << " #" << id << " (species " << species << ", fitness " << fitness << "): " << nodes.size()
     << " nodes, " << enabled_conn_count() << "/" << conns.size() << " conns";
  if (!cell_skips.empty()) os << ", " << cell_skips.size() << " cell skips";
  os << "\n";
  for (const auto& kv : nodes) {
    os << "  node " << kv.first;
    if (kind == GenomeKind::Blueprint) os << " -> species " << kv.second.species;
    os << " [" << kv.second.hp.to_string() << "]\n";
  }
  for (const auto& kv : conns) {
    const auto& c = kv.second;
    auto nm = [&](int n) {
      if (is_input(n)) return std::string("in") + (num_inputs > 1 ? std::to_string(n) : "");
      if (is_output(n)) return std::string("out");
      return std::to_string(n);
    };
    os << "  conn " << nm(c.from) << " -> " << nm(c.to) << (c.enabled ? "" : " (disabled)") << "\n";
  }
  for (const auto& kv : cell_skips)
    os << "  cell-skip " << kv.second.from << " => " << kv.second.to << (kv.second.enabled ? "" : " (disabled)")
       << "\n";
  if (!globals.empty()) os << "  globals [" << globals.to_string() << "]\n";
  return os.str();
}

void Genome::write(std::ostream& os) const {
  os.precision(17);
  os << "genome " << id << ' ' << to_string(kind) << ' ' << num_inputs << ' ' << out_merge << ' ' << fitness << ' '
     << species << '\n';
  os << "globals " << globals.serialize() << '\n';
  os << "nodes " << nodes.size() << '\n';
  for (const auto& kv : nodes)
    os << "node " << kv.first << ' ' << kv.second.species << ' ' << kv.second.hp.serialize() << '\n';
  os << "conns " << conns.size() << '\n';
  for (const auto& kv : conns)
    os << "conn " << kv.first << ' ' << kv.second.from << ' ' << kv.second.to << ' ' << (kv.second.enabled ? 1 : 0)
       << '\n';
  os << "skips " << cell_skips.size() << '\n';
  for (const auto& kv : cell_skips)
    os << "skip " << kv.first << ' ' << kv.second.from << ' ' << kv.second.to << ' ' << (kv.second.enabled ? 1 : 0)
       << '\n';
  os << "end\n";
}

Genome Genome::read(std::istream& is) {
  auto expect = [&](const std::string& word) {
    std::string w;
    is >> w;
    if (w != word) throw std::runtime_error("Genome::read: expected '" + word + "', got '" + w + "'");
  };
  Genome g;
  std::string kind, tok;
  expect("genome");
  is >> g.id >> kind >> g.num_inputs >> g.out_merge >> g.fitness >> g.species;
  g.kind = kind_from_string(kind);
  expect("globals");
  is >> tok;
  g.globals = HyperparamValues::deserialize(tok);
  size_t n = 0;
  expect("nodes");
  is >> n;
  for (size_t k = 0; k < n; ++k) {
    expect("node");
    NodeGene ng;
    is >> ng.id >> ng.species >> tok;
    ng.hp = HyperparamValues::deserialize(tok);
    g.nodes[ng.id] = ng;
  }
  expect("conns");
  is >> n;
  for (size_t k = 0; k < n; ++k) {
    expect("conn");
    ConnGene c;
    int en = 1;
    is >> c.innovation >> c.from >> c.to >> en;
    c.enabled = en != 0;
    g.conns[c.innovation] = c;
  }
  expect("skips");
  is >> n;
  for (size_t k = 0; k < n; ++k) {
    expect("skip");
    ConnGene c;
    int en = 1;
    is >> c.innovation >> c.from >> c.to >> en;
    c.enabled = en != 0;
    g.cell_skips[c.innovation] = c;
  }
  expect("end");
  if (!is) throw std::runtime_error("Genome::read: truncated input");
  std::string why;
  if (!g.validate(&why)) throw std::runtime_error("Genome::read: invalid genome: " + why);
  g.evaluated = false;
  g.has_prior_fitness = true;
  return g;
}

// ============================================================================
// InnovationTracker
// ============================================================================

InnovationTracker::InnovationTracker(int num_inputs) : initial_node_(num_inputs + 1), next_node_(num_inputs + 2) {}

int InnovationTracker::connection(int from, int to) {
  auto key = std::make_pair(from, to);
  auto it = conn_.find(key);
  if (it != conn_.end()) return it->second;
  int id = next_innov_++;
  conn_[key] = id;
  return id;
}

int InnovationTracker::cell_skip(int from, int to) {
  auto key = std::make_pair(from, to);
  auto it = skip_.find(key);
  if (it != skip_.end()) return it->second;
  int id = next_innov_++;
  skip_[key] = id;
  return id;
}

int InnovationTracker::split_node(int conn_innovation, const Genome& g) {
  auto& ids = split_[conn_innovation];
  for (int id : ids)
    if (!g.nodes.count(id)) return id;
  int id = next_node_++;
  ids.push_back(id);
  return id;
}

// ============================================================================
// GenomeOperators
// ============================================================================

GenomeOperators::GenomeOperators(GenomeKind kind, HyperparamSpace node_space, HyperparamSpace global_space,
                                 MutationParams mutation, SpeciationParams speciation,
                                 std::vector<std::string> recurrent_types)
    : kind_(kind),
      node_space_(std::move(node_space)),
      global_space_(std::move(global_space)),
      mutation_(mutation),
      speciation_(speciation),
      recurrent_types_(std::move(recurrent_types)) {}

bool GenomeOperators::is_recurrent_node(const NodeGene& n) const {
  if (kind_ == GenomeKind::Blueprint) return false;
  std::string t = n.hp.s("type");
  return std::find(recurrent_types_.begin(), recurrent_types_.end(), t) != recurrent_types_.end();
}

NodeGene GenomeOperators::make_node(int id, Rng& rng, const std::vector<int>& species_choices) const {
  NodeGene n;
  n.id = id;
  n.hp = node_space_.sample(rng);
  if (kind_ == GenomeKind::Blueprint) {
    if (species_choices.empty()) throw std::logic_error("blueprint node needs at least one module species");
    n.species = rng.choice(species_choices);
  }
  return n;
}

Genome GenomeOperators::create_initial(int num_inputs, InnovationTracker& innov, Rng& rng,
                                       const std::vector<int>& species_choices) const {
  Genome g(kind_, num_inputs);
  int h = innov.initial_node();
  g.nodes[h] = make_node(h, rng, species_choices);
  for (int i = 0; i < num_inputs; ++i) {
    int c = innov.connection(i, h);
    g.conns[c] = ConnGene{c, i, h, true};
  }
  int c = innov.connection(h, g.output_id());
  g.conns[c] = ConnGene{c, h, g.output_id(), true};
  if (kind_ != GenomeKind::Module) g.globals = global_space_.sample(rng);
  for (int k = 0; k < mutation_.initial_structural_mutations; ++k) {
    if (rng.bernoulli(0.5))
      mutate_add_node(g, innov, rng, species_choices);
    else
      mutate_add_connection(g, innov, rng);
  }
  return g;
}

bool GenomeOperators::mutate_add_node(Genome& g, InnovationTracker& innov, Rng& rng,
                                      const std::vector<int>& species_choices) const {
  if (static_cast<int>(g.nodes.size()) >= mutation_.max_nodes) return false;
  std::vector<int> enabled;
  for (const auto& kv : g.conns)
    if (kv.second.enabled) enabled.push_back(kv.first);
  if (enabled.empty()) return false;
  ConnGene& old = g.conns.at(rng.choice(enabled));
  int nid = innov.split_node(old.innovation, g);
  old.enabled = false;
  g.nodes[nid] = make_node(nid, rng, species_choices);
  int c1 = innov.connection(old.from, nid);
  int c2 = innov.connection(nid, old.to);
  g.conns[c1] = ConnGene{c1, old.from, nid, true};
  g.conns[c2] = ConnGene{c2, nid, old.to, true};
  return true;
}

bool GenomeOperators::mutate_add_connection(Genome& g, InnovationTracker& innov, Rng& rng) const {
  std::vector<int> sources, targets;
  for (int i = 0; i < g.num_inputs; ++i) sources.push_back(i);
  for (const auto& kv : g.nodes) {
    sources.push_back(kv.first);
    targets.push_back(kv.first);
  }
  targets.push_back(g.output_id());
  for (int attempt = 0; attempt < 32; ++attempt) {
    int a = rng.choice(sources), b = rng.choice(targets);
    if (a == b) continue;
    if (ConnGene* existing = g.find_conn(a, b)) {
      if (!existing->enabled) {
        existing->enabled = true;
        return true;
      }
      continue;
    }
    if (g.would_create_cycle(a, b)) continue;
    int c = innov.connection(a, b);
    g.conns[c] = ConnGene{c, a, b, true};
    return true;
  }
  return false;
}

bool GenomeOperators::mutate_toggle_connection(Genome& g, Rng& rng) const {
  if (g.conns.empty()) return false;
  std::vector<int> keys;
  for (const auto& kv : g.conns) keys.push_back(kv.first);
  ConnGene& c = g.conns.at(rng.choice(keys));
  if (!c.enabled) {
    c.enabled = true;
    return true;
  }
  c.enabled = false;
  if (!g.has_input_output_path()) {
    c.enabled = true;  // never disconnect the genome entirely
    return false;
  }
  return true;
}

bool GenomeOperators::mutate_add_cell_skip(Genome& g, InnovationTracker& innov, Rng& rng) const {
  std::vector<int> rec;
  for (const auto& kv : g.nodes)
    if (is_recurrent_node(kv.second)) rec.push_back(kv.first);
  if (rec.size() < 2) return false;
  for (int attempt = 0; attempt < 8; ++attempt) {
    int a = rng.choice(rec), b = rng.choice(rec);
    if (a == b) continue;
    if (ConnGene* s = g.find_cell_skip(a, b)) {
      if (!s->enabled) {
        s->enabled = true;
        return true;
      }
      continue;
    }
    int c = innov.cell_skip(a, b);
    g.cell_skips[c] = ConnGene{c, a, b, true};
    return true;
  }
  return false;
}

bool GenomeOperators::mutate_toggle_cell_skip(Genome& g, Rng& rng) const {
  if (g.cell_skips.empty()) return false;
  std::vector<int> keys;
  for (const auto& kv : g.cell_skips) keys.push_back(kv.first);
  ConnGene& c = g.cell_skips.at(rng.choice(keys));
  c.enabled = !c.enabled;
  return true;
}

bool GenomeOperators::mutate_node_hyperparams(Genome& g, Rng& rng) const {
  bool changed = false;
  for (auto& kv : g.nodes)
    if (rng.bernoulli(mutation_.node_hyperparams)) {
      node_space_.complete(kv.second.hp, rng);
      node_space_.mutate(kv.second.hp, rng, mutation_.hyperparam_rate);
      changed = true;
    }
  return changed;
}

bool GenomeOperators::mutate_species_pointers(Genome& g, Rng& rng, const std::vector<int>& species_choices) const {
  if (kind_ != GenomeKind::Blueprint || species_choices.size() < 2) return false;
  bool changed = false;
  for (auto& kv : g.nodes)
    if (rng.bernoulli(mutation_.change_species)) {
      int old = kv.second.species;
      std::vector<int> others;
      for (int s : species_choices)
        if (s != old) others.push_back(s);
      if (others.empty()) continue;
      kv.second.species = rng.choice(others);
      changed = true;
    }
  return changed;
}

bool GenomeOperators::mutate_globals(Genome& g, Rng& rng) const {
  if (kind_ == GenomeKind::Module || global_space_.empty()) return false;
  global_space_.complete(g.globals, rng);
  global_space_.mutate(g.globals, rng, mutation_.hyperparam_rate);
  return true;
}

void GenomeOperators::mutate(Genome& g, InnovationTracker& innov, Rng& rng,
                             const std::vector<int>& species_choices) const {
  if (rng.bernoulli(mutation_.add_node)) mutate_add_node(g, innov, rng, species_choices);
  if (rng.bernoulli(mutation_.add_connection)) mutate_add_connection(g, innov, rng);
  if (rng.bernoulli(mutation_.toggle_connection)) mutate_toggle_connection(g, rng);
  if (kind_ != GenomeKind::Blueprint) {
    if (rng.bernoulli(mutation_.add_cell_skip)) mutate_add_cell_skip(g, innov, rng);
    if (rng.bernoulli(mutation_.toggle_cell_skip)) mutate_toggle_cell_skip(g, rng);
  }
  mutate_node_hyperparams(g, rng);
  if (kind_ == GenomeKind::Blueprint) {
    mutate_species_pointers(g, rng, species_choices);
    // Pointers to module species that went extinct are re-targeted.
    if (!species_choices.empty())
      for (auto& kv : g.nodes)
        if (std::find(species_choices.begin(), species_choices.end(), kv.second.species) == species_choices.end())
          kv.second.species = rng.choice(species_choices);
  }
  if (kind_ != GenomeKind::Module && rng.bernoulli(mutation_.globals)) mutate_globals(g, rng);
  if (rng.bernoulli(mutation_.out_merge)) g.out_merge = g.out_merge == "concat" ? "sum" : "concat";
}

void GenomeOperators::repair(Genome& g, Rng& rng) const {
  while (!g.has_input_output_path()) {
    std::vector<int> disabled;
    for (const auto& kv : g.conns)
      if (!kv.second.enabled) disabled.push_back(kv.first);
    if (disabled.empty()) break;
    g.conns.at(rng.choice(disabled)).enabled = true;
  }
}

Genome GenomeOperators::crossover(const Genome& fitter, const Genome& other, Rng& rng) const {
  Genome child = fitter;
  child.reset_evaluation();
  child.has_prior_fitness = false;
  auto cross_gene = [&](ConnGene& c, const ConnGene* o) {
    if (!o) return;
    if (!c.enabled || !o->enabled)
      c.enabled = !rng.bernoulli(0.75);  // NEAT: disabled in either parent -> 75% disabled
  };
  for (auto& kv : child.conns) {
    auto it = other.conns.find(kv.first);
    cross_gene(kv.second, it == other.conns.end() ? nullptr : &it->second);
  }
  for (auto& kv : child.cell_skips) {
    auto it = other.cell_skips.find(kv.first);
    cross_gene(kv.second, it == other.cell_skips.end() ? nullptr : &it->second);
  }
  for (auto& kv : child.nodes) {
    auto it = other.nodes.find(kv.first);
    if (it == other.nodes.end()) continue;
    kv.second.hp = node_space_.crossover(kv.second.hp, it->second.hp, rng);
    if (kind_ == GenomeKind::Blueprint && rng.bernoulli(0.5)) kv.second.species = it->second.species;
  }
  if (kind_ != GenomeKind::Module) child.globals = global_space_.crossover(fitter.globals, other.globals, rng);
  if (rng.bernoulli(0.5)) child.out_merge = other.out_merge;
  repair(child, rng);
  return child;
}

double GenomeOperators::distance(const Genome& a, const Genome& b) const {
  auto gene_dist = [](const std::map<int, ConnGene>& x, const std::map<int, ConnGene>& y, double& excess,
                      double& disjoint) {
    int max_x = x.empty() ? -1 : x.rbegin()->first;
    int max_y = y.empty() ? -1 : y.rbegin()->first;
    for (const auto& kv : x)
      if (!y.count(kv.first)) (kv.first > max_y ? excess : disjoint) += 1.0;
    for (const auto& kv : y)
      if (!x.count(kv.first)) (kv.first > max_x ? excess : disjoint) += 1.0;
  };
  double excess = 0.0, disjoint = 0.0;
  gene_dist(a.conns, b.conns, excess, disjoint);
  gene_dist(a.cell_skips, b.cell_skips, excess, disjoint);
  double n = static_cast<double>(std::max<size_t>({a.gene_count(), b.gene_count(), 1}));
  const auto& sp = speciation_;
  double d = (sp.c_excess * excess + sp.c_disjoint * disjoint) / n;
  double hp = 0.0, sp_mis = 0.0;
  int matching = 0;
  for (const auto& kv : a.nodes) {
    auto it = b.nodes.find(kv.first);
    if (it == b.nodes.end()) continue;
    ++matching;
    hp += node_space_.distance(kv.second.hp, it->second.hp);
    if (kind_ == GenomeKind::Blueprint && kv.second.species != it->second.species) sp_mis += 1.0;
  }
  if (matching > 0) {
    d += sp.c_hyperparams * hp / matching;
    d += sp.c_species * sp_mis / matching;
  }
  if (kind_ != GenomeKind::Module && !global_space_.empty())
    d += sp.c_globals * global_space_.distance(a.globals, b.globals);
  if (a.out_merge != b.out_merge) d += 0.05;
  return d;
}

}  // namespace cdn
