// Genomes: hyperparameters, innovations, mutation and crossover invariants, assembly,
// serialisation, speciation.

#include <set>
#include <sstream>

#include "test_util.hpp"

using namespace cdn;

namespace {

GenomeOperators module_ops(double structural = 0.5) {
  MutationParams m;
  m.add_node = structural;
  m.add_connection = structural;
  m.toggle_connection = 0.2;
  m.add_cell_skip = 0.3;
  m.toggle_cell_skip = 0.2;
  m.max_nodes = 12;
  return GenomeOperators(GenomeKind::Module, spaces::recurrent_layers(), HyperparamSpace(), m, SpeciationParams());
}

GenomeOperators blueprint_ops() {
  MutationParams m;
  m.add_node = 0.5;
  m.add_connection = 0.5;
  m.change_species = 0.3;
  m.max_nodes = 8;
  return GenomeOperators(GenomeKind::Blueprint, spaces::blueprint_nodes(), spaces::rl_globals(), m,
                         SpeciationParams());
}

void check_valid(const Genome& g) {
  std::string why;
  if (!g.validate(&why)) throw TestFailure{"invalid genome: " + why + "\n" + g.describe()};
  CHECK(g.has_input_output_path());
  g.active_nodes();  // throws on cycles
}

}  // namespace

TEST(hyperparam_space_sample_mutate_bounds) {
  Rng rng(1);
  HyperparamSpace s = spaces::rl_globals();
  s.add(ParamSpec::integer("layers", 1, 4));
  HyperparamValues v = s.sample(rng);
  for (int i = 0; i < 2000; ++i) {
    s.mutate(v, rng, 0.5);
    for (const auto& spec : s.specs()) {
      if (spec.kind == ParamKind::Float || spec.kind == ParamKind::Int) {
        CHECK(v.f(spec.name) >= spec.lo - 1e-12 && v.f(spec.name) <= spec.hi + 1e-12);
      }
      if (spec.kind == ParamKind::Choice)
        CHECK(std::find(spec.choices.begin(), spec.choices.end(), v.s(spec.name)) != spec.choices.end());
    }
  }
  HyperparamValues back = HyperparamValues::deserialize(v.serialize());
  CHECK(back == v);
  CHECK(s.distance(v, v) == 0.0);
  HyperparamValues w = v;
  s.mutate(w, rng, 0.0);  // forces at least one change
  CHECK(s.distance(v, w) > 0.0);
}

TEST(innovation_tracker_alignment) {
  auto ops = module_ops();
  InnovationTracker innov(1);
  Rng rng(3);
  Genome a = ops.create_initial(1, innov, rng), b = ops.create_initial(1, innov, rng);
  CHECK(a.conns.size() == 2 && b.conns.size() == 2);
  for (const auto& kv : a.conns) CHECK(b.conns.count(kv.first));
  // Splitting the same connection gives the same node id in both genomes.
  int c = a.conns.begin()->first;
  int na = innov.split_node(c, a), nb = innov.split_node(c, b);
  CHECK(na == nb);
  a.nodes[na] = NodeGene{na, {}, -1};
  CHECK(innov.split_node(c, a) != na);  // already present -> fresh id
  CHECK(innov.connection(0, 5) == innov.connection(0, 5));
}

TEST(module_mutation_invariants) {
  auto ops = module_ops();
  InnovationTracker innov(1);
  Rng rng(7);
  for (int trial = 0; trial < 20; ++trial) {
    Genome g = ops.create_initial(1, innov, rng);
    for (int i = 0; i < 150; ++i) {
      ops.mutate(g, innov, rng);
      check_valid(g);
    }
    CHECK(static_cast<int>(g.nodes.size()) <= ops.mutation().max_nodes);
  }
}

TEST(lstm_cell_skip_mutations) {
  auto ops = module_ops();
  InnovationTracker innov(1);
  Rng rng(11);
  Genome g = ops.create_initial(1, innov, rng);
  // Grow a chain of LSTM nodes.
  for (int i = 0; i < 4; ++i) ops.mutate_add_node(g, innov, rng, {});
  for (auto& kv : g.nodes) kv.second.hp.set_choice("type", "lstm");
  int added = 0;
  for (int i = 0; i < 30; ++i) added += ops.mutate_add_cell_skip(g, innov, rng) ? 1 : 0;
  CHECK(added > 0 && !g.cell_skips.empty());
  for (const auto& kv : g.cell_skips) {
    CHECK(g.is_hidden(kv.second.from) && g.is_hidden(kv.second.to) && kv.second.from != kv.second.to);
  }
  CHECK(ops.mutate_toggle_cell_skip(g, rng));
  check_valid(g);
  // Skips survive assembly as cell-skip edges between LSTM layers.
  AssembledGenome ag;
  Genome dn = g;
  dn.kind = GenomeKind::DeepNEAT;
  ag.blueprint = dn;
  IOSpec io;
  io.inputs.push_back({"x", {3}});
  io.outputs.push_back({"y", 2});
  NetworkGraph graph = assemble_network(ag, io);
  size_t enabled = 0;
  for (const auto& kv : g.cell_skips) {
    bool active_from = false, active_to = false;
    for (int n : g.active_nodes()) {
      active_from |= n == kv.second.from;
      active_to |= n == kv.second.to;
    }
    enabled += (kv.second.enabled && active_from && active_to) ? 1 : 0;
  }
  CHECK(graph.cell_skips.size() == enabled);
  Network net(graph, BuildOptions(), 1);
  net.forward(Tensor::ones({2, 3}));
}

TEST(crossover_invariants) {
  auto ops = module_ops();
  InnovationTracker innov(1);
  Rng rng(13);
  std::vector<Genome> pop;
  for (int i = 0; i < 12; ++i) {
    Genome g = ops.create_initial(1, innov, rng);
    for (int k = 0; k < 20; ++k) ops.mutate(g, innov, rng);
    g.fitness = rng.uniform();
    pop.push_back(g);
  }
  for (int i = 0; i < 400; ++i) {
    const Genome& a = pop[rng.index(pop.size())];
    const Genome& b = pop[rng.index(pop.size())];
    Genome c = a.fitness >= b.fitness ? ops.crossover(a, b, rng) : ops.crossover(b, a, rng);
    check_valid(c);
    const Genome& fitter = a.fitness >= b.fitness ? a : b;
    CHECK(c.conns.size() == fitter.conns.size());  // topology comes from the fitter parent
  }
}

TEST(distance_properties) {
  auto ops = module_ops();
  InnovationTracker innov(1);
  Rng rng(17);
  Genome a = ops.create_initial(1, innov, rng);
  Genome b = a;
  CHECK(ops.distance(a, b) == 0.0);
  for (int k = 0; k < 5; ++k) ops.mutate_add_node(b, innov, rng, {});
  double d1 = ops.distance(a, b), d2 = ops.distance(b, a);
  CHECK(d1 > 0.0);
  CHECK_NEAR(d1, d2, 1e-12);
}

TEST(blueprint_species_pointers_and_globals) {
  auto ops = blueprint_ops();
  InnovationTracker innov(2);
  Rng rng(19);
  std::vector<int> species = {3, 7, 9};
  Genome g = ops.create_initial(2, innov, rng, species);
  CHECK(g.num_inputs == 2 && g.output_id() == 2);
  CHECK(!g.globals.empty());
  for (int i = 0; i < 300; ++i) {
    ops.mutate(g, innov, rng, species);
    check_valid(g);
    for (const auto& kv : g.nodes)
      CHECK(std::find(species.begin(), species.end(), kv.second.species) != species.end());
  }
  // Species that disappear are re-targeted on the next mutation.
  std::vector<int> fewer = {7};
  ops.mutate(g, innov, rng, fewer);
  for (const auto& kv : g.nodes) CHECK(kv.second.species == 7);
}

TEST(genome_serialization_roundtrip) {
  auto ops = module_ops();
  InnovationTracker innov(1);
  Rng rng(23);
  Genome g = ops.create_initial(1, innov, rng);
  for (int k = 0; k < 40; ++k) ops.mutate(g, innov, rng);
  g.id = 42;
  g.fitness = 1.25;
  std::stringstream ss;
  g.write(ss);
  Genome r = Genome::read(ss);
  CHECK(r.describe() == g.describe());
  CHECK(r.conns.size() == g.conns.size() && r.cell_skips.size() == g.cell_skips.size());
  for (const auto& kv : g.nodes) CHECK(r.nodes.at(kv.first).hp == kv.second.hp);
}

TEST(assembly_repeats_modules_per_species) {
  // Blueprint: in -> A -> B -> out, A and B point to the same species; C to another.
  Genome bp(GenomeKind::Blueprint, 1);
  HyperparamValues hp;
  hp.set_choice("merge", "concat");
  bp.nodes[2] = NodeGene{2, hp, 10};
  bp.nodes[3] = NodeGene{3, hp, 10};
  bp.nodes[4] = NodeGene{4, hp, 20};
  bp.conns[0] = ConnGene{0, 0, 2, true};
  bp.conns[1] = ConnGene{1, 2, 3, true};
  bp.conns[2] = ConnGene{2, 3, 1, true};
  bp.conns[3] = ConnGene{3, 0, 4, true};
  bp.conns[4] = ConnGene{4, 4, 1, true};
  std::string why;
  CHECK(bp.validate(&why));

  auto make_module = [](int id, const std::string& type, int units) {
    Genome m(GenomeKind::Module, 1);
    m.id = id;
    HyperparamValues h;
    h.set_choice("type", type).set_int("units", units).set_choice("activation", "relu");
    m.nodes[2] = NodeGene{2, h, -1};
    m.conns[0] = ConnGene{0, 0, 2, true};
    m.conns[1] = ConnGene{1, 2, 1, true};
    return m;
  };
  AssembledGenome ag;
  ag.blueprint = bp;
  ag.modules[10] = make_module(100, "dense", 8);
  ag.modules[20] = make_module(200, "lstm", 5);
  IOSpec io;
  io.inputs.push_back({"obs", {4}});
  io.outputs.push_back({"policy", 2});
  NetworkGraph g = assemble_network(ag, io);
  CHECK(g.find("bp2/m100/n2") >= 0 && g.find("bp3/m100/n2") >= 0);  // same module twice
  CHECK(g.find("bp4/m200/n2") >= 0);
  Network net(g, BuildOptions(), 1);
  // Repeated module instances have independent weights.
  CHECK(net.parameter("bp2/m100/n2.weight").data() != net.parameter("bp3/m100/n2.weight").data() ||
        net.parameter("bp2/m100/n2.weight").shape() != net.parameter("bp3/m100/n2.weight").shape());
  CHECK(net.node("bp3/m100/n2").origin.species == 10);
  CHECK(net.node("bp4/m200/n2").type == "lstm");
  CHECK(net.is_recurrent());
  auto out = net.forward(Tensor::ones({3, 4}));
  CHECK((out.shape() == Shape{3, 2}));

  std::stringstream ss;
  ag.write(ss);
  AssembledGenome back = AssembledGenome::read(ss);
  CHECK(assemble_network(back, io).to_dot() == g.to_dot());
}

TEST(population_speciation_and_reproduction) {
  PopulationParams p;
  p.size = 30;
  p.speciation.target_species = 4;
  p.mutation.add_node = 0.3;
  p.mutation.add_connection = 0.3;
  Population pop(p, module_ops(0.3), 1);
  Rng rng(29);
  pop.initialize(rng);
  CHECK(static_cast<int>(pop.genomes().size()) == 30);
  for (int gen = 0; gen < 15; ++gen) {
    for (auto& g : pop.genomes()) g.fitness = static_cast<double>(g.nodes.size()) + rng.uniform();
    pop.reproduce(gen, rng);
    CHECK(static_cast<int>(pop.genomes().size()) == 30);
    std::set<int> ids;
    size_t members = 0;
    for (const auto& s : pop.species()) {
      CHECK(!s.members.empty());
      CHECK(ids.insert(s.id).second);
      members += s.members.size();
      for (int m : s.members) CHECK(pop.genomes()[m].species == s.id);
    }
    CHECK(members == 30);
    for (const auto& g : pop.genomes()) check_valid(g);
  }
  // Selection pressure on size should grow the genomes.
  double mean_nodes = 0;
  for (const auto& g : pop.genomes()) mean_nodes += g.nodes.size();
  CHECK(mean_nodes / 30.0 > 1.5);
}
