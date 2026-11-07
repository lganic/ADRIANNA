// End to end: evolution on a regression task, determinism, fitness attribution, DeepNEAT, A2C.

#include <cstdio>
#include <set>

#include "test_util.hpp"

using namespace cdn;

namespace {

/// Supervised proxy task: regress y = sin(2 x0) + x1 * x2. Fitness = -MSE after
/// a short training run with the evolved learning rate / optimizer.
double regression_fitness(EvalContext& ctx) {
  Rng data_rng(1234);
  const int N = 64;
  std::vector<float> xs(N * 3), ys(N);
  for (int i = 0; i < N; ++i) {
    for (int k = 0; k < 3; ++k) xs[i * 3 + k] = static_cast<float>(data_rng.uniform(-1, 1));
    ys[i] = std::sin(2 * xs[i * 3]) + xs[i * 3 + 1] * xs[i * 3 + 2];
  }
  Tensor X({N, 3}, xs), Y({N, 1}, ys);
  auto opt = make_optimizer_from_globals(ctx.globals, ctx.net.named_parameters());
  for (int step = 0; step < 60; ++step) {
    Tensor loss = mse_loss(ctx.net.forward(X), Y);
    opt->zero_grad();
    loss.backward();
    clip_grad_norm(ctx.net.parameters(), 5.0f);
    opt->step();
  }
  ctx.net.eval();
  NoGradGuard ng;
  return -mse_loss(ctx.net.forward(X), Y).item();
}

CoDeepNEATConfig small_config(int threads) {
  CoDeepNEATConfig cfg;
  cfg.io.inputs.push_back({"x", {3}});
  cfg.io.outputs.push_back({"y", 1});
  cfg.module_node_space = spaces::dense_layers(4, 24);
  cfg.module_node_space.find("dropout")->hi = 0.0;
  HyperparamSpace g;
  g.add(ParamSpec::real("learning_rate", 1e-3, 5e-2, true));
  g.add(ParamSpec::choice("optimizer", {"adam", "sgd"}));
  g.add(ParamSpec::real("momentum", 0.68, 0.99));
  cfg.global_space = g;
  cfg.modules.size = 12;
  cfg.modules.speciation.target_species = 3;
  cfg.modules.mutation.add_node = 0.3;
  cfg.modules.mutation.add_connection = 0.3;
  cfg.blueprints.size = 6;
  cfg.blueprints.speciation.target_species = 2;
  cfg.blueprints.mutation.add_node = 0.3;
  cfg.networks_per_generation = 12;
  cfg.num_threads = threads;
  cfg.seed = 77;
  return cfg;
}

void check_population_consistency(const CoDeepNEAT& evo) {
  std::set<int> species;
  for (const auto& s : evo.modules().species()) species.insert(s.id);
  for (const auto& bp : evo.blueprints().genomes()) {
    std::string why;
    if (!bp.validate(&why)) throw TestFailure{"invalid blueprint: " + why};
    for (const auto& kv : bp.nodes) CHECK(species.count(kv.second.species));
  }
  for (const auto& m : evo.modules().genomes()) {
    std::string why;
    if (!m.validate(&why)) throw TestFailure{"invalid module: " + why};
  }
}

}  // namespace

TEST(codeepneat_evolves_supervised_task) {
  CoDeepNEAT evo(small_config(2));
  check_population_consistency(evo);
  double prev_best = -1e300;
  double first_mean = 0, last_best = 0;
  for (int gen = 0; gen < 6; ++gen) {
    GenerationStats st = evo.step(regression_fitness);
    CHECK(st.failures == 0);
    CHECK(st.evaluations == 12);
    CHECK(st.best_ever >= prev_best);
    prev_best = st.best_ever;
    if (gen == 0) first_mean = st.mean_fitness;
    last_best = st.best_ever;
    check_population_consistency(evo);
  }
  CHECK(last_best > first_mean);
  // The best network can be rebuilt (and saved / loaded) for further training.
  const AssembledGenome& best = evo.best();
  std::string path = "cdn_test_best.txt";
  best.save(path);
  AssembledGenome loaded = AssembledGenome::load(path);
  std::remove(path.c_str());
  auto a = evo.build(best, 3), b = evo.build(loaded, 3);
  CHECK(a->num_parameters() == b->num_parameters());
  CHECK(a->summary() == b->summary());
}

TEST(codeepneat_deterministic_across_thread_counts) {
  CoDeepNEAT a(small_config(1)), b(small_config(3));
  for (int gen = 0; gen < 3; ++gen) {
    GenerationStats sa = a.step(regression_fitness), sb = b.step(regression_fitness);
    CHECK(sa.records.size() == sb.records.size());
    for (size_t i = 0; i < sa.records.size(); ++i) CHECK(sa.records[i].fitness == sb.records[i].fitness);
  }
}

TEST(codeepneat_fitness_attribution_is_average) {
  CoDeepNEATConfig cfg = small_config(1);
  CoDeepNEAT evo(cfg);
  // Fitness = blueprint id + module count, so the averages are predictable.
  std::map<int, std::vector<double>> per_bp;
  auto eval = [&](EvalContext& ctx) {
    double f = static_cast<double>(ctx.genome.blueprint.id % 5) + 0.1 * ctx.genome.modules.size();
    return f;
  };
  // Snapshot blueprint ids before the step (reproduction replaces them).
  std::vector<int> ids;
  for (const auto& g : evo.blueprints().genomes()) ids.push_back(g.id);
  GenerationStats st = evo.step(eval);
  for (const auto& r : st.records) per_bp[r.blueprint_id].push_back(r.fitness);
  // Every blueprint was evaluated (12 networks, 6 blueprints -> 2 each).
  for (int id : ids) CHECK(per_bp[id].size() == 2);
  // Errors in the evaluator are caught and recorded.
  GenerationStats st2 = evo.step([](EvalContext&) -> double { throw std::runtime_error("boom"); });
  CHECK(st2.failures == 12);
  CHECK(st2.records[0].error == "boom");
}

TEST(deepneat_evolves_supervised_task) {
  DeepNEATConfig cfg;
  cfg.io.inputs.push_back({"x", {3}});
  cfg.io.outputs.push_back({"y", 1});
  cfg.node_space = spaces::dense_layers(4, 24);
  cfg.node_space.find("dropout")->hi = 0.0;
  cfg.global_space = small_config(1).global_space;
  cfg.population.size = 10;
  cfg.population.speciation.target_species = 2;
  cfg.population.mutation.add_node = 0.3;
  cfg.num_threads = 2;
  DeepNEAT evo(cfg);
  double first = 0, last = 0;
  for (int gen = 0; gen < 4; ++gen) {
    GenerationStats st = evo.step(regression_fitness);
    CHECK(st.failures == 0);
    if (gen == 0) first = st.mean_fitness;
    last = st.best_ever;
  }
  CHECK(last >= first);
  auto net = evo.build(evo.best());
  CHECK(net->num_parameters() > 0);
}

TEST(rl_environments_and_a2c_smoke) {
  Rng rng(1);
  rl::CartPole cp;
  auto o = cp.reset(rng);
  CHECK(o.size() == 4);
  int steps = 0;
  while (!cp.step(1, rng).done) ++steps;
  CHECK(steps > 3 && steps < 100);  // always pushing right falls over quickly
  rl::MemoryCue mc(3);
  auto obs = mc.reset(rng);
  int cue = obs[0] > 0.5f ? 0 : 1;
  CHECK(!mc.step(0, rng).done && !mc.step(0, rng).done);
  auto go = mc.step(0, rng);
  CHECK(go.observation[2] == 1.0f && !go.done);
  auto fin = mc.step(cue, rng);
  CHECK(fin.done && fin.reward == 1.0f);

  // Catch with a hand-written conv actor-critic learns well above chance.
  rl::Catch env;
  NetworkGraph g;
  int in = g.add_input("input:obs", env.observation_shape());
  HyperparamValues dh;
  dh.set_int("units", 32).set_choice("activation", "relu");
  int d = g.add_layer("d", "dense", dh, {in});
  g.add_output("head:policy", d, env.num_actions(), "linear", 0.1f);
  g.add_output("head:value", d, 1);
  Network net(g, BuildOptions(), 3);
  rl::A2CConfig ac;
  ac.total_steps = 25000;
  ac.learning_rate = 1e-2f;
  rl::EnvFactory make = [] { return std::unique_ptr<rl::Environment>(new rl::Catch()); };
  rl::A2CTrainer trainer(net, make, ac, 5);
  int updates_seen = 0;
  trainer.on_update = [&](const rl::UpdateInfo& info, Network& n) {
    ++updates_seen;
    CHECK(std::isfinite(info.grad_norm));
    CHECK(n.parameter("d.weight").has_grad());
  };
  rl::TrainStats stats = trainer.train();
  CHECK(updates_seen == stats.updates);
  double score = rl::evaluate_policy(net, make, 50, true, 9);
  std::cout << "         catch score after A2C: " << score << std::endl;
  CHECK(score > 0.5);
}
