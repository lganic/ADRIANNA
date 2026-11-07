// DeepNEAT (Sec. 3.1) on a small supervised problem (two interleaved spirals),
// side by side with CoDeepNEAT on the same task.

#include <cmath>
#include <cstdlib>
#include <iostream>

#include "codeepneat/all.hpp"

using namespace cdn;

struct Dataset {
  Tensor X;
  std::vector<int> y;
};

static Dataset spirals(int n, uint64_t seed) {
  Rng rng(seed);
  std::vector<float> xs;
  std::vector<int> ys;
  for (int i = 0; i < n; ++i) {
    int c = i % 2;
    double r = rng.uniform(0.1, 1.0), t = 5.0 * 3.14159 * r + c * 3.14159 + rng.normal(0, 0.05);
    xs.push_back(static_cast<float>(r * std::cos(t)));
    xs.push_back(static_cast<float>(r * std::sin(t)));
    ys.push_back(c);
  }
  return {Tensor({n, 2}, xs), ys};
}

static double accuracy(Network& net, const Dataset& d) {
  NoGradGuard ng;
  net.eval();
  auto pred = argmax_rows(net.forward(d.X));
  int ok = 0;
  for (size_t i = 0; i < pred.size(); ++i) ok += pred[i] == d.y[i];
  net.train();
  return double(ok) / pred.size();
}

int main(int argc, char** argv) {
  int generations = argc > 1 ? std::atoi(argv[1]) : 10;
  Dataset train = spirals(256, 1), valid = spirals(256, 2);

  // Like the paper: train each candidate for a fixed short budget, fitness =
  // validation accuracy.
  Evaluator eval = [&](EvalContext& ctx) {
    auto opt = make_optimizer_from_globals(ctx.globals, ctx.net.named_parameters(), "adam", 1e-2f);
    for (int epoch = 0; epoch < 80; ++epoch) {
      Tensor loss = cross_entropy(ctx.net.forward(train.X), train.y);
      opt->zero_grad();
      loss.backward();
      opt->step();
    }
    return accuracy(ctx.net, valid);
  };

  HyperparamSpace globals;
  globals.add(ParamSpec::real("learning_rate", 1e-3, 5e-2, true));
  globals.add(ParamSpec::choice("optimizer", {"adam", "rmsprop"}));
  globals.add(ParamSpec::choice("weight_init", {"glorot_uniform", "he_normal"}));

  IOSpec io;
  io.inputs.push_back({"xy", {2}});
  io.outputs.push_back({"class", 2});

  DeepNEATConfig dcfg;
  dcfg.io = io;
  dcfg.node_space = spaces::dense_layers(4, 32);
  dcfg.node_space.find("dropout")->hi = 0.0;
  dcfg.global_space = globals;
  dcfg.population.size = 20;
  dcfg.population.speciation.target_species = 3;
  dcfg.population.mutation.add_node = 0.3;
  dcfg.num_threads = 4;
  DeepNEAT deep(dcfg);
  std::cout << "DeepNEAT:\n";
  deep.run(generations, eval, [](const GenerationStats& s) {
    std::cout << "  " << s.to_string() << "\n";
    return true;
  });

  CoDeepNEATConfig ccfg;
  ccfg.io = io;
  ccfg.module_node_space = dcfg.node_space;
  ccfg.global_space = globals;
  ccfg.modules.size = 16;
  ccfg.blueprints.size = 8;
  ccfg.networks_per_generation = 20;
  ccfg.num_threads = 4;
  CoDeepNEAT co(ccfg);
  std::cout << "CoDeepNEAT:\n";
  co.run(generations, eval, [](const GenerationStats& s) {
    std::cout << "  " << s.to_string() << "\n";
    return true;
  });

  std::cout << "\nBest DeepNEAT network:\n" << deep.build(deep.best())->summary();
  std::cout << "\nBest CoDeepNEAT network:\n" << co.build(co.best())->summary();
  return 0;
}
