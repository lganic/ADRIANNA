// Evolving convolutional architectures from pixels (the CIFAR-10 style search
// space of Sec. 3.3, scaled down) on the "Catch" game.
//
// Modules are graphs of conv2d nodes (filters, kernel size, max pooling,
// dropout, weight scaling, merge method); blueprints wire modules together and
// may add skip connections that bypass whole modules. Merges between feature
// maps of different resolution are handled by max pooling to the smallest size.
//
// usage: catch_conv [generations=6] [threads=hw]

#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <thread>

#include "codeepneat/all.hpp"

using namespace cdn;

int main(int argc, char** argv) {
  int generations = argc > 1 ? std::atoi(argv[1]) : 6;
  int threads = argc > 2 ? std::atoi(argv[2]) : static_cast<int>(std::max(1u, std::thread::hardware_concurrency()));

  rl::EnvFactory make_env = [] { return std::unique_ptr<rl::Environment>(new rl::Catch(10, 5)); };
  auto proto = make_env();

  CoDeepNEATConfig cfg;
  cfg.io = rl::actor_critic_io(*proto);
  cfg.module_node_space = spaces::conv_layers(4, 16);
  cfg.module_node_space.find("dropout")->hi = 0.1;
  cfg.global_space = spaces::rl_globals();
  cfg.modules.size = 16;
  cfg.modules.speciation.target_species = 3;
  cfg.modules.mutation.max_nodes = 4;
  cfg.blueprints.size = 8;
  cfg.blueprints.speciation.target_species = 2;
  cfg.blueprints.mutation.max_nodes = 4;
  cfg.networks_per_generation = 16;
  cfg.num_threads = threads;
  cfg.seed = 10;

  Evaluator evaluate = [&](EvalContext& ctx) {
    rl::A2CConfig ac;
    ac.total_steps = 12000;
    ac.rollout_length = 9;
    rl::A2CTrainer trainer(ctx.net, make_env, ac, ctx.rng.next_u64());
    trainer.apply_globals(ctx.globals);
    trainer.train();
    return rl::evaluate_policy(ctx.net, make_env, 30, true, ctx.rng.next_u64());
  };

  CoDeepNEAT evo(cfg);
  std::cout << "CoDeepNEAT on Catch (pixels " << shape_str(proto->observation_shape()) << ")\n";
  evo.run(generations, evaluate, [](const GenerationStats& st) {
    std::cout << st.to_string() << std::endl;
    return true;
  });
  const AssembledGenome& best = evo.best();
  auto net = evo.build(best, 1);
  std::cout << "\n" << net->summary() << "\n";
  std::ofstream("catch_best.dot") << net->to_dot();
  std::cout << "Champion graph written to catch_best.dot (render with: dot -Tpng catch_best.dot -o catch.png)\n";
  return 0;
}
