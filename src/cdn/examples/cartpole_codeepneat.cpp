// CoDeepNEAT as the outer loop around actor-critic RL on CartPole.
//
// Every assembled network is trained from scratch with A2C using its evolved
// global hyperparameters (learning rate, optimizer, entropy / value
// coefficients, gradient clipping, weight init), and its fitness is the mean
// greedy return afterwards. Evolution therefore searches for architectures +
// hyperparameters that *learn* fast, as in the paper.
//
// usage: cartpole_codeepneat [generations=10] [threads=hw] [train_steps=12000]

#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <thread>

#include "codeepneat/all.hpp"

using namespace cdn;

int main(int argc, char** argv) {
  int generations = argc > 1 ? std::atoi(argv[1]) : 10;
  int threads = argc > 2 ? std::atoi(argv[2]) : static_cast<int>(std::max(1u, std::thread::hardware_concurrency()));
  long train_steps = argc > 3 ? std::atol(argv[3]) : 12000;

  rl::EnvFactory make_env = [] { return std::unique_ptr<rl::Environment>(new rl::CartPole()); };
  auto proto = make_env();

  CoDeepNEATConfig cfg;
  cfg.io = rl::actor_critic_io(*proto);
  cfg.module_node_space = spaces::dense_layers(8, 64);
  cfg.module_node_space.find("dropout")->hi = 0.0;  // dropout rarely helps on-policy RL
  cfg.global_space = spaces::rl_globals();
  cfg.modules.size = 24;
  cfg.modules.speciation.target_species = 3;
  cfg.blueprints.size = 12;
  cfg.blueprints.speciation.target_species = 2;
  cfg.networks_per_generation = 24;
  cfg.num_threads = threads;
  cfg.seed = 2017;

  Evaluator evaluate = [&](EvalContext& ctx) {
    rl::A2CConfig ac;
    ac.total_steps = train_steps;
    rl::A2CTrainer trainer(ctx.net, make_env, ac, ctx.rng.next_u64());
    trainer.apply_globals(ctx.globals);
    trainer.train();
    return rl::evaluate_policy(ctx.net, make_env, 5, true, ctx.rng.next_u64());
  };

  CoDeepNEAT evo(cfg);
  std::cout << "CoDeepNEAT on CartPole: " << generations << " generations, " << cfg.networks_per_generation
            << " networks/generation, " << threads << " threads\n";
  evo.run(generations, evaluate, [&](const GenerationStats& st) {
    std::cout << st.to_string() << std::endl;
    return st.best_ever < 500.0;  // solved
  });

  const AssembledGenome& best = evo.best();
  best.save("cartpole_best_genome.txt");
  std::cout << "\nBest network (fitness " << best.fitness << "), saved to cartpole_best_genome.txt\n";
  std::cout << best.describe() << "\n";

  // Rebuild the champion, retrain it with its evolved hyperparameters, inspect it.
  auto net = evo.build(best, 7);
  std::cout << net->summary() << "\n";
  rl::A2CConfig ac;
  ac.total_steps = train_steps * 2;
  rl::A2CTrainer trainer(*net, make_env, ac, 99);
  trainer.apply_globals(best.globals());
  trainer.train();
  double score = rl::evaluate_policy(*net, make_env, 20, true, 123);
  std::cout << "Retrained champion: mean greedy return over 20 episodes = " << score << "\n";
  net->save_weights("cartpole_best_weights.txt");
  std::ofstream("cartpole_best.dot") << net->to_dot();
  std::cout << "Weights -> cartpole_best_weights.txt, graph -> cartpole_best.dot\n";
  TorchExportOptions torch_opt;
  torch_opt.class_name = "CartPolePolicy";
  torch_opt.description = "CartPole actor-critic; greedy return " + std::to_string(score) + ".";
  export_torch(*net, "cartpole_policy", torch_opt);
  std::cout << "PyTorch module -> cartpole_policy.py + cartpole_policy.safetensors\n"
            << "  python: from cartpole_policy import CartPolePolicy;"
               " m = CartPolePolicy.from_pretrained('cartpole_policy.safetensors')\n";
  return 0;
}
