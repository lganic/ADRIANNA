// Evolving recurrent architectures (Sec. 4 of the paper) for a memory task.
//
// MemoryCue shows a cue on the first step and asks for it back several steps
// later; memoryless (dense-only) networks cannot exceed 0 mean reward. Module
// nodes may be dense or LSTM layers, and the two LSTM-specific mutations of
// the paper are active: enabling/disabling connections between layers and
// adding/removing gated memory-cell skip connections between LSTM nodes
// (including feedback connections from later to earlier LSTM cells).
//
// usage: memory_lstm [generations=8] [threads=hw] [delay=10]

#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <thread>

#include "codeepneat/all.hpp"

using namespace cdn;

int main(int argc, char** argv) {
  int generations = argc > 1 ? std::atoi(argv[1]) : 8;
  int threads = argc > 2 ? std::atoi(argv[2]) : static_cast<int>(std::max(1u, std::thread::hardware_concurrency()));
  int delay = argc > 3 ? std::atoi(argv[3]) : 10;

  rl::EnvFactory make_env = [delay] { return std::unique_ptr<rl::Environment>(new rl::MemoryCue(delay)); };
  auto proto = make_env();

  CoDeepNEATConfig cfg;
  cfg.io = rl::actor_critic_io(*proto);
  cfg.module_node_space = spaces::recurrent_layers(8, 32);
  cfg.module_node_space.find("dropout")->hi = 0.0;
  cfg.global_space = spaces::rl_globals();
  cfg.modules.size = 20;
  cfg.modules.speciation.target_species = 3;
  cfg.modules.mutation.add_cell_skip = 0.15;
  cfg.modules.mutation.toggle_cell_skip = 0.05;
  cfg.blueprints.size = 10;
  cfg.blueprints.speciation.target_species = 2;
  cfg.networks_per_generation = 20;
  cfg.num_threads = threads;
  cfg.seed = 4;

  Evaluator evaluate = [&](EvalContext& ctx) {
    rl::A2CConfig ac;
    ac.total_steps = 24000;
    ac.rollout_length = delay + 2;
    rl::A2CTrainer trainer(ctx.net, make_env, ac, ctx.rng.next_u64());
    trainer.apply_globals(ctx.globals);
    trainer.train();
    return rl::evaluate_policy(ctx.net, make_env, 40, true, ctx.rng.next_u64());
  };

  CoDeepNEAT evo(cfg);
  std::cout << "CoDeepNEAT on MemoryCue(delay=" << delay << "): memoryless policies score 0, perfect memory 1\n";
  evo.run(generations, evaluate, [&](const GenerationStats& st) {
    std::cout << st.to_string() << std::endl;
    return true;
  });

  const AssembledGenome& best = evo.best();
  std::cout << "\n" << best.describe() << "\n";
  auto net = evo.build(best, 1);
  std::cout << net->summary() << "\n";
  int lstm_layers = 0;
  for (const auto& n : net->nodes()) lstm_layers += n.type == "lstm" ? 1 : 0;
  std::cout << "Champion has " << lstm_layers << " LSTM layer(s) and " << net->graph().cell_skips.size()
            << " memory-cell skip connection(s).\n";

  // Look inside the recurrent state of the champion while it solves an episode.
  rl::A2CConfig ac;
  ac.total_steps = 24000;
  ac.rollout_length = delay + 2;
  rl::A2CTrainer trainer(*net, make_env, ac, 5);
  trainer.apply_globals(best.globals());
  trainer.train();
  std::cout << "Retrained champion score: " << rl::evaluate_policy(*net, make_env, 100, true, 77) << "\n";
  if (net->is_recurrent()) {
    Rng rng(3);
    auto env = make_env();
    auto obs = env->reset(rng);
    net->eval();
    net->reset_state(1);
    NoGradGuard ng;
    for (int t = 0; t <= delay; ++t) {
      auto out = net->forward(std::vector<Tensor>{rl::batch_observations({obs}, env->observation_shape())});
      std::cout << "t=" << t << " obs=[" << obs[0] << "," << obs[1] << "," << obs[2] << "] policy="
                << softmax(out[0]).to_string(2);
      auto st = net->get_state();
      if (!st.empty()) std::cout << " |" << st[1].name << "|=" << std::sqrt(sum(square(st[1].tensor)).item());
      std::cout << "\n";
      obs = env->step(argmax_rows(out[0])[0], rng).observation;
    }
  }
  return 0;
}
