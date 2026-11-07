// Tour of the introspection / manipulation API on an evolved network:
// activations, gradients, weights, recurrent state, optimizer buffers.

#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>

#include "codeepneat/all.hpp"

using namespace cdn;

static float l2(const std::vector<float>& v) {
  double s = 0;
  for (float x : v) s += double(x) * x;
  return static_cast<float>(std::sqrt(s));
}

int main() {
  // ---------------------------------------------------------------------------
  // 1. Get an evolved, non-trivial architecture. A cheap evaluator that rewards
  //    size grows the genomes quickly (no training needed for this demo).
  // ---------------------------------------------------------------------------
  CoDeepNEATConfig cfg;
  cfg.io.inputs.push_back({"obs", {6}});
  cfg.io.outputs.push_back({"policy", 3, "linear", 0.1f});
  cfg.io.outputs.push_back({"value", 1});
  cfg.module_node_space = spaces::recurrent_layers(4, 16);
  cfg.modules.size = 12;
  cfg.modules.mutation.add_node = 0.6;
  cfg.modules.mutation.add_cell_skip = 0.4;
  cfg.blueprints.size = 6;
  cfg.blueprints.mutation.add_node = 0.6;
  cfg.networks_per_generation = 12;
  CoDeepNEAT evo(cfg);
  evo.run(6, [](EvalContext& ctx) { return static_cast<double>(ctx.net.nodes().size()); });
  const AssembledGenome& genome = evo.best();
  auto net_ptr = evo.build(genome, 1);
  Network& net = *net_ptr;
  std::cout << genome.describe() << "\n" << net.summary() << "\n";

  // ---------------------------------------------------------------------------
  // 2. Weights: list, read, edit in place.
  // ---------------------------------------------------------------------------
  std::cout << "== parameters\n";
  for (const auto& p : net.named_parameters())
    std::cout << "  " << std::left << std::setw(36) << p.name << shape_str(p.tensor.shape()) << "  |w|=" << l2(p.tensor.data())
              << "\n";
  Tensor head_w = net.parameter("head:value.weight");  // shares storage with the network
  head_w.data()[0] = 0.0f;                                 // surgical edit

  // ---------------------------------------------------------------------------
  // 3. Activations: monitor every node with one forward hook, and record them.
  // ---------------------------------------------------------------------------
  std::cout << "\n== activation statistics (forward hook)\n";
  int monitor = net.add_forward_hook([](const NodeInfo& n, Tensor& out) {
    double mean = 0, sq = 0;
    for (float v : out.data()) mean += v, sq += double(v) * v;
    mean /= out.numel();
    std::cout << "  " << std::left << std::setw(24) << n.name << std::setw(8) << n.type << " mean=" << std::setw(10)
              << mean << " rms=" << std::sqrt(sq / out.numel()) << "\n";
  });
  net.record_activations(true);
  Rng rng(0);
  Tensor x = Tensor::randn({4, 6}, rng);
  std::vector<Tensor> out = net.forward(std::vector<Tensor>{x});
  net.remove_hook(monitor);

  // ---------------------------------------------------------------------------
  // 4. Gradients w.r.t. activations: backward hooks + recorded activation grads.
  // ---------------------------------------------------------------------------
  std::cout << "\n== activation gradient norms (after backward)\n";
  Tensor loss = sum(square(out[0])) + sum(out[1]);
  loss.backward();
  for (const auto& kv : net.activations())
    if (kv.second.has_grad()) std::cout << "  d loss / d " << std::setw(24) << kv.first << " = " << l2(kv.second.grad_data()) << "\n";

  // Clip the gradient flowing through every LSTM output to unit norm.
  int clip = net.add_backward_hook(
      [](const NodeInfo&, Tensor& g) {
        float n = l2(g.data());
        if (n > 1.0f)
          for (auto& v : g.data()) v /= n;
      },
      [](const NodeInfo& n) { return n.type == "lstm"; });

  // ---------------------------------------------------------------------------
  // 5. Ablation: silence one module instance and measure the effect.
  // ---------------------------------------------------------------------------
  std::string module_out;
  for (const auto& n : net.nodes())
    if (n.origin.role == "module_out" || (n.origin.role == "layer" && module_out.empty())) module_out = n.name;
  if (!module_out.empty()) {
    net.reset_state(4);
    auto base = net.forward(std::vector<Tensor>{x})[0];
    net.reset_state(4);
    int ablate = net.add_forward_hook(module_out, [](const NodeInfo&, Tensor& o) { o = o * 0.0f; });
    auto ablated = net.forward(std::vector<Tensor>{x})[0];
    net.remove_hook(ablate);
    std::cout << "\n== ablating " << module_out << " changes policy logits by " << l2((base - ablated).data()) << "\n";
  }

  // ---------------------------------------------------------------------------
  // 6. Parameter gradient hooks and freezing (per blueprint node / module).
  // ---------------------------------------------------------------------------
  int lr_mult = net.add_parameter_hook("head:policy.weight", [](Tensor& g) {
    for (auto& v : g.data()) v *= 0.1f;  // 10x smaller effective learning rate for this tensor
  });
  int first_bp_node = -1;
  for (const auto& n : net.nodes())
    if (n.origin.blueprint_node >= 0) {
      first_bp_node = n.origin.blueprint_node;
      break;
    }
  size_t frozen = net.set_trainable([&](const NodeInfo& n) { return n.origin.blueprint_node == first_bp_node; }, false);
  std::cout << "\n== froze " << frozen << " tensors belonging to blueprint node " << first_bp_node << "\n";

  // ---------------------------------------------------------------------------
  // 7. A few optimisation steps with BPTT; inspect optimizer buffers and LSTM gates.
  // ---------------------------------------------------------------------------
  Adam opt(net.named_parameters(), 1e-2f);
  for (int it = 0; it < 5; ++it) {
    net.reset_state(4);
    Tensor total = Tensor::scalar(0.0f);
    for (int t = 0; t < 3; ++t) total = total + mean(square(net.forward(std::vector<Tensor>{x})[1] - 1.0f));
    opt.zero_grad();
    total.backward();
    float gn = clip_grad_norm(net.parameters(), 1.0f);
    opt.step();
    std::cout << "  step " << it << " loss=" << total.item() << " grad-norm=" << gn << "\n";
  }
  if (const ParamState* st = opt.state("head:value.weight"))
    std::cout << "  Adam first moment of head:value.weight: |m|=" << l2(st->m) << " after " << st->steps << " steps\n";
  for (Layer* l : net.layers())
    if (auto* lstm = dynamic_cast<LSTMLayer*>(l)) {
      std::cout << "  " << lstm->name() << ": " << lstm->num_cell_skips() << " cell skip(s), forget gate mean=";
      double m = 0;
      const Tensor& f = lstm->last_gates().at("f");
      for (float v : f.data()) m += v;
      std::cout << m / f.numel() << ", |c|=" << l2(lstm->cell().data()) << "\n";
    }
  for (const auto& s : net.get_state()) std::cout << "  state " << s.name << " " << shape_str(s.tensor.shape()) << "\n";

  net.remove_hook(clip);
  net.remove_hook(lr_mult);
  net.save_weights("inspection_weights.txt");
  genome.save("inspection_genome.txt");
  std::ofstream("inspection.dot") << net.to_dot();
  std::cout << "\nSaved inspection_weights.txt, inspection_genome.txt, inspection.dot\n";
  return 0;
}
