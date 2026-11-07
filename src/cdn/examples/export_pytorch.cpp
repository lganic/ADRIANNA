// Export evolved networks to PyTorch.
//
//   export_pytorch [out_dir=torch_export]                 export + write reference runs
//   python tools/verify_torch_export.py --dir torch_export --write-back
//   export_pytorch [out_dir] --check-import               load the PyTorch-updated weights back into C++
//
// Cases exported:
//   handbuilt   every supported feature: two inputs, strided conv, conv on flat
//               input, spatial and flat sum-merges with projections, mixed
//               spatial/flat merges, stacked LSTMs with forward + feedback
//               memory-cell skips, several activations, two heads
//   zoo_*       random CoDeepNEAT assemblies from the mixed conv/dense/LSTM space
//   cartpole    an evolved and A2C-trained CartPole actor-critic

#include <cmath>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "codeepneat/all.hpp"

using namespace cdn;

namespace {

struct Case {
  std::string name;
  std::unique_ptr<Network> net;
};

HyperparamValues hp_dense(int units, const std::string& act, const std::string& merge = "concat", float drop = 0) {
  HyperparamValues h;
  h.set_choice("type", "dense").set_int("units", units).set_choice("activation", act).set_choice("merge", merge);
  h.set_float("dropout", drop);
  return h;
}
HyperparamValues hp_conv(int filters, int k, bool pool, const std::string& act, int stride = 1) {
  HyperparamValues h;
  h.set_choice("type", "conv2d").set_int("filters", filters).set_int("kernel_size", k).set_bool("max_pool", pool);
  h.set_choice("activation", act).set_int("stride", stride);
  return h;
}
HyperparamValues hp_lstm(int units, float drop = 0) {
  HyperparamValues h;
  h.set_choice("type", "lstm").set_int("units", units).set_float("dropout", drop);
  return h;
}

std::unique_ptr<Network> handbuilt() {
  NetworkGraph g;
  int img = g.add_input("input:image", {2, 8, 8});
  int vec = g.add_input("input:vec", {5});
  int c1 = g.add_layer("c1", "conv2d", hp_conv(6, 3, true, "relu"), {img});        // [6,4,4]
  int c2 = g.add_layer("c2", "conv2d", hp_conv(4, 3, false, "elu", 2), {img});     // [4,4,4] (stride 2)
  int c3 = g.add_layer("c3", "conv2d", hp_conv(3, 1, false, "leaky_relu"), {img}); // [3,8,8]
  int ssum = g.add_identity("spatial_sum", {c1, c2, c3}, "sum");                   // pool c3, project to 3 ch
  int scat = g.add_layer("spatial_cat", "conv2d", hp_conv(5, 3, false, "tanh"), {c1, c3}, "concat");
  int cv = g.add_layer("conv_on_vec", "conv2d", hp_conv(4, 1, false, "softplus"), {vec});  // [5]->[5,1,1]
  int mix = g.add_layer("mix", "dense", hp_dense(16, "relu", "concat", 0.3f), {ssum, scat, cv, vec});
  int l1 = g.add_layer("lstm1", "lstm", hp_lstm(12, 0.2f), {mix});
  int d2 = g.add_layer("d2", "dense", hp_dense(9, "sigmoid"), {mix, vec});
  int fsum = g.add_identity("flat_sum", {l1, d2}, "sum");                          // project 12 -> 9
  int l2 = g.add_layer("lstm2", "lstm", hp_lstm(7), {fsum, vec}, "concat");
  int l3 = g.add_layer("lstm3", "lstm", hp_lstm(7), {l2});
  g.add_cell_skip(l1, l2);  // forward, projected 12 -> 7
  g.add_cell_skip(l3, l1);  // feedback (previous step), projected 7 -> 12
  g.add_cell_skip(l2, l3);  // forward, same size (no projection)
  int body = g.add_identity("body", {l3, d2}, "concat");
  g.add_output("head:policy", body, 4, "linear", 0.1f);
  g.add_output("head:value", body, 1);
  return std::unique_ptr<Network>(new Network(g, BuildOptions(), 11));
}

std::vector<Case> zoo(int count) {
  CoDeepNEATConfig cfg;
  cfg.io.inputs.push_back({"image", {2, 8, 8}});
  cfg.io.inputs.push_back({"vec", {5}});
  cfg.io.outputs.push_back({"policy", 4, "linear", 0.1f});
  cfg.io.outputs.push_back({"value", 1});
  cfg.module_node_space = spaces::mixed_layers();
  cfg.module_node_space.find("units")->hi = 24;
  cfg.module_node_space.find("filters")->hi = 12;
  cfg.modules.size = 16;
  cfg.modules.mutation.add_node = 0.6;
  cfg.modules.mutation.add_connection = 0.4;
  cfg.modules.mutation.add_cell_skip = 0.5;
  cfg.modules.mutation.max_nodes = 6;
  cfg.blueprints.size = 8;
  cfg.blueprints.mutation.add_node = 0.5;
  cfg.blueprints.mutation.add_connection = 0.4;
  cfg.blueprints.mutation.max_nodes = 4;
  cfg.networks_per_generation = 16;
  cfg.seed = 123;
  CoDeepNEAT evo(cfg);
  // Reward size so the structures grow (no training needed to test export).
  evo.run(5, [](EvalContext& ctx) { return static_cast<double>(ctx.net.nodes().size()); });
  std::vector<Case> cases;
  Rng rng(7);
  for (int i = 0; i < count; ++i) {
    AssembledGenome ag = i == 0 ? evo.best() : evo.sample_assembly(i % evo.blueprints().genomes().size(), rng);
    cases.push_back({"zoo_" + std::to_string(i), evo.build(ag, 100 + i)});
  }
  return cases;
}

std::unique_ptr<Network> cartpole() {
  rl::EnvFactory make_env = [] { return std::unique_ptr<rl::Environment>(new rl::CartPole()); };
  CoDeepNEATConfig cfg;
  cfg.io = rl::actor_critic_io(*make_env());
  cfg.module_node_space = spaces::dense_layers(8, 32);
  cfg.module_node_space.find("dropout")->hi = 0.0;
  cfg.modules.size = 12;
  cfg.blueprints.size = 6;
  cfg.networks_per_generation = 12;
  cfg.num_threads = 8;
  cfg.seed = 5;
  CoDeepNEAT evo(cfg);
  evo.run(2, [&](EvalContext& ctx) {
    rl::A2CConfig ac;
    ac.total_steps = 8000;
    rl::A2CTrainer trainer(ctx.net, make_env, ac, ctx.rng.next_u64());
    trainer.apply_globals(ctx.globals);
    trainer.train();
    return rl::evaluate_policy(ctx.net, make_env, 3, true, ctx.rng.next_u64());
  });
  auto net = evo.build(evo.best(), 1);
  rl::A2CConfig ac;
  ac.total_steps = 8000;
  rl::A2CTrainer trainer(*net, make_env, ac, 3);
  trainer.apply_globals(evo.best().globals());
  trainer.train();
  std::cout << "cartpole champion return after retraining: " << rl::evaluate_policy(*net, make_env, 5, true, 1)
            << "\n";
  return net;
}

std::vector<Case> all_cases() {
  std::vector<Case> cases;
  cases.push_back({"handbuilt", handbuilt()});
  for (auto& c : zoo(8)) cases.push_back(std::move(c));
  cases.push_back({"cartpole", cartpole()});
  return cases;
}

std::vector<std::vector<Tensor>> random_steps(const Network& net, int steps, int batch, Rng& rng) {
  std::vector<std::vector<Tensor>> r(steps);
  for (auto& s : r)
    for (const auto& shape : net.input_shapes()) {
      Shape b = shape;
      b.insert(b.begin(), batch);
      s.push_back(Tensor::randn(b, rng));
    }
  return r;
}

}  // namespace

int main(int argc, char** argv) {
  std::string dir = "torch_export";
  bool check_import = false;
  for (int i = 1; i < argc; ++i) {
    if (std::strcmp(argv[i], "--check-import") == 0)
      check_import = true;
    else
      dir = argv[i];
  }
  std::filesystem::create_directories(dir);
  std::vector<Case> cases = all_cases();

  if (!check_import) {
    Rng rng(42);
    for (auto& c : cases) {
      std::string prefix = dir + "/" + c.name;
      TorchExportOptions opt;
      opt.class_name = "EvolvedNet";
      opt.description = "Case '" + c.name + "' from examples/export_pytorch.cpp.";
      export_torch(*c.net, prefix, opt);
      export_torch_reference(*c.net, random_steps(*c.net, 4, 3, rng), prefix + ".reference.safetensors");
      std::cout << "exported " << prefix << ".py (" << c.net->num_parameters() << " params, "
                << c.net->nodes().size() << " nodes" << (c.net->is_recurrent() ? ", recurrent" : "") << ")\n";
    }
    std::cout << "\nverify with:\n  python tools/verify_torch_export.py --dir " << dir << " --write-back\n"
              << "then check weights coming back from PyTorch with:\n  export_pytorch " << dir
              << " --check-import\n";
    return 0;
  }

  // Load weights updated in PyTorch back into the C++ networks and compare outputs.
  bool ok = true;
  for (auto& c : cases) {
    std::string prefix = dir + "/" + c.name;
    import_torch_weights(*c.net, prefix + ".torch_trained.safetensors");
    std::map<std::string, Tensor> ref, expected;
    for (auto& t : load_safetensors(prefix + ".reference.safetensors")) ref[t.name] = t.tensor;
    for (auto& t : load_safetensors(prefix + ".torch_outputs.safetensors")) expected[t.name] = t.tensor;
    std::vector<Tensor> inputs;
    for (size_t k = 0; k < c.net->input_names().size(); ++k) inputs.push_back(ref.at("step0.input" + std::to_string(k)));
    c.net->eval();
    c.net->reset_state(inputs[0].shape()[0]);
    std::vector<Tensor> outs;
    {
      NoGradGuard ng;
      outs = c.net->forward(inputs);
    }
    double worst = 0;
    for (size_t k = 0; k < outs.size(); ++k) {
      const Tensor& e = expected.at("output" + std::to_string(k));
      for (size_t i = 0; i < e.numel(); ++i)
        worst = std::max(worst, std::fabs(double(outs[k].data()[i]) - e.data()[i]) / (1.0 + std::fabs(e.data()[i])));
    }
    bool pass = worst < 1e-4;
    ok = ok && pass;
    std::cout << (pass ? "OK   " : "FAIL ") << c.name << "  max rel diff vs PyTorch-updated model: " << worst << "\n";
  }
  std::cout << (ok ? "all imports match" : "MISMATCH") << "\n";
  return ok ? 0 : 1;
}
