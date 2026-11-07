// Networks: shapes, BPTT through cell skips, merges, hooks, freezing, state, weight I/O.

#include <cstdio>

#include "test_util.hpp"

using namespace cdn;

namespace {

HyperparamValues dense_hp(int units, const std::string& act = "tanh", const std::string& merge = "concat") {
  HyperparamValues hp;
  hp.set_choice("type", "dense").set_int("units", units).set_choice("activation", act).set_choice("merge", merge);
  return hp;
}

HyperparamValues lstm_hp(int units) {
  HyperparamValues hp;
  hp.set_choice("type", "lstm").set_int("units", units);
  return hp;
}

HyperparamValues conv_hp(int filters, int k, bool pool, const std::string& act = "tanh") {
  HyperparamValues hp;
  hp.set_choice("type", "conv2d").set_int("filters", filters).set_int("kernel_size", k).set_bool("max_pool", pool);
  hp.set_choice("activation", act);
  return hp;
}

/// Recurrent graph with a sum-merge (projection), two LSTMs and both forward and
/// feedback memory-cell skip connections.
NetworkGraph recurrent_graph() {
  NetworkGraph g;
  int in = g.add_input("obs", {4});
  int d1 = g.add_layer("d1", "dense", dense_hp(8), {in});
  int l1 = g.add_layer("l1", "lstm", lstm_hp(6), {d1});
  int d2 = g.add_layer("d2", "dense", dense_hp(5), {d1});
  int m = g.add_identity("m", {l1, d2}, "sum");
  int l2 = g.add_layer("l2", "lstm", lstm_hp(7), {m, in}, "concat");
  g.add_cell_skip(l1, l2);  // forward: uses l1's current cell
  g.add_cell_skip(l2, l1);  // feedback: uses l2's previous cell
  g.add_output("head:policy", l2, 3);
  g.add_output("head:value", l2, 1);
  return g;
}

}  // namespace

TEST(network_build_and_shapes) {
  Network net(recurrent_graph(), BuildOptions(), 3);
  CHECK(net.is_recurrent());
  CHECK(net.input_names().size() == 1 && net.output_names().size() == 2);
  CHECK((net.node("m").merged_shape == Shape{5}));  // sum merge projects 6 -> 5
  CHECK(net.has_parameter("m.merge.proj0"));
  CHECK(net.has_parameter("l2.skip0.proj"));      // 6 -> 7 projection
  CHECK(net.has_parameter("l1.skip0.gate_weight"));
  auto outs = net.forward(std::vector<Tensor>{Tensor::zeros({5, 4})});
  CHECK((outs[0].shape() == Shape{5, 3}) && (outs[1].shape() == Shape{5, 1}));
  CHECK(net.summary().find("lstm") != std::string::npos);
  CHECK(net.to_dot().find("cell") != std::string::npos);
}

TEST(network_bptt_gradcheck_with_cell_skips) {
  Network net(recurrent_graph(), BuildOptions(), 11);
  Rng rng(5);
  std::vector<Tensor> xs;
  for (int t = 0; t < 3; ++t) xs.push_back(Tensor::randn({2, 4}, rng));
  auto loss = [&] {
    net.reset_state(2);
    Tensor total = Tensor::scalar(0.0f);
    for (const auto& x : xs) {
      auto o = net.forward(std::vector<Tensor>{x});
      total = total + sum(square(o[0])) + sum(o[1]);
    }
    return total;
  };
  gradcheck(loss, net.parameters(), 3e-2, 5e-3f, 12, "recurrent net");
}

TEST(network_conv_merge_gradcheck) {
  NetworkGraph g;
  int img = g.add_input("img", {2, 6, 6});
  int vec = g.add_input("vec", {3});
  int c1 = g.add_layer("c1", "conv2d", conv_hp(4, 3, true), {img});   // -> [4,3,3]
  int c2 = g.add_layer("c2", "conv2d", conv_hp(3, 1, false), {img});  // -> [3,6,6]
  int cat = g.add_identity("cat", {c1, c2}, "concat");               // pools c2 to 3x3 -> [7,3,3]
  int sm = g.add_identity("sm", {c1, c2}, "sum");                    // 1x1 projection 3->... -> [3,3,3]
  int c3 = g.add_layer("c3", "conv2d", conv_hp(2, 3, false), {vec});  // flat input as [3,1,1]
  int mix = g.add_layer("mix", "dense", dense_hp(6), {cat, sm, c3, vec});  // spatial+flat -> flatten
  g.add_output("head:out", mix, 2);
  Network net(g, BuildOptions(), 7);
  CHECK((net.node("cat").merged_shape == Shape{7, 3, 3}));
  CHECK((net.node("sm").merged_shape == Shape{3, 3, 3}));
  CHECK((net.node("mix").merged_shape == Shape{7 * 9 + 3 * 9 + 2 + 3}));
  Rng rng(8);
  Tensor xi = Tensor::randn({2, 2, 6, 6}, rng), xv = Tensor::randn({2, 3}, rng);
  auto loss = [&] { return sum(square(net.forward(std::vector<Tensor>{xi, xv})[0])); };
  gradcheck(loss, net.parameters(), 3e-2, 5e-3f, 8, "conv net");
}

TEST(network_hooks_and_recording) {
  Network net(recurrent_graph(), BuildOptions(), 4);
  net.record_activations(true);
  Rng rng(1);
  Tensor x = Tensor::randn({3, 4}, rng);

  // Backward hook on an activation: compare with the recorded activation grad.
  std::vector<float> hook_grad;
  int bh = net.add_backward_hook("d2", [&](const NodeInfo& n, Tensor& g) {
    CHECK(n.name == "d2");
    hook_grad = g.data();
  });
  auto outs = net.forward(std::vector<Tensor>{x});
  sum(outs[0]).backward();
  Tensor act = net.activation("d2");
  CHECK(act.has_grad());
  CHECK(hook_grad == act.grad_data());
  net.remove_hook(bh);

  // Forward hook that silences a layer changes the output.
  net.reset_state(3);
  auto base = net.forward(std::vector<Tensor>{x})[0].data();
  net.reset_state(3);
  int fh = net.add_forward_hook("l2", [](const NodeInfo&, Tensor& out) { out = out * 0.0f; });
  auto silenced = net.forward(std::vector<Tensor>{x})[0];
  Tensor bias = net.parameter("head:policy.bias");
  for (size_t i = 0; i < silenced.numel(); ++i) CHECK_NEAR(silenced.data()[i], bias.data()[i % 3], 1e-6);
  CHECK(base != silenced.data());
  net.remove_hook(fh);

  // Filtered hooks see every LSTM node.
  int lstm_calls = 0;
  int fh2 = net.add_forward_hook([&](const NodeInfo&, Tensor&) { ++lstm_calls; },
                                 [](const NodeInfo& n) { return n.type == "lstm"; });
  net.forward(std::vector<Tensor>{x});
  CHECK(lstm_calls == 2);
  net.remove_hook(fh2);

  // Pre-hook sees the merged input.
  Shape seen;
  int ph = net.add_forward_pre_hook([&](const NodeInfo&, Tensor& in) { seen = in.shape(); },
                                    [](const NodeInfo& n) { return n.name == "l2"; });
  net.forward(std::vector<Tensor>{x});
  CHECK((seen == Shape{3, 9}));
  net.remove_hook(ph);

  // Parameter gradient hook scales the gradient.
  net.zero_grad();
  net.reset_state(3);
  sum(net.forward(std::vector<Tensor>{x})[0]).backward();
  std::vector<float> plain = net.parameter("d1.weight").grad_data();
  net.zero_grad();
  int pgh = net.add_parameter_hook("d1.weight", [](Tensor& g) {
    for (auto& v : g.data()) v *= 2.0f;
  });
  net.reset_state(3);
  sum(net.forward(std::vector<Tensor>{x})[0]).backward();
  auto scaled = net.parameter("d1.weight").grad_data();
  for (size_t i = 0; i < plain.size(); ++i) CHECK_NEAR(scaled[i], 2.0f * plain[i], 1e-5);
  net.remove_hook(pgh);
}

TEST(network_weight_surgery_and_freezing) {
  Network net(recurrent_graph(), BuildOptions(), 5);
  Tensor w = net.parameter("head:value.weight");
  w.fill(0.0f);
  net.parameter("head:value.bias").fill(1.5f);
  auto out = net.forward(std::vector<Tensor>{Tensor::ones({2, 4})});
  CHECK_NEAR(out[1].data()[0], 1.5, 1e-6);

  size_t frozen = net.set_trainable("d1.", false);
  CHECK(frozen == 2);
  net.zero_grad();
  sum(net.forward(std::vector<Tensor>{Tensor::ones({2, 4})})[0]).backward();
  CHECK(!net.parameter("d1.weight").has_grad());
  CHECK(net.parameter("l1.weight").has_grad());
  auto before = net.parameter("d1.weight").data();
  Adam opt(net.named_parameters(), 0.1f);
  opt.step();
  CHECK(net.parameter("d1.weight").data() == before);
  CHECK(opt.state("l1.weight") && !opt.state("l1.weight")->m.empty());
  net.set_trainable("", true);
  CHECK(net.parameter("d1.weight").requires_grad());
  size_t by_filter = net.set_trainable([](const NodeInfo& n) { return n.type == "lstm"; }, false);
  CHECK(by_filter > 0 && !net.parameter("l2.weight").requires_grad());
}

TEST(network_state_save_restore_and_mask) {
  Network net(recurrent_graph(), BuildOptions(), 6);
  Rng rng(2);
  Tensor x = Tensor::randn({2, 4}, rng);
  net.reset_state(2);
  net.forward(std::vector<Tensor>{x});
  net.forward(std::vector<Tensor>{x});
  auto saved = net.get_state();
  CHECK(saved.size() == 4);
  auto a = net.forward(std::vector<Tensor>{x})[0].data();
  net.set_state(saved);
  auto b = net.forward(std::vector<Tensor>{x})[0].data();
  CHECK(a == b);
  net.mask_state({1.0f, 0.0f});
  for (const auto& s : net.get_state()) {
    CHECK(s.tensor.at({1, 0}) == 0.0f);
  }
  net.detach_state();
  for (const auto& s : net.get_state()) CHECK(s.tensor.is_leaf());
}

TEST(network_weights_roundtrip) {
  Network a(recurrent_graph(), BuildOptions(), 1), b(recurrent_graph(), BuildOptions(), 2);
  CHECK(a.parameter("d1.weight").data() != b.parameter("d1.weight").data());
  std::string path = "cdn_test_weights.txt";
  a.save_weights(path);
  b.load_weights(path);
  std::remove(path.c_str());
  for (const auto& p : a.named_parameters()) CHECK(p.tensor.data() == b.parameter(p.name).data());
  Tensor x = Tensor::ones({1, 4});
  CHECK(a.forward(std::vector<Tensor>{x})[0].data() == b.forward(std::vector<Tensor>{x})[0].data());
}

TEST(network_rejects_bad_graphs) {
  NetworkGraph g;
  int in = g.add_input("x", {3});
  int a = g.add_layer("a", "dense", dense_hp(2), {in});
  g.nodes[a].inputs.push_back(a);  // self loop
  g.add_output("head:y", a, 1);
  CHECK_THROWS(Network(g, BuildOptions(), 1));
  NetworkGraph g2;
  int in2 = g2.add_input("x", {3});
  int bad = g2.add_layer("a", "no_such_layer", HyperparamValues(), {in2});
  g2.add_output("head:y", bad, 1);
  CHECK_THROWS(Network(g2, BuildOptions(), 1));
}

TEST(custom_layer_registration) {
  // Users can extend the search space with their own layer types.
  struct ScaleLayer : Layer {
    ScaleLayer(const std::string& n, const Shape& in) : Layer(n, "scale", in, HyperparamValues()) {
      out_shape_ = in;
      s_ = add_param("s", Tensor::ones({1}));
    }
    Tensor forward(const Tensor& x, LayerContext&) override { return mul(x, s_); }
    Tensor s_;
  };
  LayerRegistry::instance().add("scale", [](const std::string& n, const HyperparamValues&, const Shape& in,
                                            const BuildOptions&, Rng&) {
    return std::unique_ptr<Layer>(new ScaleLayer(n, in));
  });
  NetworkGraph g;
  int in = g.add_input("x", {3});
  int s = g.add_layer("s", "scale", HyperparamValues(), {in});
  g.add_output("head:y", s, 2);
  Network net(g, BuildOptions(), 1);
  CHECK(net.has_parameter("s.s"));
  sum(net.forward(Tensor::ones({2, 3}))).backward();
  CHECK(net.parameter("s.s").has_grad());
}
