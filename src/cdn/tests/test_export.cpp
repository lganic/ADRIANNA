// safetensors round trips, parameter name mapping, and weight import. The comparison
// against real PyTorch lives in tools/verify_torch_export.py.

#include <cstdio>
#include <cstring>
#include <fstream>
#include <set>

#include "test_util.hpp"

using namespace cdn;

namespace {

NetworkGraph export_graph() {
  NetworkGraph g;
  HyperparamValues conv, dense, lstm;
  conv.set_choice("type", "conv2d").set_int("filters", 3).set_int("kernel_size", 3).set_bool("max_pool", true);
  dense.set_choice("type", "dense").set_int("units", 6).set_choice("activation", "tanh");
  lstm.set_choice("type", "lstm").set_int("units", 5);
  int img = g.add_input("input:img", {1, 6, 6});
  int vec = g.add_input("input:vec", {4});
  int c = g.add_layer("c", "conv2d", conv, {img});
  int d = g.add_layer("d", "dense", dense, {c, vec});
  int l1 = g.add_layer("l1", "lstm", lstm, {d});
  int s = g.add_identity("s", {l1, d}, "sum");
  int l2 = g.add_layer("l2", "lstm", lstm, {s});
  g.add_cell_skip(l2, l1);
  g.add_output("head:pi", l2, 3);
  g.add_output("head:v", s, 1);
  return g;
}

void write_raw_safetensors(const std::string& path, const std::string& header, const std::vector<unsigned char>& data) {
  std::ofstream f(path, std::ios::binary);
  uint64_t n = header.size();
  for (int i = 0; i < 8; ++i) f.put(static_cast<char>((n >> (8 * i)) & 0xFF));
  f << header;
  f.write(reinterpret_cast<const char*>(data.data()), static_cast<std::streamsize>(data.size()));
}

}  // namespace

TEST(safetensors_roundtrip) {
  Rng rng(1);
  std::vector<NamedTensor> ts = {{"a.weight", Tensor::randn({3, 4}, rng)},
                                 {"scalar", Tensor::scalar(2.5f)},
                                 {"b", Tensor::randn({2, 1, 3, 3}, rng)}};
  std::string path = "cdn_test.safetensors";
  save_safetensors(ts, path, {{"format", "test"}, {"quote", "it's \"ok\""}});
  std::map<std::string, std::string> meta;
  auto back = load_safetensors(path, &meta);
  std::remove(path.c_str());
  CHECK(back.size() == 3);
  for (size_t i = 0; i < ts.size(); ++i) {
    CHECK(back[i].name == ts[i].name);
    CHECK(back[i].tensor.shape() == ts[i].tensor.shape());
    CHECK(back[i].tensor.data() == ts[i].tensor.data());
  }
  CHECK(meta.at("format") == "test" && meta.at("quote") == "it's \"ok\"");
}

TEST(safetensors_reads_other_dtypes) {
  // F16 1.0 = 0x3C00, BF16 1.5 = 0x3FC0, F64 -2.25.
  std::string header =
      "{\"h\":{\"dtype\":\"F16\",\"shape\":[1],\"data_offsets\":[0,2]},"
      "\"bf\":{\"dtype\":\"BF16\",\"shape\":[1],\"data_offsets\":[2,4]},"
      "\"d\":{\"dtype\":\"F64\",\"shape\":[1],\"data_offsets\":[4,12]}}";
  std::vector<unsigned char> data = {0x00, 0x3C, 0xC0, 0x3F};
  double v = -2.25;
  unsigned char b[8];
  std::memcpy(b, &v, 8);
  data.insert(data.end(), b, b + 8);
  std::string path = "cdn_test_dtypes.safetensors";
  write_raw_safetensors(path, header, data);
  auto t = load_safetensors(path);
  std::remove(path.c_str());
  CHECK(t.size() == 3);
  CHECK(t[0].tensor.item() == 1.0f);
  CHECK(t[1].tensor.item() == 1.5f);
  CHECK(t[2].tensor.item() == -2.25f);
}

TEST(torch_parameter_mapping_complete) {
  Network net(export_graph(), BuildOptions(), 3);
  auto names = torch_parameter_names(net);
  CHECK(names.size() == net.named_parameters().size());
  std::set<std::string> keys;
  for (const auto& kv : names) {
    CHECK(kv.second.rfind("nodes.", 0) == 0);
    CHECK(keys.insert(kv.second).second);
  }
  CHECK(names.at("c.weight") == "nodes.c.conv.weight");
  CHECK(names.at("d.weight") == "nodes.d.linear.weight");
  CHECK(names.at("head:pi.bias") == "nodes.head_pi.linear.bias");
  CHECK(names.at("l1.skip0.gate_weight") == "nodes.l1.skips.0.gate.weight");
  CHECK(names.at("s.merge.proj1") == "nodes.s.merge.proj.1.weight");  // d: 6 -> 5
  std::string src = torch_module_source(net);
  CHECK(src.find("class EvolvedNet(CoDeepNEATModule)") != std::string::npos);
  CHECK(src.find("SPEC = json.loads(r'''") != std::string::npos);
  CHECK(src.find("\"head_pi\"") != std::string::npos);
  TorchExportOptions bad;
  bad.class_name = "not valid";
  CHECK_THROWS(torch_module_source(net, bad));
}

TEST(torch_weights_roundtrip_into_cpp) {
  Network a(export_graph(), BuildOptions(), 1), b(export_graph(), BuildOptions(), 2);
  std::string path = "cdn_test_torch_weights.safetensors";
  save_torch_weights(a, path);
  import_torch_weights(b, path);
  std::remove(path.c_str());
  Rng rng(4);
  std::vector<Tensor> x = {Tensor::randn({2, 1, 6, 6}, rng), Tensor::randn({2, 4}, rng)};
  a.eval();
  b.eval();
  for (int t = 0; t < 3; ++t) {
    auto ya = a.forward(x), yb = b.forward(x);
    for (size_t k = 0; k < ya.size(); ++k) CHECK(ya[k].data() == yb[k].data());
  }
}

TEST(torch_export_rejects_unknown_layers) {
  struct Custom : Layer {
    Custom(const std::string& n, const Shape& in) : Layer(n, "custom_export_test", in, HyperparamValues()) {
      out_shape_ = in;
    }
    Tensor forward(const Tensor& x, LayerContext&) override { return x; }
  };
  LayerRegistry::instance().add("custom_export_test", [](const std::string& n, const HyperparamValues&,
                                                         const Shape& in, const BuildOptions&, Rng&) {
    return std::unique_ptr<Layer>(new Custom(n, in));
  });
  NetworkGraph g;
  int in = g.add_input("x", {3});
  int c = g.add_layer("c", "custom_export_test", HyperparamValues(), {in});
  g.add_output("head:y", c, 2);
  Network net(g, BuildOptions(), 1);
  CHECK_THROWS(torch_module_source(net));
}
