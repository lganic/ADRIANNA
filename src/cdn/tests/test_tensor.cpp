// Tensor engine: gradients of every operation against finite differences, hooks,
// no-grad mode, and very deep graphs.

#include "test_util.hpp"

using namespace cdn;

namespace {
Tensor param(const Shape& s, Rng& rng, float scale = 1.0f) { return Tensor::randn(s, rng, scale, true); }
}  // namespace

TEST(tensor_basics) {
  Tensor t({2, 3}, {1, 2, 3, 4, 5, 6});
  CHECK(t.numel() == 6);
  CHECK(t.at({1, 2}) == 6.0f);
  t.at({0, 0}) = 9.0f;
  CHECK(t.data()[0] == 9.0f);
  CHECK_THROWS(Tensor({2, 2}, {1, 2, 3}));
  Tensor r = reshape(t, {3, -1});
  CHECK((r.shape() == Shape{3, 2}));
  CHECK(sum(t).item() == 9 + 2 + 3 + 4 + 5 + 6);
}

TEST(grad_broadcast_binary_ops) {
  Rng rng(1);
  Tensor a = param({2, 3}, rng), b = param({3}, rng), c = param({2, 1}, rng);
  Tensor pos = Tensor::uniform({2, 3}, rng, 0.5f, 2.0f, true);
  gradcheck([&] { return sum(mul(add(a, b), c)); }, {a, b, c}, 2e-2, 1e-2f, 24, "add/mul");
  gradcheck([&] { return sum(div(a, pos)); }, {a, pos}, 2e-2, 1e-2f, 24, "div");
  gradcheck([&] { return sum(sub(maximum(a, c), minimum(b, a))); }, {a, b, c}, 2e-2, 1e-3f, 24, "max/min");
  gradcheck([&] { return sum((a * 2.0f + 1.0f) / 3.0f - b); }, {a, b}, 2e-2, 1e-2f, 24, "scalar ops");
  gradcheck([&] { return sum(rdiv(1.0f, pos) + rsub(2.0f, a)); }, {a, pos}, 2e-2, 1e-2f, 24, "rdiv/rsub");
}

TEST(grad_unary_ops) {
  Rng rng(2);
  Tensor x = param({3, 4}, rng);
  Tensor pos = Tensor::uniform({3, 4}, rng, 0.3f, 2.0f, true);
  for (const auto& act : activation_names())
    gradcheck([&] { return sum(square(activation(x, act))); }, {x}, 3e-2, 1e-3f, 24, act);
  gradcheck([&] { return sum(exp(x)); }, {x}, 2e-2, 1e-2f, 24, "exp");
  gradcheck([&] { return sum(log(pos)); }, {pos}, 2e-2, 1e-3f, 24, "log");
  gradcheck([&] { return sum(sqrt(pos)); }, {pos}, 2e-2, 1e-3f, 24, "sqrt");
  gradcheck([&] { return sum(pow(pos, 1.7f)); }, {pos}, 2e-2, 1e-3f, 24, "pow");
  gradcheck([&] { return sum(abs(x) * x); }, {x}, 2e-2, 1e-3f, 24, "abs");
  gradcheck([&] { return sum(clamp(x, -0.5f, 0.5f) * x); }, {x}, 2e-2, 1e-3f, 24, "clamp");
  gradcheck([&] { return sum(x.clone() * x); }, {x}, 2e-2, 1e-2f, 24, "clone");
}

TEST(grad_linear_algebra) {
  Rng rng(3);
  Tensor a = param({3, 4}, rng), b = param({4, 5}, rng);
  Tensor x = param({2, 4}, rng), w = param({3, 4}, rng), bias = param({3}, rng);
  gradcheck([&] { return sum(square(matmul(a, b))); }, {a, b}, 2e-2, 1e-2f, 24, "matmul");
  gradcheck([&] { return sum(square(linear(x, w, bias))); }, {x, w, bias}, 2e-2, 1e-2f, 24, "linear");
  gradcheck([&] { return sum(square(linear(x, w, Tensor()))); }, {x, w}, 2e-2, 1e-2f, 24, "linear-nobias");
  // Value check.
  Tensor m = matmul(Tensor({1, 2}, {1, 2}), Tensor({2, 2}, {3, 4, 5, 6}));
  CHECK(m.data()[0] == 13.0f && m.data()[1] == 16.0f);
}

TEST(grad_reductions_and_shapes) {
  Rng rng(4);
  Tensor x = param({2, 3, 4}, rng);
  Tensor y = param({2, 2, 4}, rng);
  gradcheck([&] { return sum(square(sum(x, 1))); }, {x}, 2e-2, 1e-2f, 24, "sum axis");
  gradcheck([&] { return sum(square(mean(x, -1, true))); }, {x}, 2e-2, 1e-2f, 24, "mean axis");
  gradcheck([&] { return sum(square(max(x, 2))); }, {x}, 2e-2, 1e-3f, 24, "max axis");
  gradcheck([&] { return mean(square(x)); }, {x}, 2e-2, 1e-2f, 24, "mean");
  gradcheck([&] { return sum(square(concat({x, y}, 1)) * 0.5f); }, {x, y}, 2e-2, 1e-2f, 24, "concat");
  gradcheck([&] { return sum(square(narrow(x, 2, 1, 2))); }, {x}, 2e-2, 1e-2f, 24, "narrow");
  gradcheck([&] { return sum(square(flatten(x)) * 0.5f); }, {x}, 2e-2, 1e-2f, 24, "flatten");
  gradcheck([&] { return sum(square(stack({x, x * 2.0f}))); }, {x}, 2e-2, 1e-2f, 24, "stack");
  CHECK((concat({x, y}, 1).shape() == Shape{2, 5, 4}));
}

TEST(grad_softmax_family) {
  Rng rng(5);
  Tensor x = param({3, 5}, rng);
  Tensor w = param({3, 5}, rng);
  std::vector<int> idx = {0, 4, 2};
  gradcheck([&] { return sum(softmax(x) * w); }, {x}, 2e-2, 1e-2f, 24, "softmax");
  gradcheck([&] { return sum(log_softmax(x) * w); }, {x}, 2e-2, 1e-2f, 24, "log_softmax");
  gradcheck([&] { return sum(gather(x, idx) * 3.0f); }, {x}, 2e-2, 1e-2f, 24, "gather");
  gradcheck([&] { return cross_entropy(x, idx); }, {x}, 2e-2, 1e-2f, 24, "cross_entropy");
  Tensor s = softmax(x);
  for (int r = 0; r < 3; ++r) {
    float t = 0;
    for (int j = 0; j < 5; ++j) t += s.at({r, j});
    CHECK_NEAR(t, 1.0, 1e-5);
  }
}

TEST(grad_conv_and_pool) {
  Rng rng(6);
  Tensor x = param({2, 3, 6, 5}, rng);
  Tensor w = param({4, 3, 3, 3}, rng, 0.3f), b = param({4}, rng);
  Tensor w1 = param({2, 3, 1, 1}, rng);
  gradcheck([&] { return sum(square(conv2d(x, w, b, 1, 1))) * 0.1f; }, {x, w, b}, 3e-2, 1e-2f, 24, "conv s1p1");
  gradcheck([&] { return sum(square(conv2d(x, w, b, 2, 0))) * 0.1f; }, {x, w, b}, 3e-2, 1e-2f, 24, "conv s2p0");
  gradcheck([&] { return sum(square(conv2d(x, w1, Tensor(), 1, 0))); }, {x, w1}, 3e-2, 1e-2f, 24, "conv 1x1");
  gradcheck([&] { return sum(square(max_pool2d(x, 2, 2))); }, {x}, 2e-2, 1e-3f, 24, "max_pool");
  gradcheck([&] { return sum(square(adaptive_max_pool2d(x, 4, 3))); }, {x}, 2e-2, 1e-3f, 24, "adaptive_pool");
  CHECK((conv2d(x, w, b, 1, 1).shape() == Shape{2, 4, 6, 5}));
  CHECK((conv2d(x, w, b, 2, 0).shape() == Shape{2, 4, 2, 2}));
  CHECK((max_pool2d(x, 2, 2).shape() == Shape{2, 3, 3, 2}));
  // Known value: 1x1 identity kernel.
  Tensor img({1, 1, 2, 2}, {1, 2, 3, 4});
  Tensor k({1, 1, 1, 1}, {2});
  CHECK(conv2d(img, k, Tensor(), 1, 0).data()[3] == 8.0f);
}

TEST(grad_dropout_and_losses) {
  Rng rng(7);
  Tensor x = param({4, 6}, rng), t = param({4, 6}, rng);
  gradcheck(
      [&] {
        Rng r(99);  // same mask on every call
        return sum(square(dropout(x, 0.4f, r, true)));
      },
      {x}, 2e-2, 1e-2f, 24, "dropout");
  Rng r0(1);
  CHECK(dropout(x, 0.5f, r0, false).is_same(x));  // eval mode: identity
  gradcheck([&] { return mse_loss(x, t); }, {x, t}, 2e-2, 1e-2f, 24, "mse");
  gradcheck([&] { return huber_loss(x * 3.0f, t, 1.0f); }, {x, t}, 2e-2, 1e-3f, 24, "huber");
}

TEST(hooks_observe_and_modify_gradients) {
  Tensor w({3}, {1, 2, 3}, true);
  Tensor h = mul(w, 2.0f);  // intermediate activation
  std::vector<float> seen;
  h.register_hook([&](Tensor& g) {
    seen = g.data();
    for (auto& v : g.data()) v *= 10.0f;  // amplify the gradient flowing back
  });
  Tensor loss = sum(h);
  loss.backward();
  CHECK(seen.size() == 3 && seen[0] == 1.0f);
  CHECK(w.grad_data()[0] == 20.0f);  // 1 * 10 * 2
  CHECK(h.has_grad() && h.grad_data()[1] == 10.0f);  // activation grad retained (post-hook)

  // Leaf hooks see the contribution of the current pass only; grads accumulate.
  std::vector<float> contrib;
  int id = w.register_hook([&](Tensor& g) { contrib = g.data(); });
  sum(mul(w, 3.0f)).backward();
  CHECK(contrib[0] == 3.0f);
  CHECK(w.grad_data()[0] == 23.0f);
  w.remove_hook(id);
  w.zero_grad();
  CHECK(!w.has_grad());

  // Hook can replace the gradient entirely.
  Tensor v({2}, {1, 1}, true);
  Tensor u = v * v;
  u.register_hook([](Tensor& g) { g = Tensor({2}, {0.0f, 5.0f}); });
  sum(u).backward();
  CHECK(v.grad_data()[0] == 0.0f && v.grad_data()[1] == 10.0f);
}

TEST(no_grad_and_detach) {
  Tensor w({2}, {1, 2}, true);
  {
    NoGradGuard g;
    Tensor y = w * 3.0f;
    CHECK(!y.requires_grad());
  }
  Tensor y = w * 3.0f;
  CHECK(y.requires_grad());
  Tensor d = y.detach();
  CHECK(!d.requires_grad() && d.data()[1] == 6.0f);
  CHECK_THROWS(d.backward());
}

TEST(deep_graph_no_stack_overflow) {
  Tensor w = Tensor::scalar(1.0f, true);
  Tensor acc = w;
  for (int i = 0; i < 200000; ++i) acc = add(acc, 0.0f);
  acc.backward();
  CHECK(w.grad_data()[0] == 1.0f);
  acc = Tensor();  // releases a 200k-deep chain iteratively
  CHECK(w.defined());
}
