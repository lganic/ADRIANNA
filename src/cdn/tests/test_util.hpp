#pragma once
// Minimal self-contained test harness + numerical gradient checker.

#include <algorithm>
#include <cmath>
#include <functional>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "codeepneat/all.hpp"

struct TestCase {
  const char* name;
  std::function<void()> fn;
};

inline std::vector<TestCase>& test_registry() {
  static std::vector<TestCase> r;
  return r;
}

struct TestRegistrar {
  TestRegistrar(const char* n, std::function<void()> f) { test_registry().push_back({n, std::move(f)}); }
};

struct TestFailure {
  std::string msg;
};

#define TEST(name)                                            \
  static void name();                                         \
  static TestRegistrar registrar_##name(#name, name);         \
  static void name()

#define CHECK(cond)                                                                                         \
  do {                                                                                                      \
    if (!(cond)) throw TestFailure{std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": CHECK(" #cond \
                                   ") failed"};                                                             \
  } while (0)

#define CHECK_NEAR(a, b, tol)                                                                              \
  do {                                                                                                     \
    double va_ = (a), vb_ = (b);                                                                           \
    if (!(std::fabs(va_ - vb_) <= (tol))) {                                                                \
      std::ostringstream os_;                                                                              \
      os_ << __FILE__ << ":" << __LINE__ << ": CHECK_NEAR(" #a ", " #b ") failed: " << va_ << " vs " << vb_; \
      throw TestFailure{os_.str()};                                                                        \
    }                                                                                                      \
  } while (0)

#define CHECK_THROWS(expr)                                                                                \
  do {                                                                                                    \
    bool threw_ = false;                                                                                  \
    try {                                                                                                 \
      expr;                                                                                               \
    } catch (...) {                                                                                       \
      threw_ = true;                                                                                      \
    }                                                                                                     \
    if (!threw_)                                                                                          \
      throw TestFailure{std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": expected exception: " #expr}; \
  } while (0)

/// Compare analytic gradients of the scalar `loss()` w.r.t. `params` against
/// central finite differences. `loss` must rebuild the graph each call.
/// Returns the worst normalised error; throws TestFailure above `tol`.
inline double gradcheck(const std::function<cdn::Tensor()>& loss, std::vector<cdn::Tensor> params,
                        double tol = 2e-2, float eps = 1e-2f, size_t max_checks_per_param = 24,
                        const std::string& label = "") {
  for (auto& p : params) p.zero_grad();
  cdn::Tensor L = loss();
  L.backward();
  std::vector<std::vector<float>> analytic;
  for (auto& p : params) analytic.push_back(p.has_grad() ? p.grad_data() : std::vector<float>(p.numel(), 0.0f));
  double worst = 0.0;
  for (size_t k = 0; k < params.size(); ++k) {
    cdn::Tensor p = params[k];
    size_t n = p.numel();
    size_t stride = std::max<size_t>(1, n / max_checks_per_param);
    for (size_t i = 0; i < n; i += stride) {
      float orig = p.data()[i];
      p.data()[i] = orig + eps;
      double fp;
      {
        cdn::NoGradGuard g;
        fp = loss().item();
      }
      p.data()[i] = orig - eps;
      double fm;
      {
        cdn::NoGradGuard g;
        fm = loss().item();
      }
      p.data()[i] = orig;
      double num = (fp - fm) / (2.0 * eps);
      double ana = analytic[k][i];
      double err = std::fabs(num - ana) / std::max(1.0, std::max(std::fabs(num), std::fabs(ana)));
      if (err > worst) worst = err;
      if (err > tol) {
        std::ostringstream os;
        os << "gradcheck " << label << " failed for param " << k << " (" << p.name() << ") index " << i
           << ": numeric " << num << " analytic " << ana;
        throw TestFailure{os.str()};
      }
    }
  }
  return worst;
}
