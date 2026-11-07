#pragma once
// Seedable random number generator used everywhere in the library so that runs
// are reproducible (per platform) given a seed.

#include <algorithm>
#include <cstdint>
#include <random>
#include <stdexcept>
#include <vector>

namespace cdn {

/// Thin wrapper around std::mt19937_64 with the sampling helpers the library needs.
///
/// Not thread-safe: give every thread its own instance. CoDeepNEAT does this for
/// you by handing each evaluation a dedicated `EvalContext::rng`.
///
/// Sequences are reproducible for a given seed on one platform. They can differ
/// between standard libraries, because the std:: distributions are
/// implementation-defined.
class Rng {
 public:
  explicit Rng(uint64_t seed = 0x5EEDu) : eng_(seed) {}

  /// Restart the generator from `s`.
  void seed(uint64_t s) { eng_.seed(s); }

  /// Uniform in [0, 1).
  double uniform() { return std::uniform_real_distribution<double>(0.0, 1.0)(eng_); }
  /// Uniform in [lo, hi).
  double uniform(double lo, double hi) { return lo + (hi - lo) * uniform(); }
  /// Gaussian sample.
  double normal(double mean = 0.0, double stddev = 1.0) {
    return std::normal_distribution<double>(mean, stddev)(eng_);
  }
  /// Uniform integer in [lo, hi] (inclusive).
  int randint(int lo, int hi) { return std::uniform_int_distribution<int>(lo, hi)(eng_); }
  /// Uniform index in [0, n).
  size_t index(size_t n) {
    if (n == 0) throw std::invalid_argument("Rng::index: empty range");
    return std::uniform_int_distribution<size_t>(0, n - 1)(eng_);
  }
  /// True with probability p.
  bool bernoulli(double p) { return uniform() < p; }
  /// Raw 64-bit draw; handy for deriving seeds of child generators.
  uint64_t next_u64() { return eng_(); }

  /// Uniformly chosen element. Throws std::invalid_argument if `v` is empty.
  template <class T>
  const T& choice(const std::vector<T>& v) {
    return v[index(v.size())];
  }

  /// Sample an index from (unnormalised, non-negative) probabilities. If they sum
  /// to zero the index is drawn uniformly.
  template <class T>
  int categorical(const std::vector<T>& probs) {
    double total = 0.0;
    for (const auto& p : probs) total += static_cast<double>(p);
    if (!(total > 0.0)) return static_cast<int>(index(probs.size()));
    double r = uniform() * total;
    for (size_t i = 0; i < probs.size(); ++i) {
      r -= static_cast<double>(probs[i]);
      if (r < 0.0) return static_cast<int>(i);
    }
    return static_cast<int>(probs.size()) - 1;
  }

  /// In-place random permutation.
  template <class T>
  void shuffle(std::vector<T>& v) {
    std::shuffle(v.begin(), v.end(), eng_);
  }

  /// The underlying engine, for use with other std:: distributions.
  std::mt19937_64& engine() { return eng_; }

 private:
  std::mt19937_64 eng_;
};

}  // namespace cdn
