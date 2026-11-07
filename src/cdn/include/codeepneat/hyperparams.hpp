#pragma once
// Hyperparameter tables. In DeepNEAT/CoDeepNEAT every node gene carries "a table
// of real and binary valued hyperparameters that are mutated through uniform
// Gaussian distribution and random bit-flipping" (Sec. 3.1). The blueprint also
// carries global hyperparameters (learning rate, optimizer, ...). Both are
// represented here as a typed, user-definable search space.

#include <initializer_list>
#include <map>
#include <string>
#include <vector>

#include "codeepneat/rng.hpp"

namespace cdn {

enum class ParamKind { Float, Int, Bool, Choice };

/// One typed value. Name and choice strings must not contain ',', ':' or
/// whitespace, because the genome file format relies on those as separators.
struct HPValue {
  ParamKind kind = ParamKind::Float;
  double num = 0.0;  // Float / Int / Bool
  std::string str;   // Choice
};

/// A concrete assignment of hyperparameter values (a node's "table").
class HyperparamValues {
 public:
  bool has(const std::string& name) const { return v_.count(name) != 0; }
  bool empty() const { return v_.empty(); }

  /// Numeric read. Choice values are parsed as numbers when possible.
  double f(const std::string& name, double def = 0.0) const;
  int i(const std::string& name, int def = 0) const;
  bool b(const std::string& name, bool def = false) const;
  /// String read. Numbers are formatted.
  std::string s(const std::string& name, const std::string& def = "") const;

  HyperparamValues& set_float(const std::string& name, double v);
  HyperparamValues& set_int(const std::string& name, int v);
  HyperparamValues& set_bool(const std::string& name, bool v);
  HyperparamValues& set_choice(const std::string& name, const std::string& v);
  void erase(const std::string& name) { v_.erase(name); }

  const std::map<std::string, HPValue>& values() const { return v_; }
  std::map<std::string, HPValue>& values() { return v_; }

  /// Human readable "name=value" list.
  std::string to_string() const;
  /// Whitespace-free token serialisation (used by genome I/O).
  std::string serialize() const;
  static HyperparamValues deserialize(const std::string& text);

  bool operator==(const HyperparamValues& o) const;
  bool operator!=(const HyperparamValues& o) const { return !(*this == o); }

 private:
  std::map<std::string, HPValue> v_;
};

/// Definition of one searchable parameter. Build with the static factories, which
/// validate the range and can be chained:
///   ParamSpec::integer("units", 8, 64).only_for({"dense", "lstm"})
struct ParamSpec {
  std::string name;
  ParamKind kind = ParamKind::Float;
  double lo = 0.0, hi = 1.0;  ///< inclusive range of Float and Int params
  bool log_scale = false;     ///< sample, mutate and measure distance in log space
  std::vector<std::string> choices;  ///< values of a Choice param
  /// Gaussian mutation std-dev as a fraction of the (log-)range.
  double mutation_std = 0.15;
  /// If non-empty the parameter only matters when the node's "type" value is in
  /// this list (used to ignore e.g. kernel_size on dense nodes in distances).
  std::vector<std::string> when_type;

  static ParamSpec real(const std::string& name, double lo, double hi, bool log_scale = false);
  static ParamSpec integer(const std::string& name, int lo, int hi);
  static ParamSpec boolean(const std::string& name);
  static ParamSpec choice(const std::string& name, const std::vector<std::string>& choices);

  ParamSpec& only_for(const std::vector<std::string>& types) {
    when_type = types;
    return *this;
  }
  ParamSpec& with_mutation_std(double s) {
    mutation_std = s;
    return *this;
  }
};

/// An ordered set of ParamSpecs: everything a hyperparameter table may contain.
/// The library only reads the keys listed in spaces.hpp; any other key is simply
/// evolved and passed along for your evaluator or layer to read.
class HyperparamSpace {
 public:
  HyperparamSpace() = default;
  HyperparamSpace(std::initializer_list<ParamSpec> specs);

  HyperparamSpace& add(const ParamSpec& spec);  // replaces an existing spec with the same name
  bool remove(const std::string& name);
  const std::vector<ParamSpec>& specs() const { return specs_; }
  const ParamSpec* find(const std::string& name) const;
  ParamSpec* find(const std::string& name);
  bool empty() const { return specs_.empty(); }

  /// Draws every parameter uniformly (log-uniformly for log_scale floats).
  HyperparamValues sample(Rng& rng) const;
  /// Fill in parameters missing from `v` (e.g. after the space was extended).
  void complete(HyperparamValues& v, Rng& rng) const;
  /// Gaussian perturbation of numeric params, bit flips of booleans, re-draw of
  /// categorical params. Each param mutates with `per_param_prob`; at least one
  /// param is always changed (if the space is non-empty).
  void mutate(HyperparamValues& v, Rng& rng, double per_param_prob) const;
  /// Mutates one parameter; samples it if it is missing.
  void mutate_param(HyperparamValues& v, const ParamSpec& spec, Rng& rng) const;
  /// Uniform crossover.
  HyperparamValues crossover(const HyperparamValues& a, const HyperparamValues& b, Rng& rng) const;
  /// Mean normalised per-parameter distance in [0, 1].
  double distance(const HyperparamValues& a, const HyperparamValues& b) const;
  /// True if `spec` applies to a node with these values (see ParamSpec::when_type).
  static bool active(const ParamSpec& spec, const HyperparamValues& v);

 private:
  std::vector<ParamSpec> specs_;
};

}  // namespace cdn
