// Typed hyperparameter tables: sampling, mutation, crossover, distance, serialisation.

#include "codeepneat/hyperparams.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <sstream>
#include <stdexcept>

namespace cdn {

namespace {

std::string fmt_num(double v) {
  char buf[64];
  std::snprintf(buf, sizeof(buf), "%.10g", v);
  return buf;
}

bool parse_num(const std::string& s, double& out) {
  if (s.empty()) return false;
  char* end = nullptr;
  out = std::strtod(s.c_str(), &end);
  return end && *end == '\0';
}

char kind_char(ParamKind k) {
  switch (k) {
    case ParamKind::Float: return 'f';
    case ParamKind::Int: return 'i';
    case ParamKind::Bool: return 'b';
    case ParamKind::Choice: return 'c';
  }
  return 'f';
}

}  // namespace

// ============================================================================
// HyperparamValues
// ============================================================================

double HyperparamValues::f(const std::string& name, double def) const {
  auto it = v_.find(name);
  if (it == v_.end()) return def;
  if (it->second.kind == ParamKind::Choice) {
    double d;
    if (parse_num(it->second.str, d)) return d;
    if (it->second.str == "true") return 1.0;
    if (it->second.str == "false") return 0.0;
    return def;
  }
  return it->second.num;
}

int HyperparamValues::i(const std::string& name, int def) const {
  return static_cast<int>(std::lround(f(name, static_cast<double>(def))));
}

bool HyperparamValues::b(const std::string& name, bool def) const { return f(name, def ? 1.0 : 0.0) != 0.0; }

std::string HyperparamValues::s(const std::string& name, const std::string& def) const {
  auto it = v_.find(name);
  if (it == v_.end()) return def;
  switch (it->second.kind) {
    case ParamKind::Choice: return it->second.str;
    case ParamKind::Bool: return it->second.num != 0.0 ? "true" : "false";
    case ParamKind::Int: return std::to_string(static_cast<long long>(std::llround(it->second.num)));
    case ParamKind::Float: return fmt_num(it->second.num);
  }
  return def;
}

HyperparamValues& HyperparamValues::set_float(const std::string& name, double v) {
  v_[name] = HPValue{ParamKind::Float, v, {}};
  return *this;
}
HyperparamValues& HyperparamValues::set_int(const std::string& name, int v) {
  v_[name] = HPValue{ParamKind::Int, static_cast<double>(v), {}};
  return *this;
}
HyperparamValues& HyperparamValues::set_bool(const std::string& name, bool v) {
  v_[name] = HPValue{ParamKind::Bool, v ? 1.0 : 0.0, {}};
  return *this;
}
HyperparamValues& HyperparamValues::set_choice(const std::string& name, const std::string& v) {
  v_[name] = HPValue{ParamKind::Choice, 0.0, v};
  return *this;
}

std::string HyperparamValues::to_string() const {
  std::ostringstream os;
  bool first = true;
  for (const auto& kv : v_) {
    if (!first) os << ' ';
    first = false;
    os << kv.first << '=' << s(kv.first);
  }
  return os.str();
}

std::string HyperparamValues::serialize() const {
  if (v_.empty()) return "-";
  std::ostringstream os;
  bool first = true;
  for (const auto& kv : v_) {
    if (!first) os << ',';
    first = false;
    os << kv.first << ':' << kind_char(kv.second.kind) << ':';
    if (kv.second.kind == ParamKind::Choice)
      os << kv.second.str;
    else
      os << fmt_num(kv.second.num);
  }
  return os.str();
}

HyperparamValues HyperparamValues::deserialize(const std::string& text) {
  HyperparamValues r;
  if (text == "-" || text.empty()) return r;
  std::stringstream ss(text);
  std::string item;
  while (std::getline(ss, item, ',')) {
    size_t a = item.find(':');
    size_t b = item.find(':', a == std::string::npos ? a : a + 1);
    if (a == std::string::npos || b == std::string::npos)
      throw std::runtime_error("HyperparamValues::deserialize: bad token '" + item + "'");
    std::string name = item.substr(0, a);
    char k = item[a + 1];
    std::string val = item.substr(b + 1);
    double d = 0.0;
    if (k == 'c') {
      r.set_choice(name, val);
      continue;
    }
    if (!parse_num(val, d)) throw std::runtime_error("HyperparamValues::deserialize: bad number '" + val + "'");
    if (k == 'f')
      r.set_float(name, d);
    else if (k == 'i')
      r.set_int(name, static_cast<int>(std::lround(d)));
    else if (k == 'b')
      r.set_bool(name, d != 0.0);
    else
      throw std::runtime_error("HyperparamValues::deserialize: bad kind");
  }
  return r;
}

bool HyperparamValues::operator==(const HyperparamValues& o) const {
  if (v_.size() != o.v_.size()) return false;
  for (const auto& kv : v_) {
    auto it = o.v_.find(kv.first);
    if (it == o.v_.end() || it->second.kind != kv.second.kind) return false;
    if (kv.second.kind == ParamKind::Choice) {
      if (kv.second.str != it->second.str) return false;
    } else if (std::fabs(kv.second.num - it->second.num) > 1e-9 * (1.0 + std::fabs(kv.second.num))) {
      return false;
    }
  }
  return true;
}

// ============================================================================
// ParamSpec
// ============================================================================

ParamSpec ParamSpec::real(const std::string& name, double lo, double hi, bool log_scale) {
  if (hi < lo) throw std::invalid_argument("ParamSpec::real: hi < lo");
  if (log_scale && lo <= 0.0) throw std::invalid_argument("ParamSpec::real: log scale requires lo > 0");
  ParamSpec s;
  s.name = name;
  s.kind = ParamKind::Float;
  s.lo = lo;
  s.hi = hi;
  s.log_scale = log_scale;
  return s;
}
ParamSpec ParamSpec::integer(const std::string& name, int lo, int hi) {
  if (hi < lo) throw std::invalid_argument("ParamSpec::integer: hi < lo");
  ParamSpec s;
  s.name = name;
  s.kind = ParamKind::Int;
  s.lo = lo;
  s.hi = hi;
  return s;
}
ParamSpec ParamSpec::boolean(const std::string& name) {
  ParamSpec s;
  s.name = name;
  s.kind = ParamKind::Bool;
  return s;
}
ParamSpec ParamSpec::choice(const std::string& name, const std::vector<std::string>& choices) {
  if (choices.empty()) throw std::invalid_argument("ParamSpec::choice: no choices");
  ParamSpec s;
  s.name = name;
  s.kind = ParamKind::Choice;
  s.choices = choices;
  return s;
}

// ============================================================================
// HyperparamSpace
// ============================================================================

HyperparamSpace::HyperparamSpace(std::initializer_list<ParamSpec> specs) {
  for (const auto& s : specs) add(s);
}

HyperparamSpace& HyperparamSpace::add(const ParamSpec& spec) {
  for (auto& s : specs_)
    if (s.name == spec.name) {
      s = spec;
      return *this;
    }
  specs_.push_back(spec);
  return *this;
}

bool HyperparamSpace::remove(const std::string& name) {
  auto it = std::remove_if(specs_.begin(), specs_.end(), [&](const ParamSpec& s) { return s.name == name; });
  bool found = it != specs_.end();
  specs_.erase(it, specs_.end());
  return found;
}

const ParamSpec* HyperparamSpace::find(const std::string& name) const {
  for (const auto& s : specs_)
    if (s.name == name) return &s;
  return nullptr;
}
ParamSpec* HyperparamSpace::find(const std::string& name) {
  for (auto& s : specs_)
    if (s.name == name) return &s;
  return nullptr;
}

static void sample_param(HyperparamValues& v, const ParamSpec& s, Rng& rng) {
  switch (s.kind) {
    case ParamKind::Float:
      if (s.log_scale)
        v.set_float(s.name, std::exp(rng.uniform(std::log(s.lo), std::log(s.hi))));
      else
        v.set_float(s.name, rng.uniform(s.lo, s.hi));
      break;
    case ParamKind::Int:
      v.set_int(s.name, rng.randint(static_cast<int>(s.lo), static_cast<int>(s.hi)));
      break;
    case ParamKind::Bool:
      v.set_bool(s.name, rng.bernoulli(0.5));
      break;
    case ParamKind::Choice:
      v.set_choice(s.name, rng.choice(s.choices));
      break;
  }
}

HyperparamValues HyperparamSpace::sample(Rng& rng) const {
  HyperparamValues v;
  for (const auto& s : specs_) sample_param(v, s, rng);
  return v;
}

void HyperparamSpace::complete(HyperparamValues& v, Rng& rng) const {
  for (const auto& s : specs_)
    if (!v.has(s.name)) sample_param(v, s, rng);
}

void HyperparamSpace::mutate_param(HyperparamValues& v, const ParamSpec& s, Rng& rng) const {
  if (!v.has(s.name)) {
    sample_param(v, s, rng);
    return;
  }
  switch (s.kind) {
    case ParamKind::Float: {
      double x = v.f(s.name);
      if (s.log_scale) {
        double lx = std::log(std::max(x, s.lo));
        lx += rng.normal(0.0, s.mutation_std * (std::log(s.hi) - std::log(s.lo)));
        lx = std::min(std::log(s.hi), std::max(std::log(s.lo), lx));
        v.set_float(s.name, std::exp(lx));
      } else {
        x += rng.normal(0.0, s.mutation_std * (s.hi - s.lo));
        v.set_float(s.name, std::min(s.hi, std::max(s.lo, x)));
      }
      break;
    }
    case ParamKind::Int: {
      int lo = static_cast<int>(s.lo), hi = static_cast<int>(s.hi);
      int x = v.i(s.name);
      double sd = std::max(1.0, s.mutation_std * (s.hi - s.lo));
      int nx = static_cast<int>(std::lround(x + rng.normal(0.0, sd)));
      nx = std::min(hi, std::max(lo, nx));
      if (nx == x && hi > lo) nx = (x == hi) ? x - 1 : (x == lo ? x + 1 : (rng.bernoulli(0.5) ? x + 1 : x - 1));
      v.set_int(s.name, nx);
      break;
    }
    case ParamKind::Bool:
      v.set_bool(s.name, !v.b(s.name));
      break;
    case ParamKind::Choice: {
      if (s.choices.size() < 2) {
        v.set_choice(s.name, s.choices[0]);
        break;
      }
      std::string cur = v.s(s.name);
      std::vector<std::string> others;
      for (const auto& c : s.choices)
        if (c != cur) others.push_back(c);
      v.set_choice(s.name, rng.choice(others));
      break;
    }
  }
}

void HyperparamSpace::mutate(HyperparamValues& v, Rng& rng, double per_param_prob) const {
  if (specs_.empty()) return;
  bool any = false;
  for (const auto& s : specs_) {
    if (rng.bernoulli(per_param_prob)) {
      mutate_param(v, s, rng);
      any = true;
    }
  }
  if (!any) {
    // Prefer params that can actually change.
    std::vector<const ParamSpec*> mutable_specs;
    for (const auto& s : specs_)
      if (!(s.kind == ParamKind::Choice && s.choices.size() < 2) &&
          !((s.kind == ParamKind::Float || s.kind == ParamKind::Int) && s.hi <= s.lo))
        mutable_specs.push_back(&s);
    if (!mutable_specs.empty()) mutate_param(v, *rng.choice(mutable_specs), rng);
  }
}

HyperparamValues HyperparamSpace::crossover(const HyperparamValues& a, const HyperparamValues& b, Rng& rng) const {
  HyperparamValues c = a;
  for (const auto& kv : b.values()) {
    if (!c.has(kv.first) || rng.bernoulli(0.5)) c.values()[kv.first] = kv.second;
  }
  return c;
}

bool HyperparamSpace::active(const ParamSpec& spec, const HyperparamValues& v) {
  if (spec.when_type.empty()) return true;
  std::string t = v.s("type");
  return std::find(spec.when_type.begin(), spec.when_type.end(), t) != spec.when_type.end();
}

double HyperparamSpace::distance(const HyperparamValues& a, const HyperparamValues& b) const {
  double total = 0.0;
  int n = 0;
  for (const auto& s : specs_) {
    if (!active(s, a) && !active(s, b)) continue;
    bool ha = a.has(s.name), hb = b.has(s.name);
    ++n;
    if (ha != hb) {
      total += 1.0;
      continue;
    }
    if (!ha) continue;
    switch (s.kind) {
      case ParamKind::Float:
      case ParamKind::Int: {
        double x = a.f(s.name), y = b.f(s.name);
        double range;
        if (s.log_scale) {
          x = std::log(std::max(x, s.lo));
          y = std::log(std::max(y, s.lo));
          range = std::log(s.hi) - std::log(s.lo);
        } else {
          range = s.hi - s.lo;
        }
        total += range > 0 ? std::min(1.0, std::fabs(x - y) / range) : 0.0;
        break;
      }
      case ParamKind::Bool:
        total += (a.b(s.name) != b.b(s.name)) ? 1.0 : 0.0;
        break;
      case ParamKind::Choice:
        total += (a.s(s.name) != b.s(s.name)) ? 1.0 : 0.0;
        break;
    }
  }
  return n ? total / n : 0.0;
}

}  // namespace cdn
