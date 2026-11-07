// Speciation and reproduction, shared by module, blueprint and DeepNEAT populations.

#include "codeepneat/population.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>

namespace cdn {

Population::Population(PopulationParams params, GenomeOperators ops, int num_inputs)
    : params_(params),
      ops_(std::move(ops)),
      num_inputs_(num_inputs),
      innov_(num_inputs),
      threshold_(params.speciation.initial_threshold) {
  if (params_.size < 1) throw std::invalid_argument("Population: size must be >= 1");
}

void Population::initialize(Rng& rng, const std::vector<int>& species_choices) {
  genomes_.clear();
  species_.clear();
  for (int i = 0; i < params_.size; ++i) {
    Genome g = ops_.create_initial(num_inputs_, innov_, rng, species_choices);
    g.id = next_genome_id_++;
    g.birth_generation = 0;
    genomes_.push_back(std::move(g));
  }
  speciate(0);
}

std::vector<int> Population::species_ids() const {
  std::vector<int> r;
  for (const auto& s : species_) r.push_back(s.id);
  return r;
}

const Species* Population::find_species(int id) const {
  for (const auto& s : species_)
    if (s.id == id) return &s;
  return nullptr;
}

Genome* Population::find_genome(int id) {
  for (auto& g : genomes_)
    if (g.id == id) return &g;
  return nullptr;
}

void Population::speciate(int generation) {
  std::vector<char> assigned(genomes_.size(), 0);
  std::vector<Species> next;
  // Existing species pick the closest new genome as their representative.
  for (auto& s : species_) {
    int best = -1;
    double bd = std::numeric_limits<double>::infinity();
    for (size_t i = 0; i < genomes_.size(); ++i) {
      if (assigned[i]) continue;
      double d = ops_.distance(s.representative, genomes_[i]);
      if (d < bd) {
        bd = d;
        best = static_cast<int>(i);
      }
    }
    if (best < 0) continue;
    assigned[best] = 1;
    Species ns = s;
    ns.representative = genomes_[best];
    ns.members = {best};
    next.push_back(std::move(ns));
  }
  for (size_t i = 0; i < genomes_.size(); ++i) {
    if (assigned[i]) continue;
    int best = -1;
    double bd = std::numeric_limits<double>::infinity();
    for (size_t k = 0; k < next.size(); ++k) {
      double d = ops_.distance(next[k].representative, genomes_[i]);
      if (d < bd) {
        bd = d;
        best = static_cast<int>(k);
      }
    }
    if (best >= 0 && bd < threshold_) {
      next[best].members.push_back(static_cast<int>(i));
    } else {
      Species ns;
      ns.id = next_species_id_++;
      ns.representative = genomes_[i];
      ns.members = {static_cast<int>(i)};
      ns.created = generation;
      ns.last_improved = generation;
      next.push_back(std::move(ns));
    }
  }
  species_ = std::move(next);
  for (const auto& s : species_)
    for (int m : s.members) genomes_[m].species = s.id;

  const auto& sp = params_.speciation;
  int n = static_cast<int>(species_.size());
  if (n < sp.target_species)
    threshold_ = std::max(sp.min_threshold, threshold_ - sp.threshold_step);
  else if (n > sp.target_species)
    threshold_ += sp.threshold_step;
}

void Population::reproduce(int generation, Rng& rng, const std::vector<int>& species_choices) {
  const auto& rp = params_.reproduction;
  struct Info {
    size_t idx;
    double mean;
  };
  std::vector<Info> infos;
  for (size_t k = 0; k < species_.size(); ++k) {
    auto& s = species_[k];
    if (s.members.empty()) continue;
    double sum = 0.0, mx = -std::numeric_limits<double>::infinity();
    for (int m : s.members) {
      sum += genomes_[m].fitness;
      mx = std::max(mx, genomes_[m].fitness);
    }
    s.mean_fitness = sum / s.members.size();
    s.max_fitness = mx;
    if (mx > s.best_fitness) {
      s.best_fitness = mx;
      s.last_improved = generation;
    }
    infos.push_back({k, s.mean_fitness});
  }
  if (infos.empty()) throw std::logic_error("Population::reproduce: no species");
  std::sort(infos.begin(), infos.end(), [](const Info& a, const Info& b) { return a.mean > b.mean; });

  // Stagnation culling (the best `species_elitism` species are protected).
  std::vector<Info> alive;
  for (size_t r = 0; r < infos.size(); ++r) {
    const auto& s = species_[infos[r].idx];
    bool stagnant = generation - s.last_improved > rp.stagnation_limit;
    if (!stagnant || static_cast<int>(r) < rp.species_elitism) alive.push_back(infos[r]);
  }
  if (alive.empty()) alive.push_back(infos[0]);
  int pop = params_.size;
  if (static_cast<int>(alive.size()) > pop) alive.resize(pop);

  // Offspring allocation proportional to normalised mean fitness.
  double lo = alive.back().mean, hi = alive.front().mean;
  for (const auto& a : alive) lo = std::min(lo, a.mean), hi = std::max(hi, a.mean);
  std::vector<double> adj(alive.size());
  for (size_t k = 0; k < alive.size(); ++k)
    adj[k] = (hi - lo > 1e-12 ? (alive[k].mean - lo) / (hi - lo) : 1.0) + 0.05;
  int min_size = std::max(1, std::min(rp.min_species_size, pop / static_cast<int>(alive.size())));
  std::vector<int> counts(alive.size(), min_size);
  int remaining = pop - min_size * static_cast<int>(alive.size());
  double total = std::accumulate(adj.begin(), adj.end(), 0.0);
  std::vector<std::pair<double, size_t>> rema;
  int given = 0;
  for (size_t k = 0; k < alive.size(); ++k) {
    double share = remaining * adj[k] / total;
    int whole = static_cast<int>(std::floor(share));
    counts[k] += whole;
    given += whole;
    rema.push_back({share - whole, k});
  }
  std::sort(rema.begin(), rema.end(), [](const auto& a, const auto& b) { return a.first > b.first; });
  for (int k = 0; k < remaining - given; ++k) counts[rema[k % rema.size()].second] += 1;

  std::vector<Genome> next;
  next.reserve(pop);
  std::vector<Species> survivors;
  for (size_t k = 0; k < alive.size(); ++k) {
    Species& s = species_[alive[k].idx];
    survivors.push_back(s);
    std::vector<int> members = s.members;
    std::sort(members.begin(), members.end(),
              [&](int a, int b) { return genomes_[a].fitness > genomes_[b].fitness; });
    int count = counts[k];
    int produced = 0;
    int elites = std::min({rp.elitism, count, static_cast<int>(members.size())});
    for (int e = 0; e < elites; ++e) {
      Genome g = genomes_[members[e]];
      g.reset_evaluation();
      g.has_prior_fitness = true;
      next.push_back(std::move(g));
      ++produced;
    }
    int n_parents = std::max(1, static_cast<int>(std::ceil(rp.survival_threshold * members.size())));
    std::vector<int> parents(members.begin(), members.begin() + n_parents);
    while (produced < count) {
      Genome child;
      if (parents.size() >= 2 && rng.bernoulli(rp.crossover_prob)) {
        int a = rng.choice(parents), b = rng.choice(parents);
        while (b == a) b = rng.choice(parents);
        const Genome& ga = genomes_[a];
        const Genome& gb = genomes_[b];
        bool a_fitter = ga.fitness > gb.fitness || (ga.fitness == gb.fitness && rng.bernoulli(0.5));
        child = a_fitter ? ops_.crossover(ga, gb, rng) : ops_.crossover(gb, ga, rng);
      } else {
        child = genomes_[rng.choice(parents)];
      }
      child.reset_evaluation();
      child.has_prior_fitness = false;
      child.id = next_genome_id_++;
      child.birth_generation = generation + 1;
      ops_.mutate(child, innov_, rng, species_choices);
      next.push_back(std::move(child));
      ++produced;
    }
  }
  genomes_ = std::move(next);
  species_ = std::move(survivors);
  speciate(generation + 1);
}

}  // namespace cdn
