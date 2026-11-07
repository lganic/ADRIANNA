#pragma once
// NEAT-style speciated population: species persist across generations (their
// ids are what CoDeepNEAT blueprint nodes point to), offspring are allocated in
// proportion to species fitness, stagnant species are culled and the
// compatibility threshold adapts to keep roughly `target_species` species.

#include <limits>
#include <vector>

#include "codeepneat/genome.hpp"

namespace cdn {

struct ReproductionParams {
  double survival_threshold = 0.4;  // fraction of each species allowed to reproduce
  int elitism = 1;                  // unchanged copies of each species' best
  double crossover_prob = 0.25;     // otherwise clone + mutate
  int stagnation_limit = 15;        // generations without improvement before culling
  int species_elitism = 2;          // best species protected from stagnation culling
  int min_species_size = 2;
};

/// All evolution settings of one population.
struct PopulationParams {
  int size = 50;  ///< constant number of genomes
  MutationParams mutation;
  SpeciationParams speciation;
  ReproductionParams reproduction;
};

/// A group of similar genomes. Ids are never reused, and a species keeps its id
/// across generations for as long as it has members.
struct Species {
  int id = -1;
  Genome representative;  ///< genome that newcomers are compared against
  std::vector<int> members;  // indices into Population::genomes()
  double best_fitness = -std::numeric_limits<double>::infinity();  ///< best member ever
  int last_improved = 0;     ///< generation best_fitness last rose
  int created = 0;           ///< generation of first appearance
  double mean_fitness = 0.0; ///< as of the last reproduce()
  double max_fitness = 0.0;  ///< as of the last reproduce()
};

/// Fixed-size population of one GenomeKind. CoDeepNEAT owns two of these and
/// DeepNEAT one; the class can also be driven directly for custom algorithms.
class Population {
 public:
  Population() = default;
  Population(PopulationParams params, GenomeOperators ops, int num_inputs);

  /// Create `size` minimal genomes and speciate them. Blueprint populations need
  /// the module species ids in `species_choices`.
  void initialize(Rng& rng, const std::vector<int>& species_choices = {});
  /// Assign every genome to a species (re-using persistent species). Each existing
  /// species first claims its closest genome as new representative; the others
  /// join the closest species within the threshold or found a new one. Afterwards
  /// the threshold moves one step toward target_species.
  void speciate(int generation);
  /// Replace the population by the next generation (requires fitness values):
  /// culls stagnant species, allocates offspring by normalised mean species
  /// fitness, copies elites, breeds the rest from the top survivors, and speciates.
  void reproduce(int generation, Rng& rng, const std::vector<int>& species_choices = {});

  std::vector<Genome>& genomes() { return genomes_; }
  const std::vector<Genome>& genomes() const { return genomes_; }
  std::vector<Species>& species() { return species_; }
  const std::vector<Species>& species() const { return species_; }
  std::vector<int> species_ids() const;
  const Species* find_species(int id) const;
  Genome* find_genome(int id);
  double compatibility_threshold() const { return threshold_; }
  InnovationTracker& innovations() { return innov_; }
  const PopulationParams& params() const { return params_; }
  const GenomeOperators& ops() const { return ops_; }
  int num_inputs() const { return num_inputs_; }
  int next_genome_id() { return next_genome_id_++; }

 private:
  PopulationParams params_;
  GenomeOperators ops_;
  int num_inputs_ = 1;
  InnovationTracker innov_;
  std::vector<Genome> genomes_;
  std::vector<Species> species_;
  double threshold_ = 0.4;
  int next_genome_id_ = 0;
  int next_species_id_ = 0;
};

}  // namespace cdn
