#pragma once
// Graph genomes shared by DeepNEAT and CoDeepNEAT.
//
// One Genome class serves three roles (GenomeKind):
//   Module    : a small DNN; nodes are layers (hyperparameter tables), with one
//               virtual input node (id 0) and one virtual output node (id 1).
//   Blueprint : a graph whose nodes point to module *species*; one virtual input
//               node per network input (ids 0..n-1) and an output node (id n).
//               Carries the global hyperparameters.
//   DeepNEAT  : a whole network as one graph of layer nodes, plus globals.
//
// Edges carry no weights - they only say which layers are connected (Sec. 3.1).
// Connection genes and LSTM memory-cell skip genes are aligned through
// historical markings (innovation numbers) for crossover and speciation.

#include <iosfwd>
#include <map>
#include <string>
#include <utility>
#include <vector>

#include "codeepneat/hyperparams.hpp"
#include "codeepneat/rng.hpp"

namespace cdn {

/// Which role a Genome plays; see the comment at the top of this file.
enum class GenomeKind { Module, Blueprint, DeepNEAT };
const char* to_string(GenomeKind k);

/// A hidden node: a layer's hyperparameter table (Module, DeepNEAT) or a module
/// species pointer plus merge settings (Blueprint).
struct NodeGene {
  int id = -1;
  HyperparamValues hp;
  int species = -1;  // blueprint nodes: module species pointer
};

/// A directed edge between two nodes. Disabled genes stay in the genome: they are
/// ignored when building a network but still count for crossover and distance,
/// and they can be re-enabled by mutation. Memory-cell skip genes reuse this type.
struct ConnGene {
  int innovation = -1;
  int from = -1;
  int to = -1;
  bool enabled = true;
};

/// A graph genome. The class is plain data plus queries; all change happens
/// through GenomeOperators.
///
/// Node ids: 0..num_inputs-1 are the virtual inputs, num_inputs is the virtual
/// output, and larger ids are hidden nodes (the only ones stored in `nodes`).
///
/// Invariants, checked by validate() and kept by every operator:
///  - the connection genes form a DAG, counting disabled genes too;
///  - at most one gene exists per (from, to) pair;
///  - cell skips connect two distinct hidden nodes, in either direction.
/// Operators also keep an enabled input-to-output path.
class Genome {
 public:
  Genome() = default;
  Genome(GenomeKind kind, int num_inputs) : kind(kind), num_inputs(num_inputs) {}

  int id = -1;
  GenomeKind kind = GenomeKind::Module;
  int num_inputs = 1;
  std::map<int, NodeGene> nodes;       // hidden nodes only (inputs/output are virtual)
  std::map<int, ConnGene> conns;       // keyed by innovation
  std::map<int, ConnGene> cell_skips;  // LSTM memory-cell skip genes, keyed by innovation
  std::string out_merge = "concat";    // merge applied at the virtual output node
  HyperparamValues globals;            // Blueprint / DeepNEAT

  // Evaluation bookkeeping.
  double fitness = 0.0;                  ///< mean of fitness_samples, or a fallback (see assign_fitness)
  bool evaluated = false;                ///< took part in a network this generation
  bool has_prior_fitness = false;        ///< an elite, or loaded from a file: keeps `fitness` if unused
  std::vector<double> fitness_samples;   ///< one value per network this genome took part in
  int species = -1;                      ///< id of the species this genome belongs to
  int birth_generation = 0;

  int output_id() const { return num_inputs; }
  bool is_input(int node) const { return node >= 0 && node < num_inputs; }
  bool is_output(int node) const { return node == num_inputs; }
  bool is_hidden(int node) const { return nodes.count(node) != 0; }
  bool node_exists(int node) const { return is_input(node) || is_output(node) || is_hidden(node); }

  const ConnGene* find_conn(int from, int to) const;
  ConnGene* find_conn(int from, int to);
  const ConnGene* find_cell_skip(int from, int to) const;
  ConnGene* find_cell_skip(int from, int to);

  /// True if adding from->to would close a cycle (considering *all* connection
  /// genes, enabled or not, so that re-enabling can never create a cycle).
  bool would_create_cycle(int from, int to) const;
  /// Path from some input to the output over enabled connections.
  bool has_input_output_path() const;
  /// Hidden nodes lying on an enabled input->output path, in topological order.
  std::vector<int> active_nodes() const;
  /// Enabled predecessors of `node` (sorted by innovation number).
  std::vector<int> enabled_predecessors(int node) const;
  size_t gene_count() const { return conns.size() + cell_skips.size(); }
  size_t enabled_conn_count() const;

  /// Structural invariants (acyclic, endpoints exist, unique (from,to) pairs...).
  bool validate(std::string* why = nullptr) const;
  /// Clears `evaluated` and `fitness_samples` but keeps `fitness`.
  void reset_evaluation();
  /// Multi-line, human-readable listing of all genes.
  std::string describe() const;

  /// Text serialisation; see docs/file-formats.md.
  void write(std::ostream& os) const;
  /// Parses and validates; throws std::runtime_error on malformed input.
  static Genome read(std::istream& is);
};

/// Global historical markings for one population. Identical structural
/// innovations (same from->to, same split connection) receive identical numbers.
class InnovationTracker {
 public:
  InnovationTracker() = default;
  /// Reserves ids for the virtual nodes of genomes with `num_inputs` inputs.
  explicit InnovationTracker(int num_inputs);

  /// Innovation number of the connection from -> to. Unlike classic NEAT the table
  /// lives for the whole run rather than one generation, so a structure that
  /// reappears later aligns with its earlier occurrences.
  int connection(int from, int to);
  /// Same for memory-cell skips (separate table, shared counter).
  int cell_skip(int from, int to);
  /// Node id for splitting the connection with `conn_innovation`; reuses an id
  /// from an earlier identical split unless `g` already contains it.
  int split_node(int conn_innovation, const Genome& g);
  /// Id of the single hidden node shared by all initial genomes.
  int initial_node() const { return initial_node_; }
  int next_innovation() const { return next_innov_; }
  int next_node() const { return next_node_; }

 private:
  int next_innov_ = 0;
  int initial_node_ = 2;
  int next_node_ = 3;
  std::map<std::pair<int, int>, int> conn_, skip_;
  std::map<int, std::vector<int>> split_;
};

/// Mutation rates. Unless noted, each is the chance that the operator is applied
/// once to a child. docs/configuration.md discusses tuning.
struct MutationParams {
  double add_node = 0.10;          // per genome
  double add_connection = 0.10;    // per genome
  double toggle_connection = 0.05; // per genome: enable/disable a connection
  double node_hyperparams = 0.25;  // per node: mutate its hyperparameter table
  double hyperparam_rate = 0.30;   // per parameter, once a table is mutated
  double add_cell_skip = 0.05;     // per genome: add/re-enable an LSTM cell skip
  double toggle_cell_skip = 0.03;  // per genome: disable/enable a cell skip
  double change_species = 0.10;    // per blueprint node: re-point to another module species
  double globals = 0.30;           // per genome: mutate the global hyperparameters
  double out_merge = 0.02;         // per genome: flip output merge (concat/sum)
  int max_nodes = 24;              // cap on hidden nodes per genome
  /// Random add-node / add-connection mutations applied to each initial genome.
  int initial_structural_mutations = 0;
};

/// Compatibility distance weights and the threshold controller.
///
///   d = (c_excess * E + c_disjoint * D) / N
///     + c_hyperparams * mean node hyperparameter distance
///     + c_species * fraction of different species pointers   (blueprints)
///     + c_globals * global hyperparameter distance           (not modules)
///     + 0.05 if out_merge differs
///
/// E and D count excess and disjoint genes (connections and cell skips); N is the
/// larger gene count, at least 1. Node terms use nodes present in both genomes.
struct SpeciationParams {
  double c_excess = 1.0;
  double c_disjoint = 1.0;
  double c_hyperparams = 0.5;  // mean node hyperparameter distance (matching nodes)
  double c_species = 0.5;      // fraction of matching blueprint nodes with different species pointers
  double c_globals = 0.3;      // global hyperparameter distance
  /// Soft target. The threshold moves toward it but only affects how new species
  /// form; existing species last until culled for stagnation.
  int target_species = 4;
  double initial_threshold = 0.4;  ///< genomes closer than this share a species
  double threshold_step = 0.05;    ///< adjustment per speciation
  double min_threshold = 0.02;     ///< lower bound of the threshold
};

/// Mutation, crossover and compatibility distance for one GenomeKind.
class GenomeOperators {
 public:
  GenomeOperators() = default;
  GenomeOperators(GenomeKind kind, HyperparamSpace node_space, HyperparamSpace global_space, MutationParams mutation,
                  SpeciationParams speciation, std::vector<std::string> recurrent_types = {"lstm"});

  /// Which operators apply is decided by kind(): cell skips for Module and
  /// DeepNEAT, species pointers for Blueprint, globals for Blueprint and DeepNEAT.
  GenomeKind kind() const { return kind_; }
  const HyperparamSpace& node_space() const { return node_space_; }
  const HyperparamSpace& global_space() const { return global_space_; }
  const MutationParams& mutation() const { return mutation_; }
  const SpeciationParams& speciation() const { return speciation_; }

  /// Minimal genome: every input -> one hidden node -> output.
  Genome create_initial(int num_inputs, InnovationTracker& innov, Rng& rng,
                        const std::vector<int>& species_choices = {}) const;
  /// Applies each mutation with its rate from MutationParams. For blueprints,
  /// `species_choices` lists the live module species; pointers to anything else
  /// are re-targeted at random.
  void mutate(Genome& g, InnovationTracker& innov, Rng& rng, const std::vector<int>& species_choices = {}) const;
  /// NEAT crossover: matching genes from either parent, disjoint/excess genes
  /// from the fitter parent (so the child's topology stays acyclic).
  Genome crossover(const Genome& fitter, const Genome& other, Rng& rng) const;
  /// Compatibility distance; see SpeciationParams for the formula.
  double distance(const Genome& a, const Genome& b) const;

  // Individual mutation operators (return true if the genome changed).

  /// Splits a random enabled connection a->b into a->new->b and disables the
  /// original. The new node's hyperparameters are sampled from the node space.
  /// Does nothing at max_nodes.
  bool mutate_add_node(Genome& g, InnovationTracker& innov, Rng& rng, const std::vector<int>& species_choices) const;
  /// Adds a connection that keeps the graph acyclic, or re-enables a disabled one.
  /// May connect an input straight to the output, which creates a skip path.
  bool mutate_add_connection(Genome& g, InnovationTracker& innov, Rng& rng) const;
  /// Flips a random connection, refusing to cut the last input-to-output path.
  bool mutate_toggle_connection(Genome& g, Rng& rng) const;
  /// Adds or re-enables a memory-cell skip between two recurrent nodes.
  bool mutate_add_cell_skip(Genome& g, InnovationTracker& innov, Rng& rng) const;
  /// Flips a random memory-cell skip gene.
  bool mutate_toggle_cell_skip(Genome& g, Rng& rng) const;
  /// Visits every node; with rate node_hyperparams, mutates its table.
  bool mutate_node_hyperparams(Genome& g, Rng& rng) const;
  /// Visits every blueprint node; with rate change_species, re-points it.
  bool mutate_species_pointers(Genome& g, Rng& rng, const std::vector<int>& species_choices) const;
  /// Mutates the global hyperparameter table.
  bool mutate_globals(Genome& g, Rng& rng) const;

  /// True if the node's "type" is one of the recurrent layer types.
  bool is_recurrent_node(const NodeGene& n) const;
  /// Re-enable random connections until an input->output path exists.
  void repair(Genome& g, Rng& rng) const;

 private:
  NodeGene make_node(int id, Rng& rng, const std::vector<int>& species_choices) const;

  GenomeKind kind_ = GenomeKind::Module;
  HyperparamSpace node_space_;
  HyperparamSpace global_space_;
  MutationParams mutation_;
  SpeciationParams speciation_;
  std::vector<std::string> recurrent_types_ = {"lstm"};
};

}  // namespace cdn
