#pragma once
// CoDeepNEAT (Sec. 3.2) and DeepNEAT (Sec. 3.1).
//
// CoDeepNEAT co-evolves two populations:
//   * modules    - small DNN graphs whose nodes are layers,
//   * blueprints - graphs whose nodes point to module *species*, plus global
//                  hyperparameters.
// Each generation `networks_per_generation` networks are assembled: every
// blueprint node is replaced by a module drawn at random from the species it
// points to (one module per species per network, so repeated pointers repeat
// the same module). Each assembled network is built, handed to the user's
// evaluator (which trains it - e.g. with RL - and returns a fitness), and the
// fitness is attributed back to the blueprint and the modules as the average
// over all networks they took part in. Both populations then reproduce with
// NEAT (speciation, crossover via historical markings, structural and
// hyperparameter mutation).

#include <functional>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "codeepneat/graph.hpp"
#include "codeepneat/network.hpp"
#include "codeepneat/population.hpp"
#include "codeepneat/spaces.hpp"

namespace cdn {

/// One network input.
struct InputSpec {
  std::string name;
  Shape shape;  // per sample: [F] or [C,H,W]
};

/// One output head: a dense layer on top of the evolved body.
struct OutputSpec {
  std::string name;
  int size = 1;
  std::string activation = "linear";
  float init_scale = 1.0f;  // multiplier on the head's initial weights
};

/// The fixed interface of every evolved network: named inputs and output heads
/// (e.g. "policy" logits and a "value" estimate for actor-critic RL).
struct IOSpec {
  std::vector<InputSpec> inputs;
  std::vector<OutputSpec> outputs;
};

/// A concrete network genotype: a blueprint plus the module chosen for every
/// species it references (CoDeepNEAT), or a single DeepNEAT genome stored in
/// `blueprint` with no modules.
struct AssembledGenome {
  Genome blueprint;
  std::map<int, Genome> modules;  // species id -> module
  double fitness = 0.0;
  int generation = -1;

  const HyperparamValues& globals() const { return blueprint.globals; }
  /// Text serialisation of the genotype (no weights); see docs/file-formats.md.
  void write(std::ostream& os) const;
  static AssembledGenome read(std::istream& is);
  void save(const std::string& path) const;
  static AssembledGenome load(const std::string& path);
  /// Human-readable listing of the blueprint and its modules.
  std::string describe() const;
};

/// Expand an assembled genome into a NetworkGraph. Node names:
///   input:<name>, bp<N>/m<M>/in, bp<N>/m<M>/n<K>, bp<N>/m<M>/out, body, head:<name>
/// (DeepNEAT: n<K>). Module instances repeated in several blueprint nodes get
/// distinct names (and distinct weights).
NetworkGraph assemble_network(const AssembledGenome& genome, const IOSpec& io,
                              const std::vector<std::string>& recurrent_types = {"lstm"});

/// BuildOptions from global hyperparameters (reads "weight_init" if present).
BuildOptions default_build_options(const HyperparamValues& globals);

/// Everything an evaluator gets. All references are valid for the duration of the
/// call only; the network is destroyed afterwards.
struct EvalContext {
  Network& net;                    // freshly built, randomly initialised
  const HyperparamValues& globals; // global hyperparameters of this network
  Rng& rng;                        // dedicated RNG for this evaluation
  const AssembledGenome& genome;
  int generation = 0;
  int index = 0;                   // index of the network within the generation
};

/// Train/evaluate one network and return its fitness (higher is better).
/// Called concurrently from several threads when num_threads > 1, so anything it
/// shares must be thread-safe. If it throws or returns a non-finite value the
/// network gets `failed_fitness` and the record's `error` is set.
using Evaluator = std::function<double(EvalContext& ctx)>;

/// Outcome of one network evaluation.
struct EvaluationRecord {
  int index = 0;
  int blueprint_id = -1;  ///< DeepNEAT: the genome id
  std::map<int, int> modules;  // species -> module id
  double fitness = 0.0;
  size_t num_parameters = 0;
  std::string error;           // non-empty if the evaluator threw
};

/// Summary of one generation. Fitness fields describe individual networks of this
/// generation; `best_ever` covers the whole run.
struct GenerationStats {
  int generation = 0;
  double best_fitness = 0.0;
  double mean_fitness = 0.0;
  double worst_fitness = 0.0;
  double best_ever = 0.0;
  int module_species = 0;     ///< DeepNEAT: the population's species count
  int blueprint_species = 0;  ///< DeepNEAT: always 0
  int evaluations = 0;
  int failures = 0;           ///< evaluations with a non-empty error
  double mean_parameters = 0.0;
  double seconds = 0.0;       ///< wall-clock, including reproduction
  std::vector<EvaluationRecord> records;
  /// One-line summary.
  std::string to_string() const;
};

struct CoDeepNEATConfig {
  IOSpec io;  ///< required: at least one input and one output
  HyperparamSpace module_node_space = spaces::dense_layers();  ///< what a layer can be
  HyperparamSpace blueprint_node_space = spaces::blueprint_nodes();
  HyperparamSpace global_space = spaces::rl_globals();
  PopulationParams modules;     // defaults: 45 modules (paper's CIFAR-10 setup)
  PopulationParams blueprints;  // defaults: 25 blueprints
  int networks_per_generation = 100;  ///< networks assembled and evaluated per generation
  int num_threads = 1;                ///< evaluator threads; does not affect results
  uint64_t seed = 42;
  /// Fitness for networks whose evaluation failed. Module and blueprint fitness is
  /// a mean, so a huge negative value swamps the other samples it is averaged with.
  /// Choose something a little below your worst realistic score if failures are
  /// expected.
  double failed_fitness = -1e9;
  /// Layer types that take part in memory-cell skip mutations.
  std::vector<std::string> recurrent_layer_types = {"lstm"};
  /// Maps a network's global hyperparameters to its build options.
  std::function<BuildOptions(const HyperparamValues& globals)> build_options = default_build_options;
  /// Print a line per generation to stdout.
  bool verbose = false;

  /// Sets the population defaults: 45 modules with 4 target species, 25 blueprints
  /// with 3 target species and node_hyperparams = 0.1.
  CoDeepNEATConfig();
};

/// The CoDeepNEAT algorithm. The constructor creates both initial populations.
class CoDeepNEAT {
 public:
  explicit CoDeepNEAT(CoDeepNEATConfig config);

  /// Evaluate one generation and breed the next one.
  GenerationStats step(const Evaluator& evaluator);
  /// Run `generations` steps; `on_generation` may return false to stop early.
  std::vector<GenerationStats> run(int generations, const Evaluator& evaluator,
                                   const std::function<bool(const GenerationStats&)>& on_generation = nullptr);

  /// Best single assembled network seen so far (by its evaluated fitness).
  bool has_best() const { return has_best_; }
  const AssembledGenome& best() const;
  /// Assemble a network from the current populations (blueprint by index,
  /// modules sampled from the referenced species).
  AssembledGenome sample_assembly(size_t blueprint_index, Rng& rng) const;

  /// Expands a genotype with this run's IOSpec.
  NetworkGraph graph(const AssembledGenome& g) const;
  /// Builds a network with fresh random weights. Evolution stores architectures,
  /// not trained weights, so the result must be trained again.
  std::unique_ptr<Network> build(const AssembledGenome& g, uint64_t seed = 1) const;

  /// Number of completed generations.
  int generation() const { return generation_; }
  Population& modules() { return modules_; }
  const Population& modules() const { return modules_; }
  Population& blueprints() { return blueprints_; }
  const Population& blueprints() const { return blueprints_; }
  const CoDeepNEATConfig& config() const { return cfg_; }
  Rng& rng() { return rng_; }

 private:
  void fix_species_pointers();

  CoDeepNEATConfig cfg_;
  Rng rng_;
  Population modules_;
  Population blueprints_;
  int generation_ = 0;
  bool has_best_ = false;
  AssembledGenome best_;
};

/// Fields mean the same as in CoDeepNEATConfig. `population` defaults to 50 genomes.
struct DeepNEATConfig {
  IOSpec io;
  HyperparamSpace node_space = spaces::dense_layers();
  HyperparamSpace global_space = spaces::rl_globals();
  PopulationParams population;
  int evaluations_per_genome = 1;  // average fitness over repeated evaluations
  int num_threads = 1;
  uint64_t seed = 42;
  double failed_fitness = -1e9;
  std::vector<std::string> recurrent_layer_types = {"lstm"};
  std::function<BuildOptions(const HyperparamValues& globals)> build_options = default_build_options;
  bool verbose = false;
};

/// DeepNEAT: each genome is a complete network (Sec. 3.1). Each generation
/// evaluates every genome `evaluations_per_genome` times. best() returns an
/// AssembledGenome holding the winner in `blueprint`, with no modules.
class DeepNEAT {
 public:
  explicit DeepNEAT(DeepNEATConfig config);
  GenerationStats step(const Evaluator& evaluator);
  std::vector<GenerationStats> run(int generations, const Evaluator& evaluator,
                                   const std::function<bool(const GenerationStats&)>& on_generation = nullptr);
  bool has_best() const { return has_best_; }
  const AssembledGenome& best() const;
  NetworkGraph graph(const AssembledGenome& g) const;
  std::unique_ptr<Network> build(const AssembledGenome& g, uint64_t seed = 1) const;
  int generation() const { return generation_; }
  Population& population() { return pop_; }
  const DeepNEATConfig& config() const { return cfg_; }

 private:
  DeepNEATConfig cfg_;
  Rng rng_;
  Population pop_;
  int generation_ = 0;
  bool has_best_ = false;
  AssembledGenome best_;
};

}  // namespace cdn
