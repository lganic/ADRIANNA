# Configuration

Every evolution setting, its default, and advice on tuning it. The search space
itself (what a layer can be) is covered in [Search spaces](search-spaces.md).

## CoDeepNEATConfig

```cpp
CoDeepNEATConfig cfg;
cfg.io.inputs.push_back({"obs", {8}});
cfg.io.outputs.push_back({"policy", 4, "linear", 0.1f});
cfg.io.outputs.push_back({"value", 1});
cfg.networks_per_generation = 40;
cfg.num_threads = 8;
```

| Field | Default | Meaning |
|---|---|---|
| `io` | empty, **required** | Network inputs and output heads |
| `module_node_space` | `spaces::dense_layers()` | What a module's layer can be |
| `blueprint_node_space` | `spaces::blueprint_nodes()` | Hyperparameters of blueprint nodes |
| `global_space` | `spaces::rl_globals()` | Network-wide hyperparameters |
| `modules` | 45 genomes, 4 target species | Module population settings |
| `blueprints` | 25 genomes, 3 target species | Blueprint population settings; `mutation.node_hyperparams` is 0.1 |
| `networks_per_generation` | 100 | Networks assembled and evaluated per generation |
| `num_threads` | 1 | Evaluator threads. Results don't depend on this |
| `seed` | 42 | Seed of the whole run |
| `failed_fitness` | -1e9 | Fitness when an evaluation throws or returns NaN/infinity |
| `recurrent_layer_types` | `{"lstm"}` | Layer types eligible for cell-skip mutations |
| `build_options` | `default_build_options` | Maps global hyperparameters to `BuildOptions` |
| `verbose` | false | Print a line per generation |

The defaults for the population sizes and `networks_per_generation` come from the
paper's CIFAR-10 run. They suit long runs. For a first experiment, try something like
16 modules, 8 blueprints and 16 networks, as `examples/catch_conv.cpp` does.

### The interface: IOSpec

`InputSpec{name, shape}` describes one input. The shape is per sample, either `{F}` or
`{C, H, W}`.

`OutputSpec{name, size, activation, init_scale}` describes one head, a dense layer on
top of the evolved body. `activation` defaults to `"linear"`. `init_scale` multiplies
the head's initial weights, and small values such as 0.1 start a policy close to
uniform.

### failed_fitness

Module and blueprint fitness are means, so one sample of -1e9 ruins the average of
everything that shares it, and it also stretches the normalisation of offspring
allocation. Failures shouldn't normally happen. If yours do, for example because your
evaluator rejects some networks on purpose, set this a little below your worst
realistic score.

## DeepNEATConfig

| Field | Default | Meaning |
|---|---|---|
| `io` | empty, **required** | As above |
| `node_space` | `spaces::dense_layers()` | What a layer can be |
| `global_space` | `spaces::rl_globals()` | Network-wide hyperparameters |
| `population` | 50 genomes, 4 target species | Population settings |
| `evaluations_per_genome` | 1 | Fitness is the mean of this many evaluations |
| `num_threads`, `seed`, `failed_fitness`, `recurrent_layer_types`, `build_options`, `verbose` | as above | |

## PopulationParams

`cfg.modules`, `cfg.blueprints` and `dcfg.population` each have this type. It holds
`size` (default 50, overridden as shown above) and three groups of settings.

### MutationParams

Each rate is the chance that the operator runs once on a child, except where marked
"per node". Elites are not mutated.

| Field | Default | Applies to | Meaning |
|---|---|---|---|
| `add_node` | 0.10 | all | Split a connection with a new node |
| `add_connection` | 0.10 | all | Add or re-enable a connection |
| `toggle_connection` | 0.05 | all | Enable or disable a connection |
| `node_hyperparams` | 0.25 | all, per node | Mutate that node's table |
| `hyperparam_rate` | 0.30 | all | Per-parameter rate once a table is mutated; at least one changes |
| `add_cell_skip` | 0.05 | modules, DeepNEAT | Add or re-enable a memory-cell skip |
| `toggle_cell_skip` | 0.03 | modules, DeepNEAT | Flip a memory-cell skip |
| `change_species` | 0.10 | blueprints, per node | Re-point to another module species |
| `globals` | 0.30 | blueprints, DeepNEAT | Mutate the global hyperparameters |
| `out_merge` | 0.02 | all | Flip the output merge between concat and sum |
| `max_nodes` | 24 | all | Hidden-node cap per genome |
| `initial_structural_mutations` | 0 | all | Random structural mutations on each initial genome |

An assembled network can have up to `blueprints.max_nodes * modules.max_nodes` layers,
so with the defaults a very large network is possible. Lower both caps if networks
grow too big for your training budget.

### SpeciationParams

The distance between two genomes is:

```text
d = (c_excess * E + c_disjoint * D) / N
  + c_hyperparams * mean hyperparameter distance over shared nodes
  + c_species     * fraction of shared nodes with different species pointers  (blueprints)
  + c_globals     * global hyperparameter distance                            (not modules)
  + 0.05 if the output merges differ
```

`E` and `D` count excess and disjoint genes, and `N` is the larger gene count.
Hyperparameter distances are in [0, 1].

| Field | Default | Meaning |
|---|---|---|
| `c_excess` | 1.0 | Weight of excess genes |
| `c_disjoint` | 1.0 | Weight of disjoint genes |
| `c_hyperparams` | 0.5 | Weight of node hyperparameter differences |
| `c_species` | 0.5 | Weight of species-pointer differences |
| `c_globals` | 0.3 | Weight of global hyperparameter differences |
| `target_species` | 4 | Species count the threshold steers toward (see below) |
| `initial_threshold` | 0.4 | Starting compatibility threshold |
| `threshold_step` | 0.05 | Change per speciation |
| `min_threshold` | 0.02 | Lower bound |

`target_species` is a soft target. The threshold falls while there are too few species and rises while
there are too many, but it only controls how readily **new** species form. Existing species don't merge:
each one keeps its closest genome, so a species goes away only when it is culled for stagnation. The
count can therefore overshoot the target and stay there for up to `stagnation_limit` generations. In
short runs, expect more species than the target.

### ReproductionParams

| Field | Default | Meaning |
|---|---|---|
| `survival_threshold` | 0.4 | Fraction of each species allowed to breed |
| `elitism` | 1 | Unchanged copies of each species' best |
| `crossover_prob` | 0.25 | Chance a child comes from crossover rather than cloning |
| `stagnation_limit` | 15 | Generations without improvement before a species is culled |
| `species_elitism` | 2 | Top species protected from culling |
| `min_species_size` | 2 | Minimum offspring per surviving species |

## BuildOptions

Set per network by `cfg.build_options`. The default reads `weight_init` from the
global hyperparameters.

| Field | Default | Meaning |
|---|---|---|
| `weight_init` | `"glorot_uniform"` | `glorot_uniform`, `glorot_normal`, `he_normal`, `he_uniform`, `lecun_normal` or `uniform` |
| `uniform_range` | 0.05 | Range for `"uniform"`, the paper's LSTM setting |
| `lstm_forget_bias` | 1.0 | Initial forget-gate bias |

To change them for all networks, supply your own function:

```cpp
cfg.build_options = [](const HyperparamValues& globals) {
  BuildOptions opt = default_build_options(globals);
  opt.lstm_forget_bias = 2.0f;
  return opt;
};
```

## Tuning advice

These are starting points, not measured results. Fitness noise makes small differences
hard to see, so change one thing at a time.

**How much compute do I need?** Cost is about `generations x networks_per_generation x
training budget`. Set the training budget first: it should be enough for a good network
to show clear progress, and no more. The paper trained for 8 epochs during evolution and
300 afterwards.

**Keep `networks_per_generation` at or above the blueprint count.** Otherwise some
blueprints are never evaluated in a generation. Two to four times the blueprint count
lets each one be tried with different modules, which makes its average more reliable.

**Every module should get used.** A species only gets evaluated if a blueprint points to
it. If `species m=` is high but few species are referenced, raise `change_species` or
lower `modules.speciation.target_species`.

**Too few species.** Evolution converges early. Raise `target_species`, or lower
`initial_threshold` so that separation starts sooner.

**Too many tiny species.** Selection has little to work with. Raise `initial_threshold` or
`threshold_step` so that fewer species form, and lower `stagnation_limit` so that weak ones are removed
sooner. Aim for at least four or five genomes per species.

**Networks grow without improving.** Lower `add_node` and `add_connection`, or lower
`max_nodes`. You can also penalise size in the fitness:

```cpp
Evaluator evaluate = [](EvalContext& ctx) {
  double score = 1.0;  // your task score goes here
  return score - 1e-5 * static_cast<double>(ctx.net.num_parameters());
};
```

**Noisy fitness.** Evaluate on more episodes, and raise `networks_per_generation` so each
genome's average covers more samples. For DeepNEAT use `evaluations_per_genome`.

**Promising species die out.** Raise `stagnation_limit`. Fitness in RL fluctuates, so a
species can look stagnant while it is doing fine.

**Reproducibility.** A run is determined by `seed` and your evaluator, as long as the
evaluator draws randomness only from `ctx.rng`. Results are the same for any `num_threads`.
They can differ between compilers, because `std::` random distributions are
implementation-defined.
