# Architecture

How the code is organised and why. Read [Concepts](concepts.md) first if the algorithm is new to you.

## Layers of the library

Each layer depends only on those below it.

```text
 rl.hpp  export.hpp        RL tools, PyTorch export
        |
 codeepneat.hpp            CoDeepNEAT, DeepNEAT, assembly, evaluation
        |
 population.hpp            speciation, reproduction
        |
 genome.hpp                genomes, innovations, mutation, crossover
        |
        |        network.hpp   optim.hpp     executable networks, optimizers
        |            |
        |        layers.hpp                  layers, merge, registry
        |            |
        |        graph.hpp                   network description (IR)
        |            |
 hyperparams.hpp  spaces.hpp                 search spaces
        |            |
        +---- tensor.hpp ----+               autograd
                  |
               rng.hpp
```

Genomes know nothing about networks. The only bridge between them is `assemble_network`, which turns a genotype into a
`NetworkGraph`.

## From genome to trained network

```text
blueprint + modules       AssembledGenome       a complete genotype
        |
  assemble_network()      expands modules into layer nodes, adds identity nodes
        |                 at module boundaries, adds the body and heads
        v
   NetworkGraph           plain data: nodes, edges, cell skips
        |
  Network constructor     prunes, validates, sorts, infers shapes, initialises
        v
     Network              runs forward(); owns layers and merges
        |
    your evaluator        trains it and returns a fitness
```

## Source map

| File | Contents |
|---|---|
| `src/tensor.cpp` | Tensor class, `backward`, all operations, GEMM kernels |
| `src/hyperparams.cpp` | Values, specs, sampling, mutation, distance |
| `src/spaces.cpp` | Built-in search spaces |
| `src/graph.cpp` | `NetworkGraph`, topological sort, pruning, validation, DOT |
| `src/layers.cpp` | Weight init, layers, `Merge`, `LayerRegistry` |
| `src/network.cpp` | `Network`: build, forward, hooks, state, I/O |
| `src/optim.cpp` | SGD, Adam, RMSProp |
| `src/genome.cpp` | Genome queries and I/O, `InnovationTracker`, `GenomeOperators` |
| `src/population.cpp` | `Population`: speciation, reproduction |
| `src/codeepneat.cpp` | Assembly, parallel evaluation, both algorithm drivers |
| `src/rl.cpp` | Environments, A2C, `evaluate_policy` |
| `src/export.cpp` | safetensors, JSON, the export plan, the Python runtime |

`tests/` holds the unit tests, `examples/` the example programs, and `tools/` the PyTorch verification script.

## Design decisions

### Its own autograd engine

The main requirement was tight control over activations, gradients and weights, without dependencies. Owning the engine gives
that directly: values are `std::vector<float>`, every intermediate gradient stays readable, and hooks work on any tensor.
Evolution also benefits, because each candidate network is small and independent, so per-thread CPU execution with no shared state
scales well.

The trade-off is speed on large models, as there's no GPU support and no BLAS.

### Define-by-run graphs

The graph is built as operations execute. Recurrence then needs no special support, because a recurrent step is just more
operations. Unrolled BPTT, variable-length episodes and truncation all fall out of this. Each evolved network has a different
structure, which would make a compiled graph approach awkward.

### One Genome class

Modules, blueprints and DeepNEAT genomes are the same kind of graph, so a single class holds all three, with `GenomeKind` selecting
which operators apply. Speciation, crossover and reproduction are then written once.

### A separate graph IR

`NetworkGraph` sits between genomes and networks for three reasons. Networks can be written by hand without any genome. Tests can
target exact structures. Pruning and validation happen in one place, whatever the source.

### Identity nodes at module boundaries

Each module instance gets `/in` and `/out` nodes that do nothing but merge. They make assembly simple, because a module always has one
entry and one exit. They also give hooks a stable place to attach to "the output of this module". They cost almost nothing, since a
single-input identity returns its input without copying.

### Keeping the DAG invariant

The acyclicity check counts disabled connections too. This means re-enabling a gene can never create a cycle. Crossover takes all
structure from the fitter parent, so a child can't combine two parents' edges into a cycle. Together these make cycles impossible by
construction, rather than something to repair. Recurrence never comes from graph cycles: it lives inside LSTM layers and in cell-skip
genes, which are exempt from the check.

### Persistent species ids

Blueprint nodes refer to module species by id across generations, so ids must outlive any single generation. Species keep their id for
as long as they have members, and ids are never reused. Each species claims its closest genome first during speciation, so a species
doesn't vanish because a threshold moved.

## Concurrency

Evolution itself is single-threaded. Only evaluation is parallel:

1. All jobs are prepared on the main thread, each with a copy of its genotype and its own seed drawn from the master RNG.
2. Workers take jobs from an atomic counter. Each builds its own network and calls the evaluator.
3. Results go into the job's own slot, so workers share nothing but the counter.

Since seeds are assigned before any work starts and results are stored by index, the outcome doesn't depend on the thread count or on
scheduling. `codeepneat_deterministic_across_thread_counts` tests this.

| Shared thing | Rule |
|---|---|
| `LayerRegistry` | Register layers before evolution starts; read-only afterwards |
| `GradMode` | Thread-local |
| Evaluator | Called concurrently; you make it thread-safe |
| `Network`, `Tensor`, `Rng`, `Environment` | One thread at a time |

## Memory

Tensors are reference counted, and a tensor's `parents` keep its inputs alive. So a graph lives as long as something holds its output:

- A loss tensor holds its whole graph until released.
- LSTM state holds every earlier step's graph until `detach_state()`, `reset_state()`, or a run under `NoGradGuard`.
- `record_activations(true)` holds the latest step's graph until the next forward.

Backward closures capture raw pointers only, and `~TensorImpl` releases parents iteratively, so very deep graphs can't overflow the
stack in either `backward()` or destruction.

## Error handling

Errors are exceptions:

| Type | Used for |
|---|---|
| `std::invalid_argument` | Bad shapes, graphs, configurations and names |
| `std::out_of_range` | Bad indices and axes |
| `std::logic_error` | API misuse, such as `backward()` on a tensor without gradients |
| `std::runtime_error` | File I/O and parsing |

An exception thrown while evaluating a candidate network doesn't end the run. It's caught, recorded in `EvaluationRecord::error`, and
the network receives `failed_fitness`.
