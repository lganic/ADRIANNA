# CoDeepNEAT for C++ (reinforcement-learning oriented)

A dependency-free C++17 implementation of **DeepNEAT** and **CoDeepNEAT** from
Miikkulainen et al., *Evolving Deep Neural Networks* (arXiv:1703.00548). It is
built for RL use: evolution is the outer loop, and gradient-based RL (a batched
A2C trainer is included) trains each candidate network. It ships with its own
autograd engine, so every activation, gradient, weight and recurrent state can
be inspected and changed. Evolved models can be exported to PyTorch.

**Documentation is in [docs/](docs/README.md).** Good places to start:

- [Getting started](docs/getting-started.md): build it and evolve a first network.
- [Concepts](docs/concepts.md): how DeepNEAT and CoDeepNEAT work.
- [Reinforcement learning](docs/reinforcement-learning.md): use it on your own problem.
- [Inspection and control](docs/inspection.md): activations, gradients and weights.
- [Configuration](docs/configuration.md) and [Search spaces](docs/search-spaces.md):
  every setting.

The rest of this page is a summary.

```
include/codeepneat/
  export.hpp       PyTorch export (nn.Module source + safetensors weights), weight import, safetensors I/O
  tensor.hpp       reverse-mode autodiff tensors, ops, gradient hooks
  layers.hpp       dense / conv2d / lstm (+ memory-cell skips) / identity, merge op, layer registry
  graph.hpp        NetworkGraph: the inspectable IR between genomes and networks
  network.hpp      executable network + hooks, recording, state, freezing, I/O
  optim.hpp        SGD (momentum / Nesterov), Adam, RMSProp
  hyperparams.hpp  typed hyperparameter tables and search spaces
  spaces.hpp       ready-made spaces (incl. Tables 1 & 2 of the paper)
  genome.hpp       graph genomes, innovation tracking, mutation / crossover / distance
  population.hpp   NEAT speciation and reproduction
  codeepneat.hpp   CoDeepNEAT, DeepNEAT, assembly, evaluator API
  rl.hpp           environments (CartPole, Catch, MemoryCue), A2C trainer, policy evaluation
  all.hpp          umbrella header
```

## Build

```bash
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
```

```bash
cmake --build build
```

```bash
ctest --test-dir build --output-on-failure
```

Targets are `codeepneat` (static library), `cdn_tests`, and the examples
`cartpole_codeepneat`, `memory_lstm`, `catch_conv`, `deepneat_supervised`,
`inspection_demo` and `export_pytorch`. The library only needs the standard library
and threads. Tested with MinGW-w64 GCC 12.2.

## How the paper maps to the code

| Paper | Implementation |
|---|---|
| DeepNEAT node = layer with a table of real/binary hyperparameters (Sec. 3.1) | `NodeGene::hp` (`HyperparamValues`) drawn from a `HyperparamSpace`; `"type"` picks the layer |
| Gaussian mutation of reals, bit-flip of binaries | `HyperparamSpace::mutate` (Gaussian on floats/ints, log-space optional; flip bools; redraw choices) |
| Edges carry no weights, only connectivity | `ConnGene{innovation, from, to, enabled}` |
| Add-node / add-connection mutations, historical markings, crossover | `GenomeOperators::mutate_add_node / mutate_add_connection / crossover`, `InnovationTracker` |
| Speciation by similarity | `GenomeOperators::distance` (excess/disjoint genes + node hyperparameter distance + species-pointer and global differences), `Population::speciate` with a threshold that adapts toward `target_species` |
| Global hyperparameter chromosome | `Genome::globals` (blueprint or DeepNEAT genome), given to the evaluator as `ctx.globals` |
| Merge layer: concatenation or element-wise sum; mismatched parents downsampled to the smallest | `Merge`: spatial inputs max-pooled to the smallest H×W; sum merges project mismatched channels/features with a learned 1×1 conv or a dense bottleneck; mixed spatial + flat inputs are flattened |
| CoDeepNEAT module and blueprint populations (Sec. 3.2) | `CoDeepNEAT::modules()`, `CoDeepNEAT::blueprints()` (`GenomeKind::Module` / `Blueprint`) |
| Blueprint node points to a module species; a module is chosen at random from it; repeated pointers reuse the same module | `CoDeepNEAT::sample_assembly`: one module per species per network |
| Fitness of an assembled network goes back to blueprints and modules as the average | `CoDeepNEAT::step`: per-genome `fitness_samples`, then their mean |
| 25 blueprints, 45 modules, 100 networks per generation (CIFAR-10 setup) | `CoDeepNEATConfig` defaults |
| Table 1 / Table 2 search spaces | `spaces::cifar10_nodes/globals`, `spaces::captioning_nodes/globals` |
| LSTM extension (Sec. 4): enable/disable connections between LSTM layers; add/remove skip connections between memory cells (Fig. 3) | `mutate_toggle_connection`; `mutate_add_cell_skip` / `mutate_toggle_cell_skip` produce `cell_skips` genes, built as gated memory-cell connections in `LSTMLayer`. A connection from a later LSTM to an earlier one becomes a feedback connection that uses the previous time step's cell |
| Multi-input networks (image embedding + text, Sec. 5) | `IOSpec::inputs` can hold several inputs; each blueprint has one virtual input node per network input |
| Paths that bypass whole modules (Fig. 5) | add-connection can link any input or node to the output, at blueprint and module level |

### Structures evolution can produce
- Chains, branches and merges (concat or sum) of modules; residual-style
  identity paths (`in -> out` connections inside modules and blueprints).
- The same module repeated in several places (Fig. 2). Each repeat has its own
  weights.
- Conv stacks with per-layer filters, kernel size, max pooling, dropout and
  initial-weight scaling; merges across resolutions.
- Dense/LSTM hybrids with gated memory-cell skip and feedback connections
  between LSTM layers.
- Networks with several inputs and several output heads (e.g. actor and
  critic).
- Your own layer types: `LayerRegistry::instance().add("mytype", factory)`, then
  put `"mytype"` into the space's `type` choices.

## Using it for RL

```cpp
#include "codeepneat/all.hpp"
using namespace cdn;

rl::EnvFactory make_env = [] { return std::unique_ptr<rl::Environment>(new rl::CartPole()); };

CoDeepNEATConfig cfg;
cfg.io = rl::actor_critic_io(*make_env());           // input "obs"; heads "policy" + "value"
cfg.module_node_space = spaces::recurrent_layers();  // dense + LSTM nodes
cfg.global_space = spaces::rl_globals();             // lr, optimizer, momentum, entropy coef, ...
cfg.num_threads = 8;

CoDeepNEAT evo(cfg);
evo.run(20, [&](EvalContext& ctx) {                  // ctx.net is freshly built
  rl::A2CTrainer trainer(ctx.net, make_env, rl::A2CConfig{}, ctx.rng.next_u64());
  trainer.apply_globals(ctx.globals);                // use the evolved hyperparameters
  trainer.train();
  return rl::evaluate_policy(ctx.net, make_env, 10, /*greedy=*/true, ctx.rng.next_u64());
}, [](const GenerationStats& s) { std::cout << s.to_string() << "\n"; return true; });

evo.best().save("best.txt");                         // genotype; reload with AssembledGenome::load
auto net = evo.build(evo.best());                    // rebuild the champion and keep training
```

The evaluator is any `double(EvalContext&)`, so you can plug in DQN, PPO, or
evolution strategies. It runs on several threads at once, so give each call its
own environment. Results do not depend on the thread count, because every
evaluation gets its own seeded RNG.

`Network::forward` runs **one time step**. Recurrent layers keep their state
between calls, so backpropagation through time is just: forward T times, then
`backward()`. To manage that state, use `reset_state(batch)`,
`detach_state()` (truncated BPTT), `mask_state(keep)` (reset finished episodes
in a batch) and `get_state()` / `set_state()` (save or restore, e.g. for
bootstrapping or a recurrent replay buffer).

## Inspecting and changing activations, gradients and weights

```cpp
// Weights: named, share storage with the network, editable in place, freezable.
for (auto& p : net.named_parameters()) std::cout << p.name << shape_str(p.tensor.shape());
net.parameter("bp2/m7/n3.weight").data()[0] = 0.f;
net.set_trainable("bp2/", false);                                        // freeze by name prefix
net.set_trainable([](const NodeInfo& n) { return n.type == "lstm"; }, false);  // or by filter

// Activations: forward hooks observe or replace any node's output.
net.add_forward_hook([](const NodeInfo& n, Tensor& out) { /* stats, logging */ });
net.add_forward_hook("bp4/m15/out", [](const NodeInfo&, Tensor& out) { out = out * 0.f; });  // ablation
net.add_forward_pre_hook([](const NodeInfo& n, Tensor& in) { /* the merged input of a layer */ });
net.record_activations(true);            // keep every node's output of the last step
net.activation("bp2/m7/n3");             // ... and, after backward(), its .grad()

// Gradients: w.r.t. activations or parameters; read, rescale or replace them.
net.add_backward_hook([](const NodeInfo& n, Tensor& g) { /* clip / log / edit */ });
net.add_parameter_hook("head:policy.weight", [](Tensor& g) { for (auto& v : g.data()) v *= 0.1f; });
x.register_hook([](Tensor& g) { /* works on any tensor, inside or outside a network */ });

// Internals.
auto* lstm = dynamic_cast<LSTMLayer*>(net.layer("bp2/m7/n3"));
lstm->last_gates().at("f");  lstm->cell();  lstm->hidden();
Adam adam(net.named_parameters(), 1e-3f);
adam.state("bp2/m7/n3.weight")->m;   // Adam / momentum buffers
net.summary(); net.to_dot(); net.save_weights("w.txt");
```

Node names follow the genotype: `input:<name>`, `bp<blueprint node>/m<module
id>/in|n<module node>|out`, `body`, `head:<name>` (DeepNEAT: `n<node>`).
`NodeInfo::origin` holds the blueprint node, module species, module id and
module node, so a hook or a freeze can target, for example, "every layer that
came from species 3".

In-place edits to `.data()` bypass autograd, which suits weight surgery. To
change an activation so that gradients see the change, assign a new tensor in
the hook, e.g. `out = out * mask`.

## Exporting to PyTorch

Any built `Network` can be exported to plain PyTorch:

```cpp
#include "codeepneat/export.hpp"
export_torch(*evo.build(evo.best()), "policy");   // writes policy.py + policy.safetensors
```

`policy.py` is self-contained: it only imports `torch`, not this library or
the `safetensors` package. It defines an `nn.Module` that reproduces the
network exactly: merges with pooling and projections, strided and pooled
convolutions, stateful LSTMs, and forward and feedback memory-cell skips.
`policy.safetensors` holds the weights in the standard safetensors format,
keyed by the module's `state_dict()` names.

```python
from policy import EvolvedNet
model = EvolvedNet.from_pretrained("policy.safetensors")
logits, value = model(obs)            # one time step: [B, ...] or unbatched input
model.reset_state(batch)              # also detach_state / mask_state / get_state / set_state
model.nodes["bp3_m17_n5"]             # every graph node is a submodule; PyTorch hooks work on it
model.node("bp3/m17/n5")              # ... or look it up by its C++ name
model.save_pretrained("tuned.safetensors")
```

- **Training continues as normal.** The module is an ordinary `nn.Module`
  (`torch.optim`, `.to(device)`, `.double()`, ...). Weights trained in PyTorch
  come back with `cdn::import_torch_weights(net, "tuned.safetensors")`.
  `torch_parameter_names(net)` lists how C++ parameter names map to PyTorch
  keys.
- **Tracing.** `torch.jit.trace` has been tested on feed-forward exports. Recurrent
  exports keep their state in Python attributes between calls, so keep them in eager
  mode. ONNX export hasn't been tested.
- **Unsupported layers.** Custom layer types added through `LayerRegistry`
  have no PyTorch counterpart, so exporting a network that contains one throws
  `std::invalid_argument`.
- **Verification.** `export_torch_reference(net, steps, path)` records the
  inputs, outputs and parameter gradients of a multi-step run.
  `tools/verify_torch_export.py` replays that run in PyTorch and compares the
  results. `ctest` runs the full round trip automatically when Python with
  `torch` is found: export, verify in PyTorch, one SGD step in PyTorch, then
  import back into C++. Point CMake at the right interpreter with
  `-DPython3_EXECUTABLE=...`. On this machine the outputs and gradients of
  every exported network matched C++ to within float32 rounding (≤ 7e-6).

## Examples

| Example | What it shows |
|---|---|
| `cartpole_codeepneat [gens] [threads] [steps]` | CoDeepNEAT + A2C on CartPole; saves the champion's genome, weights and graph |
| `memory_lstm [gens] [threads] [delay]` | Dense/LSTM search space with cell-skip mutations on a task that needs memory |
| `catch_conv [gens] [threads]` | Conv search space on pixel observations |
| `deepneat_supervised [gens]` | DeepNEAT and CoDeepNEAT side by side (two-spirals classification) |
| `inspection_demo` | Hooks, ablation, gradient clipping per node, freezing, optimizer and LSTM internals |
| `export_pytorch [dir] [--check-import]` | Exports a hand-built network covering every feature, 8 evolved conv/dense/LSTM networks with two inputs, and a trained CartPole policy, together with reference runs; `--check-import` loads the weights back from PyTorch |

## Design notes and deviations

- **Training from scratch.** Like the paper, every assembled network starts
  from fresh random weights and is trained by your evaluator. No weights are
  inherited between generations. For Lamarckian variants, copy weights between
  networks with `state_dict()` / `load_state_dict(…, strict=false)`, which match
  parameters by name.
- **Modules no network used this generation** keep their previous fitness
  (elites). Otherwise they get their species' mean, or the population mean.
  The paper does not say how to handle this case.
- **Memory-cell skip** into LSTM *d* from LSTM *s*:
  `c_d += sigmoid(W [x_t, h_{t-1}] + b) * P c_s`. `P` is a learned projection
  when the sizes differ. The gate bias starts at -1, so a new skip only nudges
  the cell. `c_s` is the current step's cell if *s* runs before *d*, otherwise
  the previous step's (feedback).
- **Initial genomes** are minimal: every input goes to one hidden node, which
  goes to the output. Topologies grow through mutation.
  `MutationParams::initial_structural_mutations` adds diversity at start.
  `max_nodes` caps genome size.
- **Compute.** CPU only, float32, plain loops. Parallelism comes from evaluating
  networks on separate threads. The engine targets the small and medium
  networks typical in RL research, not ImageNet-scale models.
