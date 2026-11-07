# Networks

Evolution produces genomes, but everything you train and inspect is a `Network`. This
page covers how networks are described, built and run. For hooks and other runtime
control, see [Inspection and control](inspection.md).

```text
Genome(s) --assemble_network--> NetworkGraph --Network(graph)--> executable network
```

## NetworkGraph

A `NetworkGraph` is a plain description of a network: a directed acyclic graph of nodes,
plus memory-cell skip edges. Evolution generates these, but you can write one by hand,
for a baseline or a test:

```cpp
HyperparamValues dense;
dense.set_int("units", 32).set_choice("activation", "tanh");
HyperparamValues lstm;
lstm.set_int("units", 16);

NetworkGraph g;
int obs = g.add_input("obs", {8});
int hidden = g.add_layer("hidden", "dense", dense, {obs});
int memory = g.add_layer("memory", "lstm", lstm, {hidden});
int both = g.add_identity("both", {hidden, memory}, "concat");
g.add_output("head:policy", both, 4, "linear", 0.1f);
g.add_output("head:value", both, 1);

Network net(g, BuildOptions(), /*seed=*/7);
std::cout << net.summary();
```

`add_*` methods return the node's id, which later nodes use in their input lists. Node
names must be unique. `g.to_dot()` produces Graphviz source, where a dashed red edge is a
memory-cell skip.

The `Network` constructor then:

1. Removes nodes that aren't on a path from an input to an output (input nodes always
   stay, so the input list never changes).
2. Validates the graph, throwing `std::invalid_argument` on problems such as cycles,
   duplicate names, or unknown merges.
3. Sorts nodes topologically. Ties break by id, so execution order is deterministic.
4. Infers every shape, creates the merge operations and layers, and initialises weights
   from the seed.

## Shapes

Shapes in graphs and layer definitions are **per sample**: `{F}` for vectors and
`{C, H, W}` for images. Tensors passed to `forward` have a batch axis in front. For
convenience, an input of exactly the per-sample shape is treated as a batch of one.

Layers adapt to what they receive:

| Layer | Given `{F}` | Given `{C, H, W}` |
|---|---|---|
| `dense`, `lstm` | Used directly | Flattened |
| `conv2d` | Treated as an `{F, 1, 1}` image | Used directly |

## Layers

| Type | Output | Notes |
|---|---|---|
| `dense` | `{units}` | Linear, activation, dropout |
| `conv2d` | `{filters, H', W'}` | Convolution, activation, optional 2x2 max pool, dropout. With the default padding and an odd kernel, H and W are kept unless the stride or pool reduces them |
| `lstm` | `{units}` | One step per `forward`. Gates are ordered i, f, g, o in `weight` and `bias`. Dropout applies to the output, not to the state |
| `identity` | Same as input | Returns its input tensor itself |

Hyperparameters are listed in [Search spaces](search-spaces.md#layer-hyperparameters).

## Merging

A node with several inputs merges them before its layer runs. The rules follow section 3.1
of the paper, where mismatched parents are reduced to the smallest size.

| Inputs | `concat` | `sum` |
|---|---|---|
| All spatial | Max pool each to the smallest H and W, then join channels | Same pooling. Inputs whose channel count isn't the smallest get a learned 1x1 convolution (no bias) down to it |
| All flat | Join features | Inputs larger than the smallest get a learned dense projection (no bias) down to it |
| Mixed | Flatten everything, then as for flat | Flatten everything, then as for flat |

Pooling uses adaptive max pooling, so any size can reach any smaller size. Projection
parameters belong to the node and are named `<node>.merge.proj<k>`, where `k` is the index
of the input.

A merge cannot enlarge anything, so an early small feature map limits everything merged
with it.

## Node names in evolved networks

Names come from the genotype, so you can tell what any layer is:

| Name | Meaning |
|---|---|
| `input:obs` | The input named `obs` |
| `bp3/m17/in` | Input of the module instance at blueprint node 3, using module 17 |
| `bp3/m17/n5` | Layer from module 17's node 5, at blueprint node 3 |
| `bp3/m17/out` | Output of that module instance |
| `n5` | Layer from node 5 of a DeepNEAT genome |
| `body` | Merge point feeding all heads |
| `head:policy` | The output head named `policy` |

`NodeInfo::origin` has the same information as fields: `role` (`input`, `module_in`,
`module_out`, `layer`, `body`, `head`), `blueprint_node`, `species`, `module_id` and
`module_node`. The two instances of a repeated module have different `bp` numbers, and each
has its own weights.

Parameters are named `<node>.<local>`, such as `bp3/m17/n5.weight`:

| Layer | Local names |
|---|---|
| `dense`, `conv2d` | `weight`, `bias` |
| `lstm` | `weight`, `bias`, and per skip `k`: `skip<k>.gate_weight`, `skip<k>.gate_bias`, `skip<k>.proj` |
| merge | `merge.proj<k>` |

## Running a network

`forward` has three forms:

```cpp
auto outs = net.forward(std::vector<Tensor>{x});  // every head, in output order
Tensor first = net.forward(x);                    // single input; first head only
auto named = net.forward(std::map<std::string, Tensor>{{"obs", x}});  // by name
Tensor logits = outs[net.output_index("policy")];
std::cout << first.shape().size() << named.size() << logits.numel();
```

Be careful with braces: `net.forward({x})` selects the `Tensor` overload, so it returns only
the **first head**, and `net.forward({a, b})` does not compile. Write
`std::vector<Tensor>{a, b}`.

`output_index` and `input_index` accept names with or without the `head:` and `input:`
prefix. They return -1 for unknown names.

### Modes

`net.train()` and `net.eval()` switch dropout on and off. A network starts in training mode.
This is separate from gradient recording, which `NoGradGuard` turns off:

```cpp
net.eval();
NoGradGuard no_grad;  // until the end of this scope, on this thread
Tensor logits = net.forward(x);
std::cout << logits.requires_grad();  // 0
```

Use both when acting in an environment. Recording a graph you will never use wastes time and
memory.

### Recurrent state

Each `forward` is one time step, and LSTM layers keep `h` and `c` between calls.
Backpropagation through time needs no special handling:

```cpp
net.reset_state(4);  // zero state for 4 sequences
Tensor loss = Tensor::scalar(0.0f);
for (int t = 0; t < 10; ++t) {
  Tensor out = net.forward(x);
  loss = loss + mse_loss(out, target);
}
net.zero_grad();
loss.backward();  // flows through all 10 steps
```

| Method | Use |
|---|---|
| `reset_state(batch)` | Zero all state. Call at the start of an episode |
| `detach_state()` | Keep the values but cut the graph, for truncated BPTT |
| `mask_state(keep)` | Multiply row `b` by `keep[b]`; 0 resets one sequence in a batch |
| `get_state()` / `set_state(s)` | Save and restore, for example to peek one step ahead |

If the batch size of an input differs from the current state's, the state is reset to zeros
automatically. On a long run, call `detach_state()` regularly, because state tensors keep the
whole history's graph alive until you do.

## Training

Pair a network with an [optimizer](tensors.md#optimizers):

```cpp
Adam adam(net.named_parameters(), 1e-3f);
for (int step = 0; step < 100; ++step) {
  net.reset_state(4);
  Tensor loss = mse_loss(net.forward(x), target);
  adam.zero_grad();
  loss.backward();
  clip_grad_norm(net.parameters(), 1.0f);
  adam.step();
}
```

`make_optimizer_from_globals(globals, params)` picks the optimizer from evolved hyperparameters.

## Saving and loading

| To save | Use |
|---|---|
| Architecture | `AssembledGenome::save` / `load`, then `evo.build` |
| Weights, text | `net.save_weights(path)` / `load_weights(path)` |
| Weights, in memory | `net.state_dict()` / `load_state_dict(state)` |
| Weights, safetensors | `save_torch_weights` / `import_torch_weights` |
| For PyTorch | `export_torch` ([details](pytorch-export.md)) |

Weights match by parameter name, so architecture and weights must be saved together. With
`strict = false`, missing or mismatched parameters are skipped instead of throwing.

To restore a complete network:

```cpp
AssembledGenome genome = evo.best();
genome.save("best_genome.txt");
auto first = evo.build(genome);
first->save_weights("best_weights.txt");

// Later, with the same IOSpec and recurrent layer types:
AssembledGenome loaded = AssembledGenome::load("best_genome.txt");
Network restored(assemble_network(loaded, cfg.io), default_build_options(loaded.globals()), 1);
restored.load_weights("best_weights.txt");
```

A `Network` can't be copied. To duplicate one, for example as a DQN target network, build a
second network from its graph and copy the weights:

```cpp
Network target(net.graph(), BuildOptions(), 1);
target.load_state_dict(net.state_dict());
```

## Inspection helpers

| Call | Returns |
|---|---|
| `net.summary()` | Table of nodes, types, shapes and parameter counts |
| `net.to_dot()` | Graphviz source. Render it with `dot -Tpng net.dot -o net.png` |
| `net.nodes()` | `NodeInfo` for every node, in execution order |
| `net.node(name)`, `net.has_node(name)` | One node |
| `net.layer(name)`, `net.layers()` | The `Layer` objects |
| `net.merge(name)` | The `Merge` of a node with several inputs, or null |
| `net.graph()` | The pruned `NetworkGraph` |
| `net.num_parameters()` | Total scalar parameters |
