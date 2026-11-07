# Examples

The programs in `examples/` are built into `build/`. They write output files to the current
directory. All arguments are optional and positional, and `threads` defaults to the number of hardware
threads.

| Program | Task | Search space | Shows |
|---|---|---|---|
| [`cartpole_codeepneat`](#cartpole_codeepneat) | CartPole | Dense | The full workflow, including export |
| [`memory_lstm`](#memory_lstm) | MemoryCue | Dense and LSTM | Recurrent evolution, cell skips |
| [`catch_conv`](#catch_conv) | Catch, from pixels | Convolutional | Conv search, merges across resolutions |
| [`deepneat_supervised`](#deepneat_supervised) | Two spirals | Dense | DeepNEAT next to CoDeepNEAT |
| [`inspection_demo`](#inspection_demo) | None | Dense and LSTM | The inspection API |
| [`export_pytorch`](#export_pytorch) | None | Mixed | Export, verification, import |

These tasks are small on purpose, so that each example finishes in seconds. A single good network often appears
in the first generations, so they demonstrate the machinery working more than they demonstrate what evolution can
find. Watch the `mean` fitness to see evolution at work: it shows the whole population learning to be trainable.

## cartpole_codeepneat

```bash
build/cartpole_codeepneat 10 8 12000
```

Arguments are `generations`, `threads` and `train_steps`, with defaults 10, hardware threads, and 12000.

Each network trains with A2C for `train_steps` environment steps, and its fitness is the mean greedy return over five
episodes. The run stops as soon as a network scores 500. The champion is then rebuilt, retrained for twice as long and
scored over 20 episodes.

Files written:

| File | Contents |
|---|---|
| `cartpole_best_genome.txt` | The champion's genotype |
| `cartpole_best_weights.txt` | The retrained weights |
| `cartpole_best.dot` | The graph, for Graphviz |
| `cartpole_policy.py`, `cartpole_policy.safetensors` | The PyTorch export |

## memory_lstm

```bash
build/memory_lstm 8 8 10
```

Arguments are `generations`, `threads` and `delay`, with defaults 8, hardware threads, and 10.

A memoryless policy scores 0 on this task and perfect memory scores 1, so a fitness above 0 means the network is using
its recurrent state. The example raises `add_cell_skip` and sets `rollout_length` to `delay + 2`. At the end it prints the
champion's structure, retrains it, and traces one episode showing the policy and the cell-state norm at each step.

## catch_conv

```bash
build/catch_conv 6 8
```

Arguments are `generations` and `threads`, with defaults 6 and hardware threads. Writes `catch_best.dot`.

The observation is a `[1, 10, 5]` image. Modules are graphs of `conv2d` nodes, kept small by `max_nodes = 4`. Blueprints
can connect modules across resolutions, which exercises the pooling in merges.

To view the graph:

```bash
dot -Tpng catch_best.dot -o catch.png
```

## deepneat_supervised

```bash
build/deepneat_supervised 10
```

The argument is `generations`, default 10. This runs DeepNEAT and then CoDeepNEAT on the same two-spirals classification
task, with the same training budget and fitness (validation accuracy). It's the simplest place to see the supervised
workflow the paper uses. It prints the best network from each.

## inspection_demo

```bash
build/inspection_demo
```

It grows a non-trivial network with an evaluator that rewards size, then goes through:

1. Listing and editing parameters.
2. Activation statistics from a forward hook.
3. Activation gradient norms after `backward()`.
4. Per-node gradient clipping with a backward hook.
5. Ablating a module and measuring the effect.
6. Parameter gradient hooks, and freezing a blueprint node.
7. A few training steps, then reading Adam's buffers, LSTM gates and state.

Writes `inspection_weights.txt`, `inspection_genome.txt` and `inspection.dot`. [Inspection and control](inspection.md) covers
the same ground as a cookbook.

## export_pytorch

```bash
build/export_pytorch torch_export
```

```bash
python tools/verify_torch_export.py --dir torch_export --write-back
```

```bash
build/export_pytorch torch_export --check-import
```

The first command exports ten networks and a reference run for each: `handbuilt` (every feature, two inputs), `zoo_0` to `zoo_7`
(evolved conv, dense and LSTM networks) and `cartpole` (trained). The second replays them in PyTorch and takes a training step.
The third rebuilds the same networks, imports the updated weights and compares outputs with PyTorch's. See
[PyTorch export](pytorch-export.md).

The third step depends on evolution being deterministic, so it must run with the same build as the first.
