# File formats

All formats below are stable within this version. Text formats use whitespace-separated tokens, so line breaks and
indentation are not significant to the parsers.

| What | Written by | Read by | Format |
|---|---|---|---|
| Genotype | `AssembledGenome::save`, `Genome::write` | `AssembledGenome::load`, `Genome::read` | Text |
| Weights | `Network::save_weights` | `Network::load_weights` | Text |
| Weights | `save_torch_weights`, `save_safetensors` | `import_torch_weights`, `load_safetensors` | safetensors |
| Reference run | `export_torch_reference` | `tools/verify_torch_export.py` | safetensors |
| PyTorch module | `export_torch` | Python | Python source |
| Graph | `Network::to_dot`, `NetworkGraph::to_dot` | Graphviz | DOT |

## Genomes

A single genome:

```text
genome <id> <kind> <num_inputs> <out_merge> <fitness> <species>
globals <hyperparameter table>
nodes <count>
node <id> <species pointer> <hyperparameter table>
conns <count>
conn <innovation> <from> <to> <enabled>
skips <count>
skip <innovation> <from> <to> <enabled>
end
```

`kind` is `module`, `blueprint` or `deepneat`. `enabled` is 0 or 1. The species pointer is -1 outside blueprints.

An assembled genome wraps a blueprint and one module per referenced species:

```text
assembled <fitness> <generation> <module count>
<blueprint genome>
module_for_species <species id>
<module genome>
```

A DeepNEAT result is an assembled genome with a `deepneat` genome in the blueprint position and no modules.

This real example is a one-node blueprint:

```text
assembled 500 2 1
genome 29 blueprint 1 concat 395.5 1
globals entropy_coef:f:0.01024395771,learning_rate:f:0.002339921393,nesterov:b:0,optimizer:c:adam
nodes 1
node 2 0 merge:c:concat
conns 2
conn 0 0 2 1
conn 1 2 1 1
skips 0
end
module_for_species 0
genome 49 module 1 concat 231.57499999999999 0
globals -
```

Node ids follow the [genome's convention](concepts.md): with one input, 0 is the input, 1 the output, and 2 upward are hidden
nodes.

`Genome::read` validates what it reads and throws `std::runtime_error` if the genome is malformed.

### Hyperparameter tables

A table is one token with no whitespace, of the form `name:kind:value` joined by commas. `-` means empty.

| Kind | Type | Value |
|---|---|---|
| `f` | Float | Number (10 significant digits) |
| `i` | Int | Number |
| `b` | Bool | 0 or 1 |
| `c` | Choice | The string |

This is why names and choice strings can't contain commas, colons or whitespace.

### What genomes don't contain

The `IOSpec`, the search spaces and the recurrent layer types aren't saved. To rebuild a network you need the same `IOSpec`, which
your program provides:

```cpp
AssembledGenome genome = AssembledGenome::load("best.txt");
NetworkGraph graph = assemble_network(genome, cfg.io);
```

Trained weights aren't part of a genome either. Whole populations can't be saved, so a run can't be resumed from a file.

## Text weights

```text
cdn-weights 1 <count>
<name> <rank> <dim> <dim> ...
<value> <value> ...
```

The header is followed by two lines per tensor. Values have nine significant digits, which is enough to reproduce a float32
exactly. Names can't contain whitespace, so custom node names must not either.

`Network::load_weights(path, strict)` matches by name. With `strict = true` a missing or wrongly sized parameter throws; with `false`
it is skipped. Extra tensors in the file are ignored.

## safetensors

This is the [standard format](https://github.com/huggingface/safetensors): an 8-byte little-endian header length, a JSON header, and raw
little-endian tensor data.

```text
{"__metadata__":{"format":"codeepneat-torch"},
 "nodes.head_value.linear.weight":{"dtype":"F32","shape":[1,55],"data_offsets":[0,220]}, ...}
```

The writer always produces F32 and pads the header with spaces to a multiple of 8 bytes. The reader accepts F32, F64, F16 and BF16,
converting to float32, and rejects other types. Files from the Python `safetensors` package work.

`save_torch_weights` uses PyTorch `state_dict` keys. `save_safetensors` writes whatever names you give it.

### Reference runs

Files from `export_torch_reference` contain:

| Tensor | Contents |
|---|---|
| `step<t>.input<k>` | Input `k` at step `t` |
| `step<t>.output<k>` | Head `k` at step `t` |
| `grad.<state_dict key>` | Gradient of the loss with respect to that parameter |

The loss is `sum_t sum_k 0.5 * |output|^2`, from a zero recurrent state in eval mode.

| Metadata key | Value |
|---|---|
| `format` | `codeepneat-torch-reference` |
| `steps`, `inputs`, `outputs` | Counts |
| `loss` | The loss value |

## Exported PyTorch modules

The generated `.py` file has three parts: a fixed runtime (layer classes and the `CoDeepNEATModule` base class), the `SPEC` constant
with the architecture as JSON, and a small subclass with your class name.

`SPEC` has these fields:

| Field | Contents |
|---|---|
| `format`, `version` | `codeepneat-torch`, 1 |
| `inputs` | List of `{name, shape}` |
| `outputs` | Output node names |
| `output_keys` | The same, as node keys |
| `nodes` | All nodes, in execution order |
| `params` | C++ parameter name to `state_dict` key |

Each entry in `nodes` has:

| Field | Contents |
|---|---|
| `name`, `key` | C++ name and sanitised key |
| `kind` | `input`, `layer` or `output` |
| `cls` | `input`, `identity`, `dense`, `conv2d` or `lstm` |
| `out_shape` | Per-sample output shape |
| `origin` | `role`, `blueprint_node`, `species`, `module_id`, `module_node` |
| `hp` | The hyperparameter table, as text |
| `slot`, `shape` | Input nodes only: position and shape |
| `inputs` | Parent keys (all except input nodes) |
| `merge` | Null, or `{method, mode, pool, proj}` |
| `dense` | `{in, units, bias, activation, dropout}` |
| `conv` | `{image_shape, in_channels, filters, kernel, stride, padding, bias, pool, activation, dropout}` |
| `lstm` | `{in, units, dropout, skips}`, each skip being `{source, source_units, proj}` |

In `merge`, `mode` is `spatial` or `flat`, `pool` is the target height and width, and `proj` maps an input index to `[fan_in, fan_out]`.

## Graphviz

`to_dot()` gives a `digraph` with diamonds for inputs, boxes for layers (dashed for identity nodes), double circles for outputs, and
dashed red edges labelled `cell` for memory-cell skips.

```bash
dot -Tpng cartpole_best.dot -o cartpole.png
```
