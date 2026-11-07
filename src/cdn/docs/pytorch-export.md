# PyTorch export

Any built `Network` can be exported to plain PyTorch, and weights trained in PyTorch can come
back to C++.

## Export

```cpp
#include "codeepneat/export.hpp"
```

```cpp
auto champion = evo.build(evo.best());
// ... train it ...
TorchExportOptions options;
options.class_name = "Policy";
options.description = "CartPole actor-critic.";
export_torch(*champion, "policy", options);
```

This writes two files:

| File | Contents |
|---|---|
| `policy.py` | A self-contained `nn.Module`. It imports only `torch` and the standard library |
| `policy.safetensors` | Weights in the standard [safetensors](https://github.com/huggingface/safetensors) format, keyed by `state_dict()` names |

Remember that `evo.build` creates fresh weights. Train the network before exporting it, or you'll
export a random one. If the prefix includes a directory, it must already exist.

## Use in PyTorch

```python
import torch
from policy import Policy

model = Policy.from_pretrained("policy.safetensors").eval()

with torch.no_grad():
    logits, value = model(torch.zeros(1, 4))
```

Running `python policy.py` prints a summary and does a test forward pass, which is a quick way to
check an export.

### Calling the model

`forward` takes one tensor per network input, in order, or a dict by name. Like the C++ version, it
runs **one time step**. An input without a batch axis is treated as a batch of one.

It returns a tensor for a single head and a tuple for several. `named_forward` returns a dict instead:

```python
outs = model.named_forward(torch.zeros(1, 4))   # {"policy": ..., "value": ...}
print(model.input_names, model.input_shapes, model.output_names)
```

### Recurrent models

State lives in the module, and the methods mirror the C++ API:

| Method | Use |
|---|---|
| `model.reset_state(batch, device=None, dtype=None)` | Zero the state |
| `model.detach_state()` | Truncate backpropagation |
| `model.mask_state(keep)` | Multiply each state row by `keep[b]` |
| `model.get_state()` / `model.set_state(state)` | Dict keyed `<C++ node name>.h` and `.c` |
| `model.is_recurrent` | Whether there are any LSTM nodes |

Recurrent state is not exported. A model always starts from zeros, and the state resets automatically
when the batch size changes.

### Training

The model is a normal `nn.Module`, so `model.parameters()`, `torch.optim`, `.to(device)`, `.double()` and
`state_dict()` all work.

### Hooks and inspection

Every graph node is a submodule, so PyTorch hooks work on any of them:

```python
activations = {}

def record(module, inputs, output):
    activations[module.cpp_name] = output.detach()   # return None, or the output is replaced

for node in model.nodes.values():
    node.register_forward_hook(record)

print(model.summary())
```

`model.nodes` uses sanitised keys, such as `bp3_m17_n5` for `bp3/m17/n5`. `model.node("bp3/m17/n5")` looks
a node up by its C++ name. Each node has `cpp_name`, `layer_type`, `origin` and `hparams`. A node's `merge`
submodule, if it has one, can be hooked to see the merged input.

LSTM nodes expose `.h` and `.c`. `model.cpp_parameter_names` maps every C++ parameter name to its
`state_dict` key.

## Bring weights back to C++

```python
model.save_pretrained("tuned.safetensors")
```

```cpp
import_torch_weights(*champion, "tuned.safetensors");
```

The network must have the same architecture. With `strict` left at `true`, a missing or wrongly shaped
tensor throws. Files written by the `safetensors` package work too. F32, F64, F16 and BF16 are all read and
converted to float32.

## Parameter names

| C++ parameter | PyTorch key |
|---|---|
| `<node>.weight`, `.bias` on dense layers and heads | `nodes.<key>.linear.weight`, `.bias` |
| `<node>.weight`, `.bias` on conv2d | `nodes.<key>.conv.weight`, `.bias` |
| `<node>.weight`, `.bias` on lstm | `nodes.<key>.weight`, `.bias` |
| `<node>.skip<k>.gate_weight`, `.gate_bias` | `nodes.<key>.skips.<k>.gate.weight`, `.bias` |
| `<node>.skip<k>.proj` | `nodes.<key>.skips.<k>.proj.weight` |
| `<node>.merge.proj<k>` | `nodes.<key>.merge.proj.<k>.weight` |

`torch_parameter_names(net)` returns this mapping for a specific network.

`<key>` is the node name with anything other than letters, digits and `_` replaced by `_`. A name that starts
with a digit gets an `n` prefix, one that clashes with an `nn.Module` attribute gets a `_` suffix, and
duplicates are numbered.

## What is and isn't preserved

| Exported | Not exported |
|---|---|
| Architecture, weights and biases | Optimizer state |
| Dropout rates (active in `model.train()`) | Recurrent state |
| Activation functions, pooling, strides and padding | Dropout masks, which come from PyTorch's RNG |
| Merges with their pooling and projections | The genome and its global hyperparameters |
| Memory-cell skips, including feedback | Your training or environment code |

The node's hyperparameter table is included for reference, as text in the `hparams` attribute.

## Limitations

- **Custom layers can't be exported.** A network containing a layer type added through `LayerRegistry` makes
  `export_torch` throw `std::invalid_argument`, because the exporter has no PyTorch code for it.
- **Tracing.** `torch.jit.trace` has been tested on a feed-forward export. A recurrent model's state lives in
  Python attributes between calls, so a trace would capture one step with its state baked in. Keep
  recurrent models in eager mode.
- **`torch.jit.script` isn't supported**, because `forward` is driven by a dict-based execution plan.
- **ONNX export hasn't been tested.**
- **Byte order.** The generated Python loader assumes a little-endian machine. The C++ code does not.

## Verifying an export

Differences between two implementations are easy to miss, so the library can check itself. First record a
reference run in C++:

```cpp
std::vector<std::vector<Tensor>> steps;       // one vector of inputs per time step
for (int t = 0; t < 4; ++t) steps.push_back({Tensor::randn({3, 8}, rng)});
export_torch(net, "policy");
export_torch_reference(net, steps, "policy.reference.safetensors");
```

Then replay it in PyTorch:

```bash
python tools/verify_torch_export.py policy
```

```text
OK   handbuilt                                   6465 params  4 steps  max|dy|=1.79e-07  max|dgrad|=1.79e-07  (recurrent)
all exports match
```

`max|dy|` is the largest output difference, and `max|dgrad|` is the largest difference in the parameter gradients of
`sum_t sum_heads 0.5 * |y|^2`. The default tolerance is `rtol = 1e-4`, `atol = 1e-5` for outputs and ten times that
for gradients. Change it with `--rtol` and `--atol`.

| Option | Effect |
|---|---|
| `--dir DIR` | Verify every `*.reference.safetensors` in a directory |
| `--write-back` | Also take one SGD step and write `<prefix>.torch_trained.safetensors` and `<prefix>.torch_outputs.safetensors` |

`examples/export_pytorch.cpp` exercises this on a hand-built network with every feature, on eight evolved networks,
and on a trained policy. With Python and torch available, CTest runs the whole loop:

```text
torch_export_generate   export the networks and reference runs
torch_export_verify     replay in PyTorch, take an SGD step, save
torch_export_import     load back into C++ and compare outputs
```

## API summary

| Function | Purpose |
|---|---|
| `export_torch(net, prefix, options)` | Write `<prefix>.py` and `<prefix>.safetensors` |
| `torch_module_source(net, options)` | The Python source as a string |
| `save_torch_weights(net, path)` | Weights only, under PyTorch names |
| `import_torch_weights(net, path, strict)` | Load weights saved from PyTorch |
| `torch_parameter_names(net)` | C++ name to `state_dict` key |
| `export_torch_reference(net, steps, path)` | Record a reference run |
| `save_safetensors(tensors, path, metadata)` | General safetensors writer (float32) |
| `load_safetensors(path, &metadata)` | General safetensors reader |

The generated file also exposes `load_safetensors(path)` and `save_safetensors(tensors, path, metadata=None)`, so
you don't need the `safetensors` package.

The file format is described under [exported modules](file-formats.md#exported-pytorch-modules).
