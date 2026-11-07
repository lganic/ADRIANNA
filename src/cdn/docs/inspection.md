# Inspection and control

Everything inside a network can be read and changed while it runs. This page is a cookbook. The
complete tour in `examples/inspection_demo.cpp` runs on an evolved network.

## Finding what to inspect

```cpp
std::cout << net.summary();
for (const NodeInfo& n : net.nodes())
  std::cout << n.index << " " << n.name << " " << n.type << " " << shape_str(n.output_shape) << "\n";
for (const NamedTensor& p : net.named_parameters())
  std::cout << p.name << " " << shape_str(p.tensor.shape()) << "\n";
```

Node and parameter names are explained under [node names](networks.md#node-names-in-evolved-networks).

Most hook functions take a `NodeFilter`, a predicate on `NodeInfo`. These are handy filters:

```cpp
NodeFilter lstms = [](const NodeInfo& n) { return n.type == "lstm"; };
NodeFilter blueprint_node_3 = [](const NodeInfo& n) { return n.origin.blueprint_node == 3; };
NodeFilter species_2 = [](const NodeInfo& n) { return n.origin.species == 2; };
NodeFilter heads = [](const NodeInfo& n) { return n.kind == NodeKind::Output; };
NodeFilter module_outputs = [](const NodeInfo& n) { return n.origin.role == "module_out"; };
```

## Activations

### Record them

```cpp
net.record_activations(true);
auto outs = net.forward(std::vector<Tensor>{x});
Tensor hidden = net.activation("body");
for (const auto& [name, value] : net.activations())
  std::cout << name << " " << value.to_string(4) << "\n";
```

Recording keeps each node's output from the **latest** `forward` only. For a history across time steps,
copy values from a hook:

```cpp
std::vector<std::vector<float>> history;
int id = net.add_forward_hook("body", [&](const NodeInfo&, Tensor& out) { history.push_back(out.data()); });
for (int t = 0; t < 5; ++t) net.forward(std::vector<Tensor>{x});
net.remove_hook(id);
```

### Monitor them

```cpp
int id = net.add_forward_hook([](const NodeInfo& n, Tensor& out) {
  double sum_sq = 0;
  for (float v : out.data()) sum_sq += double(v) * v;
  std::cout << n.name << " rms=" << std::sqrt(sum_sq / out.numel()) << "\n";
});
net.forward(std::vector<Tensor>{x});
net.remove_hook(id);
```

### Change them

Assign a new tensor to keep the change visible to autograd:

```cpp
// Scale a layer's output. Gradients flow through the multiplication.
net.add_forward_hook("body", [](const NodeInfo&, Tensor& out) { out = out * 0.5f; });
```

Or edit values directly, which autograd doesn't know about:

```cpp
// Overwrite one unit. Gradients are computed as if nothing happened.
net.add_forward_hook("body", [](const NodeInfo&, Tensor& out) {
  for (int b = 0; b < out.size(0); ++b) out.at({b, 0}) = 0.0f;
});
```

Use the first form unless you have a reason not to. In-place edits can also reach further than you
expect, because of tensor sharing:

- Identity nodes (`/in`, `/out`, `body`) without a merge pass their input tensor through, so
  they share storage with their parent.
- Backward functions read saved outputs. Editing a `tanh` output in place changes the gradient that
  `tanh` computes.
- Forward hooks also run for input nodes, and an in-place edit there changes *your* input tensor.

### Ablate a module

```cpp
Tensor base = net.forward(std::vector<Tensor>{x})[0];
net.reset_state(x.size(0));
int id = net.add_forward_hook(
    [](const NodeInfo&, Tensor& out) { out = out * 0.0f; },
    [](const NodeInfo& n) { return n.origin.role == "module_out" && n.origin.blueprint_node == 3; });
Tensor ablated = net.forward(std::vector<Tensor>{x})[0];
net.remove_hook(id);
std::cout << "effect: " << sum(square(base - ablated)).item() << "\n";
```

### Hook a layer's input

Pre-hooks see the merged input of a layer. They have the same signature and can replace it too:

```cpp
net.add_forward_pre_hook(
    [](const NodeInfo& n, Tensor& input) { std::cout << n.name << " input " << shape_str(input.shape()) << "\n"; },
    [](const NodeInfo& n) { return n.type == "lstm"; });
```

## Gradients

### Read activation gradients

With recording on, gradients are there after `backward()`:

```cpp
net.record_activations(true);
Tensor loss = sum(square(net.forward(std::vector<Tensor>{x})[0]));
loss.backward();
for (const auto& [name, value] : net.activations())
  if (value.has_grad()) std::cout << name << " " << value.grad().to_string(4) << "\n";
```

Inputs show a gradient only if you created the input tensor with `requires_grad = true`, which gives you
saliency maps:

```cpp
Tensor image = Tensor::randn(x.shape(), rng, 1.0f, /*requires_grad=*/true);
Tensor outs0 = net.forward(std::vector<Tensor>{image})[0];
sum(outs0).backward();
Tensor saliency = image.grad();
std::cout << saliency.to_string();
```

### Change gradients in flight

A backward hook runs when the gradient reaches a node. Editing it changes what flows further back.

```cpp
// Normalise the gradient through every LSTM output.
net.add_backward_hook(
    [](const NodeInfo&, Tensor& g) {
      float norm = std::sqrt(sum(square(g)).item());
      if (norm > 1.0f) g = g / norm;
    },
    [](const NodeInfo& n) { return n.type == "lstm"; });

// Block all gradient flow back through a node.
net.add_backward_hook("body", [](const NodeInfo&, Tensor& g) { g.fill(0.0f); });
```

Backward hooks attach to tensors during `forward`, so:

- They only apply to `forward` calls made after the hook was added.
- `remove_hook` doesn't detach them from tensors already produced.
- They don't fire for tensors created under `NoGradGuard`.
- They fire once per time step in a recurrent unroll, since each step produces a new tensor.

### Parameter gradients

```cpp
net.zero_grad();
sum(net.forward(std::vector<Tensor>{x})[0]).backward();
Tensor w = net.parameter("head:policy.weight");
std::cout << w.grad().to_string() << " total norm " << grad_norm(net.parameters()) << "\n";

// Per-parameter learning-rate multiplier, as a gradient hook.
int id = net.add_parameter_hook("head:policy.weight", [](Tensor& g) {
  for (auto& v : g.data()) v *= 0.1f;
});
net.remove_hook(id);
```

A parameter hook sees one backward pass's contribution, before it is added to `grad`. To change the
accumulated gradient instead, edit `grad_data()` directly between `backward()` and `opt.step()`:

```cpp
for (auto p : net.parameters())
  if (p.has_grad())
    for (auto& g : p.grad_data()) g = std::max(-0.5f, std::min(0.5f, g));
```

## Weights

`net.parameter(name)` returns a handle to the live tensor, so writing to it changes the network.

```cpp
Tensor w = net.parameter("head:policy.weight");
w.data()[0] = 0.0f;                   // one value
w.fill(0.0f);                         // all of them
w.at({1, 2}) = 0.5f;                  // by index

Rng noise(1);                         // add noise to every weight
for (auto p : net.parameters())
  for (auto& v : p.data()) v += static_cast<float>(noise.normal(0.0, 0.01));
```

### Snapshots

```cpp
std::vector<NamedTensor> snapshot = net.state_dict();  // detached copies
// ... experiment ...
net.load_state_dict(snapshot);                          // restore
```

### Freezing

Frozen parameters get no gradient, and optimizers skip them.

```cpp
net.set_trainable("bp2/", false);                                              // by name prefix
net.set_trainable([](const NodeInfo& n) { return n.type == "conv2d"; }, false); // by filter
net.set_trainable("", true);                                                   // everything back on
net.parameter("head:value.bias").set_requires_grad(false);                     // one tensor
```

The filter form includes a node's merge projections, and the prefix form includes any parameter
whose name starts with the prefix.

## Recurrent state

```cpp
for (const NamedTensor& s : net.get_state()) std::cout << s.name << " " << s.tensor.to_string(4) << "\n";

for (Layer* l : net.layers())
  if (auto* lstm = dynamic_cast<LSTMLayer*>(l)) {
    if (!lstm->last_gates().empty()) std::cout << lstm->last_gates().at("f").to_string(4);  // i, f, g, o
    std::cout << lstm->hidden().to_string(4) << lstm->cell().to_string(4) << lstm->num_cell_skips();
  }
```

Gates hold the last step's values and are empty before the first step. You can also branch on a state
to ask "what would happen if":

```cpp
auto saved = net.get_state();
Tensor option_a = net.forward(x);
net.set_state(saved);
Tensor option_b = net.forward(x * 2.0f);
net.set_state(saved);
std::cout << option_a.numel() + option_b.numel();
```

Or set a value directly. `get_state` returns handles, so to change a state, build a new tensor and
pass it in, rather than editing one that an autograd graph may depend on:

```cpp
auto state = net.get_state();
for (auto& s : state) s.tensor = s.tensor.detach() * 0.5f;
net.set_state(state);
```

## During A2C training

`A2CTrainer::on_update` runs after each backward pass and gradient clip, before the optimizer step, with
full access to the network:

```cpp
rl::A2CTrainer trainer(net, make_env, rl::A2CConfig{}, 1);
trainer.on_update = [](const rl::UpdateInfo& info, Network& n) {
  if (info.update % 100 == 0)
    std::cout << info.steps << " return " << info.recent_return << " grad " << info.grad_norm << " "
              << grad_norm(n.parameters()) << "\n";
};
```

## Optimizer buffers

```cpp
Adam adam(net.named_parameters(), 1e-3f);
sum(net.forward(std::vector<Tensor>{x})[0]).backward();
adam.step();
if (const ParamState* st = adam.state("head:policy.weight"))
  std::cout << st->m.size() << " first moments after " << st->steps << " steps\n";
```

## Removing hooks

Every `add_*` call returns an id:

```cpp
int id = net.add_forward_hook([](const NodeInfo&, Tensor&) {});
net.remove_hook(id);
net.clear_hooks();  // all of them
```

Don't add or remove network hooks from inside a forward hook, because `forward` is iterating over them.
