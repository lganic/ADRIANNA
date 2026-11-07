# Tensors and autograd

The library includes a small reverse-mode automatic differentiation engine. It was written
for this project instead of depending on a framework, so that every value and gradient is
ordinary, directly accessible memory.

## Tensor basics

A `Tensor` is a reference-counted handle to float32 data in row-major order. Images use NCHW
layout. Copying a `Tensor` copies the handle, so both copies share data, gradient and hooks.

```cpp
Tensor a({2, 3}, {1, 2, 3, 4, 5, 6});
Tensor z = Tensor::zeros({4});
Tensor n = Tensor::randn({2, 2}, rng, /*stddev=*/0.1f);
Tensor s = Tensor::scalar(2.0f);

float v = a.at({1, 2});   // 6, bounds-checked
a.data()[0] = 9.0f;       // raw std::vector<float>
Tensor same = a;          // same storage
same.data()[1] = 7.0f;    // also changes a
Tensor copy = a.detach(); // separate storage, no graph
std::cout << v << a.to_string() << z.numel() << n.dim() << s.item() << copy.size(1);
```

An empty `Shape` means a scalar with one element, which is what `sum(x)` returns.

## Gradients

```cpp
Tensor w({3}, {1, 2, 3}, /*requires_grad=*/true);
Tensor h = w * 2.0f;
Tensor loss = sum(square(h));
loss.backward();

std::cout << w.grad().to_string();  // dloss/dw = 8w
std::cout << h.grad().to_string();  // dloss/dh = 2h, kept for intermediates too
```

The rules:

- `backward()` needs a scalar. For anything else, pass a seed: `y.backward(seed)`.
- **Leaf gradients accumulate** across `backward()` calls, until `zero_grad()`.
- **Intermediate gradients are recomputed** by each pass. They stay readable as long as you hold
  the tensor. This is how activation gradients are inspected, and it is a difference from PyTorch,
  where you'd need `retain_grad()`.
- **The graph survives `backward()`**, so you can call it again without a `retain_graph` flag. The
  graph is freed when the last handle to its output is released.
- Autograd reads saved values during `backward()`. If you edit a tensor's data between forward and
  backward, for example with an optimizer step, the gradients use the edited values.

`zero_grad()` clears the gradient rather than filling it with zeros. `has_grad()` is then false,
and optimizers skip that tensor.

### Turning recording off

```cpp
{
  NoGradGuard no_grad;  // this thread only, until the end of the scope
  Tensor y = w * 3.0f;
  std::cout << y.requires_grad();  // 0
}
```

### Copying

| Call | Values | Graph |
|---|---|---|
| `Tensor b = a` | Shared | Shared |
| `a.clone()` | Copied | Connected (gradients flow back to `a`) |
| `a.detach()` | Copied | Disconnected leaf |

### Operations that return their input

There are no views: reshapes and slices copy. Three operations return the *same tensor* when
they have nothing to do, which matters if you then edit `.data()` in place:

- `activation(x, "linear")`
- `dropout(x, ...)` in eval mode or with p = 0
- `reshape` or `flatten` to the shape `x` already has

`concat` of one tensor and `adaptive_max_pool2d` to the same size also return the input.

## Gradient hooks

A hook runs during `backward()` with the gradient arriving at a tensor. It can read it, edit it, or
replace it with a tensor of the same size.

```cpp
Tensor hidden = w * 2.0f;
int id = hidden.register_hook([](Tensor& g) {
  for (auto& v : g.data()) v = std::min(v, 1.0f);  // clip what flows back
});
sum(hidden).backward();
hidden.remove_hook(id);
```

On an intermediate tensor, the hook runs before the gradient continues to its inputs, so edits
affect everything upstream. On a leaf, it sees only the current pass's contribution, before it is
added to the accumulated gradient.

## Operations

All of these are differentiable unless stated otherwise.

### Element-wise

| Operation | Notes |
|---|---|
| `a + b`, `a - b`, `a * b`, `a / b` | Also `add`, `sub`, `mul`, `div`; numpy-style broadcasting; scalars allowed on either side |
| `maximum(a, b)`, `minimum(a, b)` | Broadcasting |
| `-a`, `neg`, `exp`, `log`, `sqrt`, `abs`, `square` | |
| `pow(x, p)`, `clamp(x, lo, hi)` | |

### Activations

Call them directly, or by name with `activation(x, name)`:

| Name | Function |
|---|---|
| `linear` (also `identity`) | No change |
| `relu` | `max(0, x)` |
| `leaky_relu` | Slope 0.01 below zero |
| `elu` | Alpha 1 |
| `tanh` | |
| `sigmoid` | |
| `softplus` | `log(1 + exp(x))` |

### Linear algebra

| Operation | Shapes |
|---|---|
| `matmul(a, b)` | `[M,K] x [K,N]` |
| `linear(x, w, b)` | `x[B,I]`, `w[O,I]`, `b[O]` or an undefined tensor. Same layout as PyTorch |

### Reductions

| Operation | Notes |
|---|---|
| `sum(x)`, `mean(x)` | All elements, returns a scalar |
| `sum(x, axis, keepdim)`, `mean(x, axis, keepdim)` | Negative axes count from the end |
| `max(x, axis, keepdim)` | Gradient goes to the first maximum |

### Shape

| Operation | Notes |
|---|---|
| `reshape(x, shape)` | One dimension may be -1 |
| `flatten(x)` | Keeps axis 0 |
| `concat(xs, axis)` | Other dimensions must match |
| `stack(xs)` | New axis 0 |
| `narrow(x, axis, start, length)` | Slice |

### Probability and losses

| Operation | Notes |
|---|---|
| `softmax(x)`, `log_softmax(x)` | Last axis; numerically stable |
| `gather(x, index)` | `x[B,N]` to `[B]`, picking `x[b, index[b]]` |
| `mse_loss(pred, target)` | Mean squared error |
| `huber_loss(pred, target, delta)` | Mean Huber loss |
| `cross_entropy(logits, targets)` | From logits and class indices |

### Convolution and regularisation

| Operation | Notes |
|---|---|
| `conv2d(x, w, b, stride, padding)` | `x[B,C,H,W]`, `w[O,C,KH,KW]`, `b[O]` or undefined |
| `max_pool2d(x, kernel, stride)` | |
| `adaptive_max_pool2d(x, h, w)` | Same bins as PyTorch; cannot upsample |
| `dropout(x, p, rng, training)` | Inverted: survivors are scaled by `1 / (1 - p)` |

### Non-differentiable helpers

| Function | Returns |
|---|---|
| `argmax_rows(x)` | Per-row argmax of `x[B,N]` |
| `one_hot(index, n)` | `[len, n]` tensor |
| `grad_norm(params)` | Global L2 norm of the gradients |
| `clip_grad_norm(params, max_norm)` | Scales gradients in place, returns the norm before clipping |

## Optimizers

`SGD`, `Adam` and `RMSProp` take named parameters and update the tensors in place. They skip frozen
parameters and parameters without a gradient.

| Class | Constructor | Notes |
|---|---|---|
| `SGD` | `(params, lr, momentum = 0, nesterov = false)` | Same convention as `torch.optim.SGD` |
| `Adam` | `(params, lr = 1e-3, beta1 = 0.9, beta2 = 0.999, eps = 1e-8)` | Bias-corrected |
| `RMSProp` | `(params, lr = 1e-3, alpha = 0.99, eps = 1e-5, momentum = 0)` | |

Two factories exist. In both, momentum and Nesterov apply only to SGD:

```cpp
auto a = make_optimizer("sgd", net.named_parameters(), 0.01f, 0.9f, /*nesterov=*/true);

HyperparamValues globals;
globals.set_choice("optimizer", "adam").set_float("learning_rate", 3e-4);
auto b = make_optimizer_from_globals(globals, net.named_parameters());
b->set_learning_rate(1e-4f);        // schedules are up to you
b->set_weight_decay(1e-5f);         // L2 added to the gradient, not decoupled
std::cout << a->type() << b->learning_rate();
```

Buffers are readable through `opt.state(index)` or `opt.state("param.name")`, which return a
`ParamState` holding `m`, `v` and `steps`. Which buffers are filled depends on the optimizer:

| Optimizer | `m` | `v` |
|---|---|---|
| `SGD` | Velocity (if momentum > 0) | Unused |
| `Adam` | First moment | Second moment |
| `RMSProp` | Momentum (if > 0) | Square average |

## Threads

Autograd has no global graph, so independent networks can be trained on different threads. The only
thread-sensitive state is the gradient-recording flag, which is thread-local: a `NoGradGuard` affects
its own thread only, and new threads start with recording enabled. Don't use one tensor or network
from two threads at once.

## Performance

Each operation allocates its result, and matrix multiplication is a plain triple loop that the
compiler vectorises. This suits the small and medium networks common in RL, where the throughput comes
from running many evaluations in parallel. It is not meant to compete with BLAS or GPU libraries on
large models. Convolution uses im2col and recomputes the columns during backward to save memory.

Long recurrent graphs are safe: both `backward()` and the destructor are iterative, so a chain hundreds
of thousands of operations deep doesn't overflow the stack. A test covers a chain of 200,000.
