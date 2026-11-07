# Documentation

This library evolves neural network architectures with CoDeepNEAT (Miikkulainen
et al., [arXiv:1703.00548](https://arxiv.org/abs/1703.00548)) and trains them with
gradient descent, typically as a reinforcement-learning inner loop. It is plain
C++17 with its own autograd engine, so every activation, gradient and weight is
open to inspection.

## Where to start

| If you want to... | Read |
|---|---|
| Build it and evolve your first network | [Getting started](getting-started.md) |
| Understand what the algorithm does | [Concepts](concepts.md) |
| Evolve networks for your RL problem | [Reinforcement learning](reinforcement-learning.md) |
| Decide what evolution may search over | [Search spaces](search-spaces.md) |
| Tune population sizes and mutation rates | [Configuration](configuration.md) |
| Look inside a network, or change it | [Inspection and control](inspection.md) |
| Move a model to PyTorch | [PyTorch export](pytorch-export.md) |
| Fix something that isn't working | [Troubleshooting](troubleshooting.md) |

## All documents

**Guides**

- [Getting started](getting-started.md): build, run, and write a first program.
- [Concepts](concepts.md): NEAT, DeepNEAT and CoDeepNEAT, one generation step by
  step, and how this code differs from the paper.
- [Reinforcement learning](reinforcement-learning.md): environments, the A2C
  trainer, recurrent policies, fitness design, and plugging in your own algorithm.
- [Inspection and control](inspection.md): a cookbook for activations, gradients,
  weights, recurrent state and optimizer buffers.
- [PyTorch export](pytorch-export.md): what is exported, how to use it, and how to
  bring weights back.
- [Extending the library](extending.md): custom layers, operations, search spaces
  and evolution loops.
- [Examples](examples.md): what each example program shows.

**Reference**

- [Configuration](configuration.md): every evolution setting and its default.
- [Search spaces](search-spaces.md): hyperparameter tables and the built-in spaces.
- [Networks](networks.md): graphs, layers, merges, naming, and the `Network` class.
- [Tensors and autograd](tensors.md): the tensor engine and its operations.
- [File formats](file-formats.md): genome, weight, safetensors and export formats.

**Project**

- [Architecture](architecture.md): how the code is organised and why.
- [Testing](testing.md): what the tests cover and how to add to them.
- [Troubleshooting](troubleshooting.md): common problems and pitfalls.

The public headers in `include/codeepneat/` carry API comments for every class and
function. They use `///`, so Doxygen can process them if you want generated pages.
Include `codeepneat/all.hpp` for everything, and use the `cdn` namespace.
