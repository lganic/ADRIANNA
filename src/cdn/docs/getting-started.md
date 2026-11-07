# Getting started

This guide takes you from a fresh checkout to a program that evolves a network.

## Requirements

- A C++17 compiler. The library has been tested with MinGW-w64 GCC 12.2.
- CMake 3.16 or newer. Ninja is optional.
- Optional: Python with PyTorch, to verify [PyTorch exports](pytorch-export.md).

There are no other dependencies. The library uses only the standard library and
threads.

## Build

```bash
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
```

```bash
cmake --build build
```

A Release build matters a lot here: the tensor engine is plain loops, and it relies
on the optimizer. CMake defaults to Release if you don't set a build type.

| CMake option | Default | Effect |
|---|---|---|
| `CDN_BUILD_EXAMPLES` | `ON` | Builds the example programs |
| `CDN_BUILD_TESTS` | `ON` | Builds `cdn_tests` and registers it with CTest |
| `CDN_TEST_PYTORCH` | `ON` | Adds the PyTorch round-trip tests, if Python can `import torch` |

If several Pythons are installed, choose the one that has torch with
`-DPython3_EXECUTABLE=/path/to/python`.

Check the build:

```bash
ctest --test-dir build --output-on-failure
```

## Run an example

```bash
build/cartpole_codeepneat
```

This evolves actor-critic networks for the CartPole task. Each line is one
generation, and the run stops early once a network scores the maximum of 500:

```text
CoDeepNEAT on CartPole: 10 generations, 24 networks/generation, 12 threads
gen 0 | best 479.000 mean 123.883 worst 9.600 | best-ever 479.000 | species m=1 b=2 | params 340 | 24 evals | 0.3s
gen 1 | best 319.800 mean 130.233 worst 9.200 | best-ever 479.000 | species m=3 b=3 | params 354 | 24 evals | 0.2s
gen 2 | best 500.000 mean 147.867 worst 9.000 | best-ever 500.000 | species m=3 b=4 | params 619 | 24 evals | 0.3s
```

Reading a line from left to right:

| Field | Meaning |
|---|---|
| `best`, `mean`, `worst` | Fitness of this generation's networks |
| `best-ever` | Highest fitness of any network so far |
| `species m= b=` | Number of module and blueprint species |
| `params` | Mean parameter count of the evaluated networks |
| `evals` | Networks evaluated, with a count of failures if there were any |
| last field | Wall-clock time of the generation |

`best` can fall between generations because every network is trained from scratch
with fresh random weights. The champion above retrains to a mean return of 296,
not 500, for the same reason. Treat a single fitness value as a noisy measurement;
[Reinforcement learning](reinforcement-learning.md#designing-the-fitness) covers
how to reduce that noise.

## Your first program

This program evolves a network for a small regression task. It shows the three
parts of every use of the library: describe the network's interface, write an
evaluator, and run evolution.

```cpp
#include <cmath>
#include <iostream>

#include "codeepneat/all.hpp"

using namespace cdn;

int main() {
  // Training data: y = sin(3x).
  const int n = 128;
  std::vector<float> xs(n), ys(n);
  Rng data_rng(1);
  for (int i = 0; i < n; ++i) {
    xs[i] = static_cast<float>(data_rng.uniform(-1, 1));
    ys[i] = std::sin(3 * xs[i]);
  }
  Tensor X({n, 1}, xs), Y({n, 1}, ys);

  // 1. The fixed interface: one input of size 1, one output head of size 1.
  CoDeepNEATConfig cfg;
  cfg.io.inputs.push_back({"x", {1}});
  cfg.io.outputs.push_back({"y", 1});

  // What may evolve: dense layers, and two global hyperparameters.
  cfg.module_node_space = spaces::dense_layers(4, 32);
  cfg.module_node_space.find("dropout")->hi = 0.0;  // no dropout for this task
  cfg.global_space = HyperparamSpace{ParamSpec::real("learning_rate", 1e-3, 5e-2, true),
                                     ParamSpec::choice("optimizer", {"adam", "rmsprop"})};

  // Small populations keep this example fast.
  cfg.modules.size = 12;
  cfg.blueprints.size = 6;
  cfg.networks_per_generation = 12;
  cfg.num_threads = 4;

  // 2. The evaluator trains one network and returns its fitness (higher is better).
  Evaluator evaluate = [&](EvalContext& ctx) {
    auto opt = make_optimizer_from_globals(ctx.globals, ctx.net.named_parameters());
    for (int step = 0; step < 100; ++step) {
      Tensor loss = mse_loss(ctx.net.forward(X), Y);
      opt->zero_grad();
      loss.backward();
      opt->step();
    }
    NoGradGuard no_grad;
    return -static_cast<double>(mse_loss(ctx.net.forward(X), Y).item());
  };

  // 3. Run evolution, printing a line per generation.
  CoDeepNEAT evo(cfg);
  evo.run(5, evaluate, [](const GenerationStats& s) {
    std::cout << s.to_string() << "\n";
    return true;  // return false to stop early
  });

  // The result is an architecture. Rebuild it and inspect it.
  std::cout << evo.best().describe() << "\n";
  std::cout << evo.build(evo.best())->summary();
  return 0;
}
```

To compile it outside this project's CMake, add `include/` to the include path and
link `libcodeepneat.a` and your platform's threads library. From your own CMake
project, the simplest route is:

```cmake
add_subdirectory(path/to/codeepneat)
target_link_libraries(my_app PRIVATE codeepneat)
```

## What happened

1. `CoDeepNEAT evo(cfg)` created 12 minimal modules and 6 minimal blueprints.
2. Each generation assembled 12 networks from them, built each one with random
   weights, and called your evaluator (on 4 threads here).
3. Fitness went back to the modules and blueprints involved, and both populations
   bred a new generation.
4. `evo.best()` kept the genotype of the highest-scoring network seen.

Note that `evo.best()` is an **architecture and its hyperparameters, not trained
weights**. The trained network was destroyed after evaluation. `evo.build()` gives
you the same architecture with new random weights, ready for a longer final
training run.

## Next steps

- [Concepts](concepts.md) explains the algorithm you just ran.
- [Reinforcement learning](reinforcement-learning.md) replaces the regression task
  with an environment and A2C.
- [Examples](examples.md) lists complete programs for dense, recurrent and
  convolutional networks.
