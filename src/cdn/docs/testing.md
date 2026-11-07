# Testing

## Running the tests

```bash
cmake --build build
```

```bash
ctest --test-dir build --output-on-failure
```

CTest runs these entries:

| Test | What it runs |
|---|---|
| `cdn_tests` | All C++ unit tests |
| `torch_export_generate` | `export_pytorch`: exports networks and reference runs |
| `torch_export_verify` | `tools/verify_torch_export.py`: replays them in PyTorch |
| `torch_export_import` | `export_pytorch --check-import`: loads PyTorch weights back |

The `torch_export_*` tests exist only if CMake found a Python that can `import torch`. The configure step prints whether they're
enabled. Choose the interpreter with `-DPython3_EXECUTABLE=...`, or turn them off with `-DCDN_TEST_PYTORCH=OFF`.

The unit tests can also run directly, and an argument filters them by substring:

```bash
build/cdn_tests
```

```bash
build/cdn_tests grad_
```

They write a few temporary files to the current directory and remove them.

## What is covered

| File | Covers |
|---|---|
| `tests/test_tensor.cpp` | Gradients of every operation against finite differences, broadcasting, hooks, `NoGradGuard`, very deep graphs |
| `tests/test_network.cpp` | Shapes, BPTT through cell skips, merges, hooks, freezing, state save and restore, weight files, invalid graphs, custom layers |
| `tests/test_genome.cpp` | Hyperparameter bounds, innovation alignment, mutation and crossover invariants, serialisation, assembly, speciation |
| `tests/test_evolution.cpp` | Evolving on a regression task, determinism across thread counts, fitness attribution, evaluator errors, DeepNEAT, environments, A2C learning |
| `tests/test_export.cpp` | safetensors round trips and dtypes, parameter mapping, weight import, rejecting unknown layers |

Some properties worth knowing about:

- `network_bptt_gradcheck_with_cell_skips` checks full-network gradients through three time steps, with forward and feedback skips.
- `module_mutation_invariants` and `crossover_invariants` apply thousands of random operations and validate every result.
- `codeepneat_deterministic_across_thread_counts` compares every fitness value between 1 and 3 threads.
- `rl_environments_and_a2c_smoke` requires A2C to actually learn `Catch`, so a sign error in the loss fails the test.

## Writing a test

Tests use a small harness in `tests/test_util.hpp`, with no external framework. A test registers itself:

```cpp
#include "test_util.hpp"

using namespace cdn;

TEST(my_new_behaviour) {
  Tensor t({2}, {1, 2});
  CHECK(t.numel() == 2);
  CHECK_NEAR(sum(t).item(), 3.0, 1e-6);
  CHECK_THROWS(t.item());
}
```

| Macro | Meaning |
|---|---|
| `TEST(name)` | Defines and registers a test |
| `CHECK(condition)` | Fails the test if false |
| `CHECK_NEAR(a, b, tolerance)` | Fails if the values differ by more than `tolerance` |
| `CHECK_THROWS(expression)` | Fails if nothing is thrown |

Add new test files to the `cdn_tests` target in `CMakeLists.txt`.

## Checking gradients

`gradcheck` compares analytic gradients with central finite differences. Pass a function that rebuilds the loss from scratch on each
call, and the tensors to check:

```cpp
TEST(my_op_gradient) {
  Rng rng(1);
  Tensor x = Tensor::randn({3, 4}, rng, 1.0f, /*requires_grad=*/true);
  gradcheck([&] { return sum(square(tanh(x))); }, {x});
}
```

Its parameters are `gradcheck(loss, params, tol = 2e-2, eps = 1e-2, max_checks_per_param = 24, label = "")`.

Everything is float32, so the tolerances are loose. A few tips:

- Use a smaller `eps`, such as `1e-3`, around kinks like ReLU or max pooling. A step across a kink gives a false failure.
- Keep losses modest in size. Rounding error grows with the loss value.
- To gradcheck dropout, re-seed the RNG inside the loss function so the mask stays the same.
- For recurrent networks, call `reset_state` inside the loss function.

## Verifying documentation snippets

The C++ blocks in `docs/` were compiled and run against this version of the library when they were written. If you change the public API,
search `docs/` for the names you touched.
