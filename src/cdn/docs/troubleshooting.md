# Troubleshooting

## Evolution

**Fitness doesn't improve.**
Check the inner loop first. Build one network by hand and train it with your evaluator's settings. If that doesn't learn, evolution
can't help. Then check that the training budget is long enough for a good network to stand out, and that the `learning_rate` range
includes values that work. On `Catch`, for example, a dense network needed around 1e-2 with Adam, and 3e-3 barely moved.

**`best` jumps around between generations.**
This is expected. Every network trains from scratch with new random weights, so fitness is a noisy measurement. Watch `mean` and
`best-ever`. To reduce noise, evaluate more episodes and raise `networks_per_generation`.

**The champion scores lower when retrained.**
Also expected: `evo.best()` is the luckiest high score, and a retrained network has different random weights. In one CartPole run a
champion that scored 500 retrained to a mean of 296. Retrain a few top candidates with several seeds before picking one.

**`species m=1` for the first generations.**
All genomes start with the same structure and differ only in hyperparameters, so they begin in one species. The threshold falls each
generation until species appear. To separate them sooner, lower `initial_threshold` or set `initial_structural_mutations`.

**More species than `target_species`.**
The target is soft. The threshold only limits how readily new species form, and existing species don't merge, so a species disappears
only through stagnation culling. Raise `initial_threshold` or `threshold_step` so fewer species form, or lower `stagnation_limit` so weak
ones go sooner. Blueprint counts can run higher because pointers and globals also add distance.

**Networks keep growing.**
Reduce `add_node` and `add_connection`, lower `max_nodes`, or subtract a size penalty from fitness. See
[tuning advice](configuration.md#tuning-advice).

**`GenerationStats::failures` is above zero.**
Look at `records[i].error` for the message:

```cpp
GenerationStats st = evo.step([](EvalContext& ctx) { return static_cast<double>(ctx.net.num_parameters()); });
for (const EvaluationRecord& r : st.records)
  if (!r.error.empty()) std::cout << r.index << ": " << r.error << "\n";
```

A non-finite fitness shows up as `non-finite fitness`, and usually means training diverged. Consider a smaller learning-rate range or
gradient clipping.

**Every fitness looks like -1000000000.**
That's the default `failed_fitness`, so every evaluation is failing. Print the errors as above. The usual cause is a mismatch
between `cfg.io` and what your evaluator feeds the network.

**A run isn't reproducible.**
Draw all randomness from `ctx.rng`. Don't share mutable state between evaluations. Results can legitimately differ between compilers,
because `std::` distributions are implementation-defined.

**Can I resume a run?**
No. Only single genotypes can be saved, not whole populations. Save `evo.best()` periodically from your generation callback.

## Networks and tensors

**`net.forward({x})` only returns one head.**
With a single tensor in braces, C++ picks the `Tensor` overload, which returns the first output. Write
`net.forward(std::vector<Tensor>{x})`. With two tensors, `net.forward({a, b})` doesn't compile at all, for the same reason.

**"backward(): implicit seed requires a scalar tensor".**
Reduce to a scalar first, with `sum` or `mean`, or pass a seed: `y.backward(Tensor::ones(y.shape()))`.

**"backward(): tensor does not require grad".**
No parameter influenced this tensor. Typical causes are running under `NoGradGuard`, freezing everything, or calling `detach()` on the
way.

**Gradients are twice as large as they should be.**
Leaf gradients accumulate. Call `zero_grad()` before each `backward()`.

**A parameter has no gradient.**
It's frozen, or nothing it feeds reaches the loss. A head you didn't use in the loss gets no gradient, and neither does its bias.

**Memory grows during a long episode.**
LSTM state holds the graph of every earlier step. Call `detach_state()` regularly, or run under `NoGradGuard` when you don't need
gradients. Also release loss tensors and turn off `record_activations` when you aren't using them.

**Acting in an environment is slow.**
Wrap it in `NoGradGuard` and use `net.eval()`. Batch environments together if you can, since per-call overhead dominates for small
networks. Also check that you're running a Release build.

**An in-place edit in a hook changed something unexpected.**
Identity nodes, linear activations and eval-mode dropout pass the same tensor through, so editing one edits the other. Assign a new
tensor instead (`out = out * mask`). See [Change them](inspection.md#change-them).

**"input ... has shape ..., expected [B]+...".**
Inputs are `[batch, ...shape]`. A vector observation of size 4 should be `[B, 4]`, and an image `[B, C, H, W]`. `rl::batch_observations`
builds this from raw observations.

**"merge supports [F] or [C,H,W] inputs".**
A custom layer has produced an output of another rank. Reshape it to one of those two.

**Conv layers do nothing useful.**
Once a dense or LSTM layer flattens the data, later conv layers see `[F, 1, 1]` images, so a 3x3 kernel only uses its centre. If that
happens often, restrict the module space to `conv_layers()`, and let the heads do the flattening.

**Everything shrinks through deep LSTM stacks.**
Stacked LSTMs with small initial weights can pass very small signals, as `inspection_demo` shows. Evolution selects against this if
your fitness does. A `weight_scale` range above 1 or a higher `lstm_forget_bias` also helps.

## Reinforcement learning

**"A2CTrainer: network needs 'policy' and 'value' heads".**
Use `rl::actor_critic_io`, or set `policy_head` and `value_head` in `A2CConfig`. Head names are matched with or without the `head:` prefix.

**"A2CTrainer: network must have one input".**
The built-in trainer supports a single input. Use your own loop for several, as described in
[Reinforcement learning](reinforcement-learning.md#several-inputs).

**The policy stays uniform.**
Entropy stays near `log(num_actions)` and nothing is learned. Raise the learning rate, lower `entropy_coef`, and check that rewards reach
the agent. `on_update` shows entropy and gradient norm during training.

**A recurrent policy doesn't learn a long dependency.**
Gradients stop at rollout boundaries. Set `rollout_length` longer than the dependency. Short-delay versions of a task are a good first check.

**My environment crashes with several threads.**
Evaluators run concurrently. Make sure each `EnvFactory` call creates a separate instance, with no shared mutable globals.

## PyTorch export

**"has no PyTorch equivalent in the exporter".**
The network contains a custom layer. Only `dense`, `conv2d`, `lstm` and `identity` can be exported.

**`load_state_dict` reports missing or unexpected keys.**
The `.py` and `.safetensors` files come from different networks. Export them together, and use `import_torch_weights` only on a network with
the same architecture.

**A PyTorch forward hook breaks the model.**
A hook that returns a value *replaces* the module's output. Make sure yours returns `None`. A lambda like
`lambda m, i, o: seen.setdefault("x", o.shape)` returns something and will break the next layer.

**The export tests don't appear in CTest.**
CMake didn't find a Python with torch. Run configure with `-DPython3_EXECUTABLE=/path/to/python` and look for the line
`PyTorch export tests enabled`.

**The exported model gives different outputs from C++.**
Both must be in eval mode, or dropout makes them differ. Recurrent models must start from the same state: call `reset_state` in both. If they
still differ, record a reference run and use `tools/verify_torch_export.py` to find out where. See
[Verifying an export](pytorch-export.md#verifying-an-export).

## Build

**Slow execution.**
Check that you built with `-DCMAKE_BUILD_TYPE=Release`. A Debug build of the tensor engine is far slower.

**Linker errors about threads.**
Link `Threads::Threads`. This is automatic through the `codeepneat` CMake target.

**MinGW: "undefined reference" to `std::thread`.**
Use a MinGW build with POSIX threads, not win32 threads.
