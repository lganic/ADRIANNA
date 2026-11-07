# Extending the library

The library is built to be extended without changing its source. This page covers the extension
points, from the most common to the least.

## Custom search spaces

Usually all you need. See [Search spaces](search-spaces.md#writing-your-own-space).

## Custom layers

A layer type needs a class and a registered factory. Once registered, naming it in a search space's
`type` choices is all it takes for evolution to use it.

This complete program adds a gated layer and evolves networks that can contain it:

```cpp
#include <iostream>

#include "codeepneat/all.hpp"

using namespace cdn;

// y = sigmoid(Wg x) * tanh(Wv x)
class GatedLayer : public Layer {
 public:
  GatedLayer(const std::string& name, const HyperparamValues& hp, const Shape& in_shape, const BuildOptions& opt,
             Rng& rng)
      : Layer(name, "gated", in_shape, hp) {
    int units = hp.i("units", 16);
    int in = static_cast<int>(shape_numel(in_shape));
    float scale = static_cast<float>(hp.f("weight_scale", 1.0));
    gate_ = add_param("gate", init_weight({units, in}, in, units, opt, scale, rng));
    value_ = add_param("value", init_weight({units, in}, in, units, opt, scale, rng));
    out_shape_ = {units};  // required: per-sample output shape
  }

  Tensor forward(const Tensor& x, LayerContext&) override {
    Tensor flat = flatten(x);  // x has a batch axis, and may be an image
    return sigmoid(linear(flat, gate_, Tensor())) * tanh(linear(flat, value_, Tensor()));
  }

 private:
  Tensor gate_, value_;
};

int main() {
  LayerRegistry::instance().add("gated", [](const std::string& name, const HyperparamValues& hp, const Shape& in,
                                            const BuildOptions& opt, Rng& rng) {
    return std::unique_ptr<Layer>(new GatedLayer(name, hp, in, opt, rng));
  });

  CoDeepNEATConfig cfg;
  cfg.io.inputs.push_back({"x", {4}});
  cfg.io.outputs.push_back({"y", 1});
  cfg.module_node_space = spaces::dense_layers(4, 16);
  cfg.module_node_space.add(ParamSpec::choice("type", {"dense", "gated"}));  // replaces the "type" spec
  cfg.modules.size = 8;
  cfg.blueprints.size = 4;
  cfg.networks_per_generation = 8;

  CoDeepNEAT evo(cfg);
  evo.run(2, [](EvalContext& ctx) { return static_cast<double>(ctx.net.num_parameters()); });
  std::cout << evo.build(evo.best())->summary();
  return 0;
}
```

The rules for a layer:

- Call the protected constructor with the node name, a type string, the input shape and the hyperparameters.
- Set `out_shape_` to the per-sample output shape.
- Create parameters with `add_param`, which registers them as `<node>.<local>` and enables gradients. They then work
  with `named_parameters`, optimizers, hooks, freezing and weight files automatically.
- In `forward`, `x` has a batch axis. Handle both flat and image inputs if your space allows them, as `flatten` does here.
- Use `ctx.training` and `*ctx.rng` for dropout-like behaviour.
- Register factories before evolution starts, because the registry isn't locked against concurrent changes.

Keep the output's rank at 1 or 3 per sample. Merges accept nothing else.

### Recurrent custom layers

Override `is_recurrent()` to return true, along with `reset_state`, `detach_state`, `mask_state`, `state` and `set_state`.
The network then calls them at the right times. Name state tensors `<layer>.<part>`, because `Network::set_state` matches
them by name.

Memory-cell skips are specific to `LSTMLayer`. Adding your type to `recurrent_layer_types` lets mutations create skip genes
for it, but `Network` ignores any skip that doesn't connect two `LSTMLayer`s.

### Export

Custom layers can't be exported to PyTorch; `export_torch` throws for them.

## Custom operations

If the built-in operations can't express something, write a tensor operation directly. Compute the output, and attach a
backward function if a gradient is needed:

```cpp
#include <iostream>
#include <memory>

#include "codeepneat/all.hpp"

using namespace cdn;

Tensor cube(const Tensor& x) {
  auto out = std::make_shared<TensorImpl>();
  out->shape = x.shape();
  out->op = "cube";
  out->data.resize(x.numel());
  for (size_t i = 0; i < x.numel(); ++i) out->data[i] = x.data()[i] * x.data()[i] * x.data()[i];

  if (GradMode::enabled() && x.requires_grad()) {
    TensorImpl* px = x.impl();  // raw pointer: `parents` keeps it alive
    out->requires_grad = true;
    out->parents = {x.impl_ptr()};
    out->backward_fn = [px](TensorImpl& self) {
      if (!px->requires_grad || px->grad.size() != px->data.size()) return;  // parent may have been frozen
      for (size_t i = 0; i < px->data.size(); ++i)
        px->grad[i] += self.grad[i] * 3.0f * px->data[i] * px->data[i];
    };
  }
  return Tensor(out);
}

int main() {
  Tensor x({2}, {2.0f, 3.0f}, true);
  sum(cube(x)).backward();
  std::cout << x.grad().to_string() << "\n";  // 12, 27
  return 0;
}
```

Three rules:

- Add to `grad`, never assign. A tensor can feed several operations.
- Capture parents as raw pointers, not `Tensor` or `shared_ptr`. The iterative destructor walks `parents` and can't see
  inside a closure, so a captured `shared_ptr` risks deep recursion when a long graph is freed.
- Check your derivative. The `gradcheck` helper in `tests/test_util.hpp` compares it with finite differences. See
  [Testing](testing.md#checking-gradients).

## Custom evolution loops

`CoDeepNEAT` and `DeepNEAT` are thin drivers. The pieces underneath are public.

### Driving a population directly

```cpp
PopulationParams params;
params.size = 20;
GenomeOperators ops(GenomeKind::Module, spaces::dense_layers(), HyperparamSpace(), params.mutation, params.speciation);
Population pop(params, ops, /*num_inputs=*/1);

Rng rng(1);
pop.initialize(rng);
for (int generation = 0; generation < 10; ++generation) {
  for (Genome& g : pop.genomes()) g.fitness = static_cast<double>(g.nodes.size());  // your fitness here
  pop.reproduce(generation, rng);
}
std::cout << pop.species().size() << " species\n";
```

### Applying mutations by hand

Each mutation is a public method of `GenomeOperators`, and they all preserve the genome's invariants:

```cpp
MutationParams mutation;
SpeciationParams speciation;
GenomeOperators ops(GenomeKind::Module, spaces::recurrent_layers(), HyperparamSpace(), mutation, speciation);
InnovationTracker innovations(1);
Rng rng(1);

Genome g = ops.create_initial(1, innovations, rng);
ops.mutate_add_node(g, innovations, rng, {});
ops.mutate_add_connection(g, innovations, rng);
ops.mutate_toggle_connection(g, rng);

std::string why;
if (!g.validate(&why)) std::cout << why;
std::cout << g.describe();
```

If you edit genes directly, call `validate` afterwards. To keep innovation numbers consistent, get them from the population's
`InnovationTracker` (`Population::innovations()`).

### Seeding known architectures

Populations are writable, so you can edit genomes after construction and before the first `step`:

```cpp
for (Genome& module : evo.modules().genomes())
  for (auto& node : module.nodes) node.second.hp.set_int("units", 32);
```

`module_node_space` still applies to later mutations, so the search can move away from your values.

### Assembling without evolving

`sample_assembly` builds an `AssembledGenome` from the current populations, which is useful for looking at what evolution
is working with:

```cpp
Rng rng(1);
AssembledGenome sample = evo.sample_assembly(0, rng);
std::cout << sample.describe() << evo.graph(sample).to_dot();
```

## Weight inheritance

The paper trains every network from scratch. If you'd like offspring to start from earlier weights, the evaluator can do that
itself, because `load_state_dict` matches by name and `strict = false` skips anything that doesn't fit. Names contain the blueprint
node and module ids, so only layers with the same origin receive weights.

```cpp
std::mutex mutex;
double best_score = -1e300;
std::vector<NamedTensor> best_weights;

Evaluator lamarckian = [&](EvalContext& ctx) {
  {
    std::lock_guard<std::mutex> lock(mutex);
    if (!best_weights.empty()) ctx.net.load_state_dict(best_weights, /*strict=*/false);
  }
  double score = 1.0;  // train ctx.net and measure it here
  std::lock_guard<std::mutex> lock(mutex);
  if (score > best_score) {
    best_score = score;
    best_weights = ctx.net.state_dict();
  }
  return score;
};
```

This shared state makes results depend on evaluation order, so runs with more than one thread stop being reproducible.

## Other environments and trainers

See [Reinforcement learning](reinforcement-learning.md#writing-your-own) for environments, and
[using your own RL algorithm](reinforcement-learning.md#using-your-own-rl-algorithm) for trainers.
