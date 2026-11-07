# Search spaces

A search space defines what evolution is allowed to choose. There are three in a
CoDeepNEAT run:

| Field | Describes | Stored in |
|---|---|---|
| `module_node_space` | One layer | Each module node |
| `blueprint_node_space` | How a module instance's inputs are merged | Each blueprint node |
| `global_space` | Network-wide settings such as the learning rate | Each blueprint |

## Parameter kinds

A `HyperparamSpace` is a list of `ParamSpec`s.

| Factory | Values | Mutation |
|---|---|---|
| `ParamSpec::real(name, lo, hi, log_scale=false)` | Float in [lo, hi] | Gaussian noise, clipped; in log space if `log_scale` |
| `ParamSpec::integer(name, lo, hi)` | Integer in [lo, hi] | Rounded Gaussian noise, and always moves by at least 1 |
| `ParamSpec::boolean(name)` | True or false | Flip |
| `ParamSpec::choice(name, {...})` | One of several strings | Redraw a different choice |

Two chainable modifiers refine a spec:

- `.only_for({"dense", "lstm"})` marks it as relevant only when the node's `type` is
  listed. It still exists on other nodes, but it doesn't count toward their distance.
- `.with_mutation_std(0.05)` sets the noise as a fraction of the range. The default is
  0.15.

Use `log_scale` for anything spanning orders of magnitude, like learning rates. Numeric
choices such as kernel sizes are written as strings (`{"1", "3"}`); reading them with
`hp.i("kernel_size")` parses the number.

Names and choice strings must not contain commas, colons or whitespace, because the
[genome file format](file-formats.md#hyperparameter-tables) uses those as separators.

## Layer hyperparameters

These are the keys the built-in layers read. Evolving them is optional: a key missing
from your space simply takes its default.

| Key | Used by | Default | Meaning |
|---|---|---|---|
| `type` | all | `dense` | Layer type: `dense`, `conv2d`, `lstm`, `identity`, or a [registered type](extending.md#custom-layers) |
| `merge` | all | `concat` | Merge for several inputs: `concat` or `sum` |
| `units` | dense, lstm | 32 | Width |
| `activation` | dense, conv2d | `relu` | See [activations](tensors.md#activations). LSTM layers ignore it |
| `dropout` | dense, conv2d, lstm | 0 | Dropout on the layer's output; must be below 1 |
| `weight_scale` | dense, conv2d, lstm | 1 | Multiplier on the initial weight scale |
| `use_bias` | dense, conv2d | true | Whether the layer has a bias |
| `filters` | conv2d | 32 | Output channels |
| `kernel_size` | conv2d | 3 | Square kernel size |
| `stride` | conv2d | 1 | Stride |
| `padding` | conv2d | `kernel_size / 2` | Zero padding |
| `max_pool` | conv2d | false | Add 2x2 max pooling, skipped if the map is smaller than 2x2 |

## Built-in layer spaces

All of these live in the `spaces` namespace.

### `dense_layers(min_units = 8, max_units = 128)`

| Key | Range |
|---|---|
| `type` | `dense` |
| `units` | [min_units, max_units] |
| `activation` | `relu`, `tanh`, `elu`, `leaky_relu`, `linear` |
| `dropout` | [0, 0.2] |
| `weight_scale` | [0.5, 2] |
| `merge` | `concat`, `sum` |

### `recurrent_layers(min_units = 8, max_units = 64)`

As above with `type` in `dense`, `lstm`, and `activation` in `relu`, `tanh`, `elu`,
`linear` (dense only). LSTM nodes enable the cell-skip mutations.

### `conv_layers(min_filters = 8, max_filters = 64)`

| Key | Range |
|---|---|
| `type` | `conv2d` |
| `filters` | [min_filters, max_filters] |
| `kernel_size` | `1`, `3` |
| `max_pool` | true, false |
| `activation` | `relu`, `elu`, `leaky_relu`, `tanh` |
| `dropout` | [0, 0.3] |
| `weight_scale` | [0.5, 2] |
| `merge` | `concat`, `sum` |

### `mixed_layers()`

`type` in `conv2d`, `dense`, `lstm`, with `units` in [8, 128] (dense, lstm), `filters` in
[8, 64], `kernel_size` and `max_pool` (conv2d), `activation` in `relu`, `elu`, `tanh`,
`linear` (conv2d, dense), `dropout` in [0, 0.3], `weight_scale` in [0.5, 2], and `merge`.

Once a dense or LSTM layer has flattened an image, a later `conv2d` sees a `[F]` input and
treats it as an `[F, 1, 1]` image. It then acts as a dense layer, so structure after a
flatten is not spatial.

### Spaces from the paper

| Function | Paper | Contents |
|---|---|---|
| `cifar10_nodes()` | Table 1 | `conv2d` with `filters` [32, 256], `dropout` [0, 0.7], `weight_scale` [0, 2], `kernel_size` {1, 3}, `max_pool` |
| `captioning_nodes()` | Table 2 | `type` {dense, lstm}, `merge`, `units` {128, 256}, `activation` {relu, linear} (dense), `dropout` [0, 0.7] |

`cifar10_nodes()` has no `activation` or `merge` key, so those layers use ReLU and
concatenation.

## Blueprint node space

`blueprint_nodes()` has one key: `merge` in `concat`, `sum`. It decides how a module
instance's inputs are combined when the blueprint gives it several.

## Global spaces

Global hyperparameters are handed to your evaluator as `ctx.globals`. **The library reads
only the keys below.** Any other key is evolved and passed along, and it's up to your
evaluator to act on it.

| Key | Read by | Meaning |
|---|---|---|
| `weight_init` | `default_build_options` | Initialisation scheme |
| `optimizer` | `A2CTrainer::apply_globals`, `make_optimizer_from_globals` | `sgd`, `adam` or `rmsprop` |
| `learning_rate` | same | Learning rate |
| `momentum`, `nesterov` | same | SGD only |
| `weight_decay` | `make_optimizer_from_globals` | L2 penalty |
| `entropy_coef`, `value_coef`, `max_grad_norm`, `gamma` | `A2CTrainer::apply_globals` | A2C settings |

### `rl_globals()`

| Key | Range |
|---|---|
| `learning_rate` | [1e-4, 5e-2], log scale |
| `optimizer` | `adam`, `rmsprop`, `sgd` |
| `momentum` | [0.68, 0.99] |
| `nesterov` | true, false |
| `entropy_coef` | [0, 0.05] |
| `value_coef` | [0.25, 1] |
| `max_grad_norm` | [0.25, 5], log scale |
| `weight_init` | `glorot_uniform`, `glorot_normal`, `he_normal` |

### `cifar10_globals()` and `captioning_globals()`

These reproduce the global rows of Tables 1 and 2. Only their optimizer-related keys and
`weight_init` are read by the library. The rest are for your evaluator to implement:

- CIFAR-10: `hue_shift`, `sv_shift`, `sv_scale`, `cropped_image_size`, `spatial_scaling`,
  `random_horizontal_flips` and `variance_normalization` describe data augmentation.
- Captioning: `shared_embedding_size`, `embedding_dropout` and `lstm_recurrent_dropout`.

## Writing your own space

Start from a built-in space and adjust it:

```cpp
HyperparamSpace s = spaces::recurrent_layers(16, 128);
s.find("dropout")->hi = 0.0;                                   // change a range
s.remove("weight_scale");                                      // stop evolving a key
s.add(ParamSpec::choice("activation", {"tanh"}).only_for({"dense"}));  // same name replaces
cfg.module_node_space = s;
```

Or build one from scratch. Your own keys are welcome, and your evaluator or custom layer
can read them:

```cpp
cfg.global_space = HyperparamSpace{
    ParamSpec::real("learning_rate", 1e-4, 1e-1, true),
    ParamSpec::integer("batch_size", 16, 128),
    ParamSpec::boolean("normalise_inputs"),
};

Evaluator evaluate = [](EvalContext& ctx) {
  int batch = ctx.globals.i("batch_size", 32);
  bool normalise = ctx.globals.b("normalise_inputs", false);
  return static_cast<double>(batch) * (normalise ? 1.0 : 0.5);  // placeholder score
};
```

## Reading values

`HyperparamValues` has typed getters that take a default for a missing key:

```cpp
HyperparamValues hp;
hp.set_int("units", 64).set_choice("activation", "tanh").set_float("dropout", 0.1).set_bool("max_pool", true);
int units = hp.i("units", 32);
double dropout = hp.f("dropout");
bool pool = hp.b("max_pool");
std::string act = hp.s("activation", "relu");
std::cout << units << dropout << pool << act << hp.to_string();
```

The getters convert where it makes sense. `i` and `f` parse numeric choice strings, and `s`
formats numbers.
