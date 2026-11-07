// Built-in search spaces, including Tables 1 and 2 of the paper.

#include "codeepneat/spaces.hpp"

namespace cdn {
namespace spaces {

HyperparamSpace dense_layers(int min_units, int max_units) {
  HyperparamSpace s;
  s.add(ParamSpec::choice("type", {"dense"}));
  s.add(ParamSpec::integer("units", min_units, max_units));
  s.add(ParamSpec::choice("activation", {"relu", "tanh", "elu", "leaky_relu", "linear"}));
  s.add(ParamSpec::real("dropout", 0.0, 0.2));
  s.add(ParamSpec::real("weight_scale", 0.5, 2.0));
  s.add(ParamSpec::choice("merge", {"concat", "sum"}));
  return s;
}

HyperparamSpace recurrent_layers(int min_units, int max_units) {
  HyperparamSpace s;
  s.add(ParamSpec::choice("type", {"dense", "lstm"}));
  s.add(ParamSpec::integer("units", min_units, max_units));
  s.add(ParamSpec::choice("activation", {"relu", "tanh", "elu", "linear"}).only_for({"dense"}));
  s.add(ParamSpec::real("dropout", 0.0, 0.2));
  s.add(ParamSpec::real("weight_scale", 0.5, 2.0));
  s.add(ParamSpec::choice("merge", {"concat", "sum"}));
  return s;
}

HyperparamSpace conv_layers(int min_filters, int max_filters) {
  HyperparamSpace s;
  s.add(ParamSpec::choice("type", {"conv2d"}));
  s.add(ParamSpec::integer("filters", min_filters, max_filters));
  s.add(ParamSpec::choice("kernel_size", {"1", "3"}));
  s.add(ParamSpec::boolean("max_pool"));
  s.add(ParamSpec::choice("activation", {"relu", "elu", "leaky_relu", "tanh"}));
  s.add(ParamSpec::real("dropout", 0.0, 0.3));
  s.add(ParamSpec::real("weight_scale", 0.5, 2.0));
  s.add(ParamSpec::choice("merge", {"concat", "sum"}));
  return s;
}

HyperparamSpace mixed_layers() {
  HyperparamSpace s;
  s.add(ParamSpec::choice("type", {"conv2d", "dense", "lstm"}));
  s.add(ParamSpec::integer("units", 8, 128).only_for({"dense", "lstm"}));
  s.add(ParamSpec::integer("filters", 8, 64).only_for({"conv2d"}));
  s.add(ParamSpec::choice("kernel_size", {"1", "3"}).only_for({"conv2d"}));
  s.add(ParamSpec::boolean("max_pool").only_for({"conv2d"}));
  s.add(ParamSpec::choice("activation", {"relu", "elu", "tanh", "linear"}).only_for({"conv2d", "dense"}));
  s.add(ParamSpec::real("dropout", 0.0, 0.3));
  s.add(ParamSpec::real("weight_scale", 0.5, 2.0));
  s.add(ParamSpec::choice("merge", {"concat", "sum"}));
  return s;
}

HyperparamSpace cifar10_nodes() {
  HyperparamSpace s;
  s.add(ParamSpec::choice("type", {"conv2d"}));
  s.add(ParamSpec::integer("filters", 32, 256));
  s.add(ParamSpec::real("dropout", 0.0, 0.7));
  s.add(ParamSpec::real("weight_scale", 0.0, 2.0));
  s.add(ParamSpec::choice("kernel_size", {"1", "3"}));
  s.add(ParamSpec::boolean("max_pool"));
  return s;
}

HyperparamSpace captioning_nodes() {
  HyperparamSpace s;
  s.add(ParamSpec::choice("type", {"dense", "lstm"}));
  s.add(ParamSpec::choice("merge", {"sum", "concat"}));
  s.add(ParamSpec::choice("units", {"128", "256"}));
  s.add(ParamSpec::choice("activation", {"relu", "linear"}).only_for({"dense"}));
  s.add(ParamSpec::real("dropout", 0.0, 0.7));
  return s;
}

HyperparamSpace blueprint_nodes() {
  HyperparamSpace s;
  s.add(ParamSpec::choice("merge", {"concat", "sum"}));
  return s;
}

HyperparamSpace rl_globals() {
  HyperparamSpace s;
  s.add(ParamSpec::real("learning_rate", 1e-4, 5e-2, true));
  s.add(ParamSpec::choice("optimizer", {"adam", "rmsprop", "sgd"}));
  s.add(ParamSpec::real("momentum", 0.68, 0.99));
  s.add(ParamSpec::boolean("nesterov"));
  s.add(ParamSpec::real("entropy_coef", 0.0, 0.05));
  s.add(ParamSpec::real("value_coef", 0.25, 1.0));
  s.add(ParamSpec::real("max_grad_norm", 0.25, 5.0, true));
  s.add(ParamSpec::choice("weight_init", {"glorot_uniform", "glorot_normal", "he_normal"}));
  return s;
}

HyperparamSpace cifar10_globals() {
  HyperparamSpace s;
  s.add(ParamSpec::real("learning_rate", 0.0001, 0.1, true));
  s.add(ParamSpec::real("momentum", 0.68, 0.99));
  s.add(ParamSpec::real("hue_shift", 0.0, 45.0));
  s.add(ParamSpec::real("sv_shift", 0.0, 0.5));
  s.add(ParamSpec::real("sv_scale", 0.0, 0.5));
  s.add(ParamSpec::integer("cropped_image_size", 26, 32));
  s.add(ParamSpec::real("spatial_scaling", 0.0, 0.3));
  s.add(ParamSpec::boolean("random_horizontal_flips"));
  s.add(ParamSpec::boolean("variance_normalization"));
  s.add(ParamSpec::boolean("nesterov"));
  return s;
}

HyperparamSpace captioning_globals() {
  HyperparamSpace s;
  s.add(ParamSpec::real("learning_rate", 0.0001, 0.1, true));
  s.add(ParamSpec::real("momentum", 0.68, 0.99));
  s.add(ParamSpec::integer("shared_embedding_size", 128, 512));
  s.add(ParamSpec::real("embedding_dropout", 0.0, 0.7));
  s.add(ParamSpec::boolean("lstm_recurrent_dropout"));
  s.add(ParamSpec::boolean("nesterov"));
  s.add(ParamSpec::choice("weight_init", {"glorot_normal", "he_normal"}));
  return s;
}

}  // namespace spaces
}  // namespace cdn
