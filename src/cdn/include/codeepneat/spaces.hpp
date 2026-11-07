#pragma once
// Ready-made hyperparameter search spaces.
//
// Node spaces describe a *layer* (module nodes in CoDeepNEAT, all nodes in
// DeepNEAT). Recognised keys used by the built-in layers:
//   type          dense | conv2d | lstm | identity | <user-registered type>
//   units         dense / lstm width
//   filters       conv2d output channels
//   kernel_size   conv2d kernel (square)
//   stride        conv2d stride (default 1)
//   max_pool      conv2d: append 2x2 max pooling
//   activation    see cdn::activation_names()
//   dropout       dropout rate on the layer output
//   weight_scale  multiplier of the initialiser's std ("initial weight scaling")
//   merge         how multiple incoming layers are merged: concat | sum
//
//   padding       conv2d padding (default kernel_size / 2)
//   use_bias      dense / conv2d bias (default true)
//
// Defaults for missing keys: units 32, filters 32, kernel_size 3, activation relu,
// dropout 0, weight_scale 1, merge concat. LSTM layers ignore "activation".
//
// Global spaces describe network-wide hyperparameters, stored on the blueprint
// (CoDeepNEAT) or on the genome (DeepNEAT) and handed to the user's evaluator.
// The library itself reads only those listed under rl_globals(); the rest (such
// as the data-augmentation settings of Table 1) are values for your evaluator to
// apply. docs/search-spaces.md lists every space.

#include "codeepneat/hyperparams.hpp"

namespace cdn {
namespace spaces {

/// Fully connected layers only.
HyperparamSpace dense_layers(int min_units = 8, int max_units = 128);
/// Dense or LSTM layers (LSTM nodes enable the LSTM-specific cell-skip mutations).
HyperparamSpace recurrent_layers(int min_units = 8, int max_units = 64);
/// Convolutional layers (in the spirit of Table 1, scaled for small inputs).
HyperparamSpace conv_layers(int min_filters = 8, int max_filters = 64);
/// conv2d, dense and lstm nodes mixed in one space.
HyperparamSpace mixed_layers();

/// Table 1 of the paper (CIFAR-10 node hyperparameters).
HyperparamSpace cifar10_nodes();
/// Table 2 of the paper (image-captioning node hyperparameters).
HyperparamSpace captioning_nodes();

/// Blueprint node genes: how the inputs of a module instance are merged.
HyperparamSpace blueprint_nodes();

/// Global hyperparameters for gradient-based RL training (consumed by
/// cdn::rl::A2CTrainer::apply_globals and make_optimizer_from_globals).
HyperparamSpace rl_globals();
/// Table 1 global hyperparameters (CIFAR-10).
HyperparamSpace cifar10_globals();
/// Table 2 global hyperparameters (image captioning).
HyperparamSpace captioning_globals();

}  // namespace spaces
}  // namespace cdn
