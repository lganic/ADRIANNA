#pragma once
// Export of built networks to PyTorch.
//
// export_torch(net, "out/policy") writes
//   out/policy.py           self-contained PyTorch source (needs only torch):
//                           an nn.Module reproducing the network exactly,
//                           including merges, conv/pool, stateful LSTMs and
//                           memory-cell skip connections; one submodule per node
//                           (model.nodes["bp3_m17_n5"]) so PyTorch hooks work.
//   out/policy.safetensors  weights keyed by the module's state_dict() names
//                           (standard safetensors format; the generated module
//                           can read it without the `safetensors` package).
//
// In Python:
//   from policy import EvolvedNet
//   model = EvolvedNet.from_pretrained("out/policy.safetensors")
//   logits, value = model(obs)                    # one time step, like Network::forward
//
// Weights trained further in PyTorch can be brought back with
// import_torch_weights().

#include <map>
#include <string>
#include <vector>

#include "codeepneat/network.hpp"

namespace cdn {

struct TorchExportOptions {
  std::string class_name = "EvolvedNet";
  /// Extra text for the generated module docstring (e.g. task / fitness).
  std::string description;
};

/// Writes <path_prefix>.py and <path_prefix>.safetensors. The directory must exist.
void export_torch(const Network& net, const std::string& path_prefix, const TorchExportOptions& options = {});

/// The generated Python source (architecture only; weights are separate).
std::string torch_module_source(const Network& net, const TorchExportOptions& options = {});

/// Maps every C++ parameter name to its key in the PyTorch state_dict, e.g.
/// "bp3/m17/n5.weight" -> "nodes.bp3_m17_n5.linear.weight".
std::map<std::string, std::string> torch_parameter_names(const Network& net);

/// Writes the network's weights under their PyTorch names.
void save_torch_weights(const Network& net, const std::string& path);
/// Loads weights saved from PyTorch (state_dict keys) into the network.
/// strict: every network parameter must be present with a matching shape.
void import_torch_weights(Network& net, const std::string& path, bool strict = true);

/// Runs the network on `steps` (one vector of inputs per time step, in input
/// order), from a zero recurrent state and in eval mode, and saves inputs,
/// outputs and the parameter gradients of loss = sum_t sum_outputs 0.5*|y|^2
/// (keyed by PyTorch names). tools/verify_torch_export.py uses this file to check
/// that the exported module matches the C++ network. Gradients and recurrent
/// state of `net` are reset afterwards; its training mode is restored.
void export_torch_reference(Network& net, const std::vector<std::vector<Tensor>>& steps, const std::string& path);

// ---- safetensors I/O (dtype F32 written; F32/F64/F16/BF16 read) ----
void save_safetensors(const std::vector<NamedTensor>& tensors, const std::string& path,
                      const std::map<std::string, std::string>& metadata = {});
std::vector<NamedTensor> load_safetensors(const std::string& path,
                                          std::map<std::string, std::string>* metadata = nullptr);

}  // namespace cdn
