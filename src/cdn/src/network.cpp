// Network: builds a graph into runtime nodes and runs it, with hooks, recurrent
// state handling, parameter access and weight I/O.

#include "codeepneat/network.hpp"

#include <algorithm>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>

namespace cdn {

Network::Network(const NetworkGraph& graph, const BuildOptions& options, uint64_t seed)
    : graph_(graph.pruned()), options_(options), rng_(seed) {
  graph_.validate();
  std::vector<int> order = graph_.topological_order();
  std::vector<int> rt_of(graph_.nodes.size(), -1);
  std::map<int, int> input_slot;
  for (int id : graph_.input_ids()) {
    input_slot[id] = static_cast<int>(input_names_.size());
    input_names_.push_back(graph_.nodes[id].name);
    input_shapes_.push_back(graph_.nodes[id].shape);
  }
  auto& registry = LayerRegistry::instance();
  for (int gid : order) {
    const GraphNode& gn = graph_.nodes[gid];
    Runtime r;
    r.info = std::make_shared<NodeInfo>();
    NodeInfo& info = *r.info;
    info.index = static_cast<int>(rt_.size());
    info.graph_id = gid;
    info.name = gn.name;
    info.kind = gn.kind;
    info.type = gn.layer_type;
    info.merge = gn.merge;
    info.origin = gn.origin;
    info.hp = gn.hp;
    if (gn.kind == NodeKind::Input) {
      r.input_slot = input_slot.at(gid);
      info.output_shape = gn.shape;
      info.merged_shape = gn.shape;
    } else {
      for (int i : gn.inputs) {
        int ri = rt_of.at(i);
        r.inputs.push_back(ri);
        info.input_shapes.push_back(rt_[ri].info->output_shape);
        info.input_names.push_back(rt_[ri].info->name);
      }
      if (r.inputs.size() > 1) {
        r.merge.reset(new Merge(gn.name, gn.merge, info.input_shapes, options_, rng_));
        info.merged_shape = r.merge->output_shape();
      } else {
        info.merged_shape = info.input_shapes.at(0);
      }
      if (gn.kind == NodeKind::Output) {
        HyperparamValues hp;
        hp.set_int("units", gn.out_size);
        hp.set_choice("activation", gn.activation);
        hp.set_float("weight_scale", gn.init_scale);
        info.hp = hp;
        r.layer.reset(new DenseLayer(gn.name, hp, info.merged_shape, options_, rng_));
      } else {
        r.layer = registry.create(gn.layer_type, gn.name, gn.hp, info.merged_shape, options_, rng_);
      }
      info.output_shape = r.layer->output_shape();
      if (r.layer->is_recurrent()) recurrent_.push_back(r.layer.get());
    }
    rt_of[gid] = static_cast<int>(rt_.size());
    by_name_[gn.name] = static_cast<int>(rt_.size());
    rt_.push_back(std::move(r));
  }
  for (const auto& e : graph_.cell_skips) {
    auto* src = dynamic_cast<LSTMLayer*>(rt_[rt_of.at(e.from)].layer.get());
    auto* dst = dynamic_cast<LSTMLayer*>(rt_[rt_of.at(e.to)].layer.get());
    if (src && dst) dst->add_cell_skip(src, options_, rng_);
  }
  for (int id : graph_.output_ids()) {
    output_rt_.push_back(rt_of.at(id));
    output_names_.push_back(graph_.nodes[id].name);
  }
  for (auto& r : rt_) {
    size_t n = 0;
    if (r.layer) n += r.layer->num_parameters();
    if (r.merge)
      for (const auto& p : r.merge->params()) n += p.tensor.numel();
    r.info->num_parameters = n;
    info_.push_back(*r.info);
  }
  reset_state(1);
}

int Network::output_index(const std::string& name) const {
  for (size_t i = 0; i < output_names_.size(); ++i)
    if (output_names_[i] == name || output_names_[i] == "head:" + name) return static_cast<int>(i);
  return -1;
}

int Network::input_index(const std::string& name) const {
  for (size_t i = 0; i < input_names_.size(); ++i)
    if (input_names_[i] == name || input_names_[i] == "input:" + name) return static_cast<int>(i);
  return -1;
}

// ============================================================================
// Forward
// ============================================================================

std::vector<Tensor> Network::forward(const std::vector<Tensor>& inputs) {
  if (inputs.size() != input_names_.size())
    throw std::invalid_argument("Network::forward: expected " + std::to_string(input_names_.size()) + " inputs, got " +
                                std::to_string(inputs.size()));
  std::vector<Tensor> xs(inputs.size());
  int batch = -1;
  for (size_t k = 0; k < inputs.size(); ++k) {
    Tensor x = inputs[k];
    const Shape& s = input_shapes_[k];
    if (x.shape() == s) {
      Shape bs = s;
      bs.insert(bs.begin(), 1);
      x = reshape(x, bs);
    }
    bool ok = x.dim() == static_cast<int>(s.size()) + 1 && std::equal(s.begin(), s.end(), x.shape().begin() + 1);
    if (!ok)
      throw std::invalid_argument("Network::forward: input '" + input_names_[k] + "' has shape " +
                                  shape_str(x.shape()) + ", expected [B]+" + shape_str(s));
    if (batch >= 0 && x.shape()[0] != batch) throw std::invalid_argument("Network::forward: inconsistent batch sizes");
    batch = x.shape()[0];
    xs[k] = x;
  }
  if (!recurrent_.empty() && state_batch_ != batch) reset_state(batch);

  LayerContext ctx{training_, &rng_};
  if (record_) activations_.clear();
  std::vector<Tensor> outs(rt_.size());
  std::vector<Tensor> args;
  for (size_t r = 0; r < rt_.size(); ++r) {
    Runtime& n = rt_[r];
    const NodeInfo& info = *n.info;
    Tensor y;
    if (n.input_slot >= 0) {
      y = xs[n.input_slot];
    } else {
      Tensor x;
      if (n.merge) {
        args.clear();
        for (int i : n.inputs) args.push_back(outs[i]);
        x = n.merge->forward(args);
      } else {
        x = outs[n.inputs[0]];
      }
      for (int h : n.pre) hooks_[h].fhook(info, x);
      y = n.layer->forward(x, ctx);
    }
    for (int h : n.fwd) hooks_[h].fhook(info, y);
    if (!n.bwd.empty() && y.requires_grad()) {
      for (int h : n.bwd) {
        std::shared_ptr<NodeInfo> ip = n.info;
        BackwardHook fn = hooks_[h].bhook;
        y.register_hook([ip, fn](Tensor& g) { fn(*ip, g); });
      }
    }
    if (record_) activations_[info.name] = y;
    outs[r] = y;
  }
  std::vector<Tensor> result;
  result.reserve(output_rt_.size());
  for (int r : output_rt_) result.push_back(outs[r]);
  return result;
}

std::map<std::string, Tensor> Network::forward(const std::map<std::string, Tensor>& inputs) {
  std::vector<Tensor> xs(input_names_.size());
  for (const auto& kv : inputs) {
    int k = input_index(kv.first);
    if (k < 0) throw std::invalid_argument("Network::forward: unknown input '" + kv.first + "'");
    xs[k] = kv.second;
  }
  for (size_t k = 0; k < xs.size(); ++k)
    if (!xs[k].defined()) throw std::invalid_argument("Network::forward: missing input '" + input_names_[k] + "'");
  std::vector<Tensor> ys = forward(xs);
  std::map<std::string, Tensor> r;
  for (size_t k = 0; k < ys.size(); ++k) {
    std::string n = output_names_[k];
    if (n.rfind("head:", 0) == 0) n = n.substr(5);
    r[n] = ys[k];
  }
  return r;
}

Tensor Network::forward(const Tensor& x) {
  if (input_names_.size() != 1) throw std::invalid_argument("Network::forward(Tensor): network has multiple inputs");
  return forward(std::vector<Tensor>{x}).at(0);
}

// ============================================================================
// Recurrent state
// ============================================================================

void Network::reset_state(int batch) {
  for (Layer* l : recurrent_) l->reset_state(batch);
  state_batch_ = batch;
}

void Network::detach_state() {
  for (Layer* l : recurrent_) l->detach_state();
}

void Network::mask_state(const std::vector<float>& keep) {
  if (recurrent_.empty()) return;
  if (static_cast<int>(keep.size()) != state_batch_)
    throw std::invalid_argument("Network::mask_state: size does not match state batch");
  Tensor k({static_cast<int>(keep.size()), 1}, keep);
  for (Layer* l : recurrent_) l->mask_state(k);
}

std::vector<NamedTensor> Network::get_state() const {
  std::vector<NamedTensor> s;
  for (Layer* l : recurrent_)
    for (auto& t : l->state()) s.push_back(t);
  return s;
}

void Network::set_state(const std::vector<NamedTensor>& state) {
  std::map<std::string, Tensor> by;
  for (const auto& t : state) by[t.name] = t.tensor;
  int batch = -1;
  for (Layer* l : recurrent_) {
    std::vector<Tensor> parts;
    for (auto& t : l->state()) {
      auto it = by.find(t.name);
      if (it == by.end()) throw std::invalid_argument("Network::set_state: missing " + t.name);
      parts.push_back(it->second);
    }
    l->set_state(parts);
    if (!parts.empty()) batch = parts[0].shape()[0];
  }
  if (batch > 0) state_batch_ = batch;
}

// ============================================================================
// Parameters
// ============================================================================

std::vector<NamedTensor> Network::named_parameters() const {
  std::vector<NamedTensor> r;
  for (const auto& n : rt_) {
    if (n.merge)
      for (const auto& p : n.merge->params()) r.push_back(p);
    if (n.layer)
      for (const auto& p : n.layer->params()) r.push_back(p);
  }
  return r;
}

std::vector<Tensor> Network::parameters() const {
  std::vector<Tensor> r;
  for (const auto& p : named_parameters()) r.push_back(p.tensor);
  return r;
}

Tensor Network::parameter(const std::string& name) const {
  for (const auto& p : named_parameters())
    if (p.name == name) return p.tensor;
  throw std::invalid_argument("Network::parameter: no parameter named '" + name + "'");
}

bool Network::has_parameter(const std::string& name) const {
  for (const auto& p : named_parameters())
    if (p.name == name) return true;
  return false;
}

size_t Network::num_parameters() const {
  size_t n = 0;
  for (const auto& p : named_parameters()) n += p.tensor.numel();
  return n;
}

void Network::zero_grad() {
  for (auto p : parameters()) p.zero_grad();
}

size_t Network::set_trainable(const std::string& prefix, bool trainable) {
  size_t n = 0;
  for (auto p : named_parameters())
    if (p.name.compare(0, prefix.size(), prefix) == 0) {
      p.tensor.set_requires_grad(trainable);
      ++n;
    }
  return n;
}

size_t Network::set_trainable(const NodeFilter& filter, bool trainable) {
  size_t n = 0;
  for (auto& r : rt_) {
    if (!filter(*r.info)) continue;
    if (r.merge)
      for (auto& p : r.merge->params()) {
        p.tensor.set_requires_grad(trainable);
        ++n;
      }
    if (r.layer)
      for (auto& p : r.layer->params()) {
        p.tensor.set_requires_grad(trainable);
        ++n;
      }
  }
  return n;
}

// ============================================================================
// Nodes
// ============================================================================

const NodeInfo& Network::node(const std::string& name) const {
  auto it = by_name_.find(name);
  if (it == by_name_.end()) throw std::invalid_argument("Network::node: no node named '" + name + "'");
  return info_[it->second];
}

bool Network::has_node(const std::string& name) const { return by_name_.count(name) != 0; }

Layer* Network::layer(const std::string& name) {
  auto it = by_name_.find(name);
  if (it == by_name_.end()) return nullptr;
  return rt_[it->second].layer.get();
}

const Layer* Network::layer(const std::string& name) const {
  auto it = by_name_.find(name);
  if (it == by_name_.end()) return nullptr;
  return rt_[it->second].layer.get();
}

const Merge* Network::merge(const std::string& name) const {
  auto it = by_name_.find(name);
  if (it == by_name_.end()) return nullptr;
  return rt_[it->second].merge.get();
}

std::vector<Layer*> Network::layers() {
  std::vector<Layer*> r;
  for (auto& n : rt_)
    if (n.layer) r.push_back(n.layer.get());
  return r;
}

// ============================================================================
// Hooks
// ============================================================================

int Network::add_hook(int kind, NodeFilter filter, ForwardHook f, BackwardHook b) {
  HookEntry e{next_hook_id_++, kind, std::move(filter), std::move(f), std::move(b)};
  hooks_.push_back(std::move(e));
  rebuild_hook_cache();
  return hooks_.back().id;
}

int Network::add_forward_hook(ForwardHook hook, NodeFilter filter) {
  return add_hook(0, std::move(filter), std::move(hook), nullptr);
}

int Network::add_forward_hook(const std::string& node_name, ForwardHook hook) {
  node(node_name);  // validate
  return add_hook(
      0, [node_name](const NodeInfo& n) { return n.name == node_name; }, std::move(hook), nullptr);
}

int Network::add_forward_pre_hook(ForwardHook hook, NodeFilter filter) {
  return add_hook(1, std::move(filter), std::move(hook), nullptr);
}

int Network::add_backward_hook(BackwardHook hook, NodeFilter filter) {
  return add_hook(2, std::move(filter), nullptr, std::move(hook));
}

int Network::add_backward_hook(const std::string& node_name, BackwardHook hook) {
  node(node_name);
  return add_hook(
      2, [node_name](const NodeInfo& n) { return n.name == node_name; }, nullptr, std::move(hook));
}

int Network::add_parameter_hook(const std::string& param_name, GradHook hook) {
  Tensor t = parameter(param_name);
  int hid = t.register_hook(std::move(hook));
  int id = next_hook_id_++;
  param_hooks_[id] = {t, hid};
  return id;
}

void Network::remove_hook(int id) {
  auto it = param_hooks_.find(id);
  if (it != param_hooks_.end()) {
    it->second.first.remove_hook(it->second.second);
    param_hooks_.erase(it);
    return;
  }
  hooks_.erase(std::remove_if(hooks_.begin(), hooks_.end(), [id](const HookEntry& e) { return e.id == id; }),
               hooks_.end());
  rebuild_hook_cache();
}

void Network::clear_hooks() {
  for (auto& kv : param_hooks_) kv.second.first.remove_hook(kv.second.second);
  param_hooks_.clear();
  hooks_.clear();
  rebuild_hook_cache();
}

void Network::rebuild_hook_cache() {
  for (auto& r : rt_) {
    r.fwd.clear();
    r.pre.clear();
    r.bwd.clear();
    for (size_t h = 0; h < hooks_.size(); ++h) {
      const auto& e = hooks_[h];
      if (e.filter && !e.filter(*r.info)) continue;
      if (e.kind == 0) r.fwd.push_back(static_cast<int>(h));
      if (e.kind == 1 && r.input_slot < 0) r.pre.push_back(static_cast<int>(h));
      if (e.kind == 2) r.bwd.push_back(static_cast<int>(h));
    }
  }
}

Tensor Network::activation(const std::string& node_name) const {
  auto it = activations_.find(node_name);
  if (it == activations_.end())
    throw std::invalid_argument("Network::activation: no recorded activation for '" + node_name +
                                "' (enable record_activations and run forward)");
  return it->second;
}

// ============================================================================
// I/O
// ============================================================================

std::vector<NamedTensor> Network::state_dict() const {
  std::vector<NamedTensor> r;
  for (const auto& p : named_parameters()) r.push_back({p.name, p.tensor.detach()});
  return r;
}

void Network::load_state_dict(const std::vector<NamedTensor>& state, bool strict) {
  std::map<std::string, const Tensor*> by;
  for (const auto& t : state) by[t.name] = &t.tensor;
  for (auto p : named_parameters()) {
    auto it = by.find(p.name);
    if (it == by.end()) {
      if (strict) throw std::invalid_argument("load_state_dict: missing parameter " + p.name);
      continue;
    }
    if (it->second->numel() != p.tensor.numel()) {
      if (strict) throw std::invalid_argument("load_state_dict: size mismatch for " + p.name);
      continue;
    }
    p.tensor.copy_from(*it->second);
  }
}

void Network::save_weights(const std::string& path) const {
  std::ofstream f(path);
  if (!f) throw std::runtime_error("save_weights: cannot open " + path);
  auto params = named_parameters();
  f << "cdn-weights 1 " << params.size() << "\n";
  f << std::setprecision(9);
  for (const auto& p : params) {
    f << p.name << ' ' << p.tensor.dim();
    for (int d : p.tensor.shape()) f << ' ' << d;
    f << '\n';
    for (size_t i = 0; i < p.tensor.numel(); ++i) f << (i ? " " : "") << p.tensor.data()[i];
    f << '\n';
  }
}

void Network::load_weights(const std::string& path, bool strict) {
  std::ifstream f(path);
  if (!f) throw std::runtime_error("load_weights: cannot open " + path);
  std::string magic;
  int version = 0;
  size_t count = 0;
  f >> magic >> version >> count;
  if (magic != "cdn-weights" || version != 1) throw std::runtime_error("load_weights: bad header in " + path);
  std::vector<NamedTensor> state;
  for (size_t k = 0; k < count; ++k) {
    std::string name;
    int rank = 0;
    f >> name >> rank;
    Shape s(rank);
    for (int& d : s) f >> d;
    std::vector<float> v(shape_numel(s));
    for (auto& x : v) f >> x;
    if (!f) throw std::runtime_error("load_weights: truncated file " + path);
    state.push_back({name, Tensor(s, std::move(v))});
  }
  load_state_dict(state, strict);
}

std::string Network::summary() const {
  std::ostringstream os;
  os << "Network: " << info_.size() << " nodes, " << num_parameters() << " parameters";
  if (is_recurrent()) os << ", recurrent";
  os << "\n";
  os << std::left << std::setw(4) << "#" << std::setw(28) << "name" << std::setw(10) << "type" << std::setw(16)
     << "output" << std::setw(10) << "params" << "details\n";
  for (const auto& n : info_) {
    std::string type = n.kind == NodeKind::Output ? "head" : n.type;
    os << std::left << std::setw(4) << n.index << std::setw(28) << n.name << std::setw(10) << type << std::setw(16)
       << shape_str(n.output_shape) << std::setw(10) << n.num_parameters;
    if (n.input_names.size() > 1) os << "merge=" << n.merge << "(" << n.input_names.size() << ") ";
    if (n.kind != NodeKind::Input && n.type != "identity") os << n.hp.to_string();
    os << "\n";
  }
  return os.str();
}

}  // namespace cdn
