// NetworkGraph: construction, topological sort, pruning, validation, DOT output.

#include "codeepneat/graph.hpp"

#include <algorithm>
#include <map>
#include <queue>
#include <set>
#include <sstream>
#include <stdexcept>

namespace cdn {

const char* to_string(NodeKind k) {
  switch (k) {
    case NodeKind::Input: return "input";
    case NodeKind::Layer: return "layer";
    case NodeKind::Output: return "output";
  }
  return "?";
}

int NetworkGraph::add_input(const std::string& name, const Shape& shape) {
  GraphNode n;
  n.id = static_cast<int>(nodes.size());
  n.name = name;
  n.kind = NodeKind::Input;
  n.layer_type = "input";
  n.shape = shape;
  n.origin.role = "input";
  nodes.push_back(n);
  return n.id;
}

int NetworkGraph::add_layer(const std::string& name, const std::string& type, const HyperparamValues& hp,
                            const std::vector<int>& inputs, const std::string& merge) {
  GraphNode n;
  n.id = static_cast<int>(nodes.size());
  n.name = name;
  n.kind = NodeKind::Layer;
  n.layer_type = type;
  n.hp = hp;
  n.inputs = inputs;
  n.merge = merge;
  n.origin.role = "layer";
  nodes.push_back(n);
  return n.id;
}

int NetworkGraph::add_identity(const std::string& name, const std::vector<int>& inputs, const std::string& merge) {
  return add_layer(name, "identity", HyperparamValues(), inputs, merge);
}

int NetworkGraph::add_output(const std::string& name, int input, int size, const std::string& activation,
                             float init_scale) {
  GraphNode n;
  n.id = static_cast<int>(nodes.size());
  n.name = name;
  n.kind = NodeKind::Output;
  n.layer_type = "dense";
  n.inputs = {input};
  n.out_size = size;
  n.activation = activation;
  n.init_scale = init_scale;
  n.origin.role = "head";
  nodes.push_back(n);
  return n.id;
}

void NetworkGraph::add_cell_skip(int from, int to) { cell_skips.push_back({from, to}); }

int NetworkGraph::find(const std::string& name) const {
  for (const auto& n : nodes)
    if (n.name == name) return n.id;
  return -1;
}

std::vector<int> NetworkGraph::input_ids() const {
  std::vector<int> r;
  for (const auto& n : nodes)
    if (n.kind == NodeKind::Input) r.push_back(n.id);
  return r;
}

std::vector<int> NetworkGraph::output_ids() const {
  std::vector<int> r;
  for (const auto& n : nodes)
    if (n.kind == NodeKind::Output) r.push_back(n.id);
  return r;
}

std::vector<int> NetworkGraph::topological_order() const {
  size_t n = nodes.size();
  std::vector<int> indeg(n, 0);
  std::vector<std::vector<int>> out(n);
  for (const auto& node : nodes)
    for (int i : node.inputs) {
      if (i < 0 || static_cast<size_t>(i) >= n) throw std::invalid_argument("graph: bad input id in " + node.name);
      out[i].push_back(node.id);
      ++indeg[node.id];
    }
  std::priority_queue<int, std::vector<int>, std::greater<int>> q;
  for (size_t i = 0; i < n; ++i)
    if (indeg[i] == 0) q.push(static_cast<int>(i));
  std::vector<int> order;
  while (!q.empty()) {
    int v = q.top();
    q.pop();
    order.push_back(v);
    for (int w : out[v])
      if (--indeg[w] == 0) q.push(w);
  }
  if (order.size() != n) throw std::invalid_argument("graph: contains a cycle");
  return order;
}

NetworkGraph NetworkGraph::pruned() const {
  size_t n = nodes.size();
  std::vector<std::vector<int>> out(n);
  for (const auto& node : nodes)
    for (int i : node.inputs) out.at(i).push_back(node.id);
  std::vector<char> fwd(n, 0), bwd(n, 0);
  std::vector<int> stack;
  for (const auto& node : nodes)
    if (node.kind == NodeKind::Input) {
      fwd[node.id] = 1;
      stack.push_back(node.id);
    }
  while (!stack.empty()) {
    int v = stack.back();
    stack.pop_back();
    for (int w : out[v])
      if (!fwd[w]) {
        fwd[w] = 1;
        stack.push_back(w);
      }
  }
  for (const auto& node : nodes)
    if (node.kind == NodeKind::Output) {
      bwd[node.id] = 1;
      stack.push_back(node.id);
    }
  while (!stack.empty()) {
    int v = stack.back();
    stack.pop_back();
    for (int w : nodes[v].inputs)
      if (!bwd[w]) {
        bwd[w] = 1;
        stack.push_back(w);
      }
  }
  NetworkGraph g;
  std::vector<int> remap(n, -1);
  for (const auto& node : nodes) {
    bool keep = node.kind == NodeKind::Input || (fwd[node.id] && bwd[node.id]);
    if (!keep) continue;
    GraphNode c = node;
    c.id = static_cast<int>(g.nodes.size());
    remap[node.id] = c.id;
    g.nodes.push_back(c);
  }
  for (auto& node : g.nodes) {
    std::vector<int> ins;
    for (int i : node.inputs)
      if (remap[i] >= 0 && fwd[i]) ins.push_back(remap[i]);
    node.inputs = ins;
  }
  for (const auto& e : cell_skips)
    if (remap.at(e.from) >= 0 && remap.at(e.to) >= 0) g.cell_skips.push_back({remap[e.from], remap[e.to]});
  return g;
}

void NetworkGraph::validate() const {
  std::set<std::string> names;
  bool has_in = false, has_out = false;
  for (size_t i = 0; i < nodes.size(); ++i) {
    const auto& node = nodes[i];
    if (node.id != static_cast<int>(i)) throw std::invalid_argument("graph: node ids must equal their index");
    if (!names.insert(node.name).second) throw std::invalid_argument("graph: duplicate node name '" + node.name + "'");
    if (node.kind == NodeKind::Input) {
      has_in = true;
      if (!node.inputs.empty()) throw std::invalid_argument("graph: input node with inputs: " + node.name);
      if (node.shape.empty()) throw std::invalid_argument("graph: input node without shape: " + node.name);
    } else {
      if (node.inputs.empty()) throw std::invalid_argument("graph: node without inputs: " + node.name);
      if (node.kind == NodeKind::Output) {
        has_out = true;
        if (node.out_size <= 0) throw std::invalid_argument("graph: output size must be > 0: " + node.name);
      }
      if (node.merge != "concat" && node.merge != "sum" && node.merge != "add")
        throw std::invalid_argument("graph: unknown merge '" + node.merge + "' in " + node.name);
    }
  }
  if (!has_in) throw std::invalid_argument("graph: no input nodes");
  if (!has_out) throw std::invalid_argument("graph: no output nodes");
  for (const auto& node : nodes)
    for (int i : node.inputs)
      if (nodes.at(i).kind == NodeKind::Output)
        throw std::invalid_argument("graph: output node used as input of " + node.name);
  for (const auto& e : cell_skips)
    if (e.from < 0 || e.to < 0 || static_cast<size_t>(e.from) >= nodes.size() ||
        static_cast<size_t>(e.to) >= nodes.size() || e.from == e.to)
      throw std::invalid_argument("graph: bad cell skip edge");
  topological_order();
}

std::string NetworkGraph::to_dot() const {
  std::ostringstream os;
  os << "digraph network {\n  rankdir=TB;\n  node [fontname=\"Helvetica\", fontsize=10];\n";
  for (const auto& n : nodes) {
    std::string shape = n.kind == NodeKind::Input ? "diamond" : (n.kind == NodeKind::Output ? "doublecircle" : "box");
    std::string label = n.name + "\\n" + (n.kind == NodeKind::Layer ? n.layer_type : to_string(n.kind));
    if (n.kind == NodeKind::Layer && n.layer_type != "identity") {
      if (n.hp.has("units")) label += " " + n.hp.s("units");
      if (n.hp.has("filters")) label += " f" + n.hp.s("filters") + " k" + n.hp.s("kernel_size", "3");
    }
    if (n.kind == NodeKind::Output) label += " " + std::to_string(n.out_size);
    if (n.inputs.size() > 1) label += "\\nmerge=" + n.merge;
    std::string style = n.layer_type == "identity" ? ", style=dashed" : "";
    os << "  n" << n.id << " [shape=" << shape << style << ", label=\"" << label << "\"];\n";
  }
  for (const auto& n : nodes)
    for (int i : n.inputs) os << "  n" << i << " -> n" << n.id << ";\n";
  for (const auto& e : cell_skips)
    os << "  n" << e.from << " -> n" << e.to << " [style=dashed, color=red, label=\"cell\"];\n";
  os << "}\n";
  return os.str();
}

}  // namespace cdn
