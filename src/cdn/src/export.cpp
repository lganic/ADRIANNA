// safetensors I/O and PyTorch export. make_plan() walks a built Network once and
// produces both the JSON architecture spec and the C++-to-PyTorch parameter name map;
// kTorchRuntime is the Python code embedded in every generated module, and must stay
// consistent with the parameter paths chosen in make_plan().

#include "codeepneat/export.hpp"

#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <set>
#include <sstream>
#include <stdexcept>

namespace cdn {

// ============================================================================
// Small JSON helpers (writer + a parser sufficient for safetensors headers)
// ============================================================================

namespace {

std::string json_str(const std::string& s) {
  std::string r = "\"";
  for (char ch : s) {
    unsigned char c = static_cast<unsigned char>(ch);
    switch (ch) {
      case '"': r += "\\\""; break;
      case '\\': r += "\\\\"; break;
      case '\n': r += "\\n"; break;
      case '\t': r += "\\t"; break;
      case '\r': r += "\\r"; break;
      case '\'': r += "\\u0027"; break;  // keeps the JSON safe inside a Python r'''...''' literal
      default:
        if (c < 0x20) {
          char buf[8];
          std::snprintf(buf, sizeof(buf), "\\u%04x", c);
          r += buf;
        } else {
          r += ch;
        }
    }
  }
  return r + "\"";
}

std::string json_shape(const Shape& s) {
  std::string r = "[";
  for (size_t i = 0; i < s.size(); ++i) r += (i ? "," : "") + std::to_string(s[i]);
  return r + "]";
}

std::string json_num(double v) {
  char buf[64];
  std::snprintf(buf, sizeof(buf), "%.9g", v);
  return buf;
}

const char* json_bool(bool b) { return b ? "true" : "false"; }

struct Json {
  enum Type { Null, Bool, Num, Str, Arr, Obj } type = Null;
  bool b = false;
  double num = 0;
  std::string str;
  std::vector<Json> arr;
  std::vector<std::pair<std::string, Json>> obj;
  const Json* get(const std::string& k) const {
    for (const auto& kv : obj)
      if (kv.first == k) return &kv.second;
    return nullptr;
  }
};

class JsonParser {
 public:
  explicit JsonParser(const std::string& s) : s_(s) {}
  Json parse() {
    Json v = value();
    ws();
    if (i_ != s_.size()) fail("trailing characters");
    return v;
  }

 private:
  [[noreturn]] void fail(const std::string& m) {
    throw std::runtime_error("safetensors header JSON: " + m + " at offset " + std::to_string(i_));
  }
  void ws() {
    while (i_ < s_.size() && std::isspace(static_cast<unsigned char>(s_[i_]))) ++i_;
  }
  char peek() {
    ws();
    if (i_ >= s_.size()) fail("unexpected end");
    return s_[i_];
  }
  void expect(char c) {
    if (peek() != c) fail(std::string("expected '") + c + "'");
    ++i_;
  }
  Json value() {
    char c = peek();
    Json v;
    if (c == '{') {
      v.type = Json::Obj;
      ++i_;
      if (peek() == '}') {
        ++i_;
        return v;
      }
      for (;;) {
        std::string k = string_lit();
        expect(':');
        v.obj.emplace_back(k, value());
        if (peek() == ',') {
          ++i_;
          continue;
        }
        expect('}');
        return v;
      }
    }
    if (c == '[') {
      v.type = Json::Arr;
      ++i_;
      if (peek() == ']') {
        ++i_;
        return v;
      }
      for (;;) {
        v.arr.push_back(value());
        if (peek() == ',') {
          ++i_;
          continue;
        }
        expect(']');
        return v;
      }
    }
    if (c == '"') {
      v.type = Json::Str;
      v.str = string_lit();
      return v;
    }
    if (s_.compare(i_, 4, "true") == 0) {
      i_ += 4;
      v.type = Json::Bool;
      v.b = true;
      return v;
    }
    if (s_.compare(i_, 5, "false") == 0) {
      i_ += 5;
      v.type = Json::Bool;
      return v;
    }
    if (s_.compare(i_, 4, "null") == 0) {
      i_ += 4;
      return v;
    }
    size_t start = i_;
    while (i_ < s_.size() && (std::isdigit(static_cast<unsigned char>(s_[i_])) || std::strchr("+-.eE", s_[i_]))) ++i_;
    if (start == i_) fail("unexpected character");
    v.type = Json::Num;
    v.num = std::stod(s_.substr(start, i_ - start));
    return v;
  }
  std::string string_lit() {
    expect('"');
    std::string r;
    while (i_ < s_.size() && s_[i_] != '"') {
      char c = s_[i_++];
      if (c != '\\') {
        r += c;
        continue;
      }
      if (i_ >= s_.size()) fail("bad escape");
      char e = s_[i_++];
      switch (e) {
        case 'n': r += '\n'; break;
        case 't': r += '\t'; break;
        case 'r': r += '\r'; break;
        case 'b': r += '\b'; break;
        case 'f': r += '\f'; break;
        case 'u': {
          if (i_ + 4 > s_.size()) fail("bad \\u escape");
          unsigned cp = static_cast<unsigned>(std::stoul(s_.substr(i_, 4), nullptr, 16));
          i_ += 4;
          if (cp < 0x80) {
            r += static_cast<char>(cp);
          } else if (cp < 0x800) {
            r += static_cast<char>(0xC0 | (cp >> 6));
            r += static_cast<char>(0x80 | (cp & 0x3F));
          } else {
            r += static_cast<char>(0xE0 | (cp >> 12));
            r += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
            r += static_cast<char>(0x80 | (cp & 0x3F));
          }
          break;
        }
        default: r += e;
      }
    }
    if (i_ >= s_.size()) fail("unterminated string");
    ++i_;
    return r;
  }
  const std::string& s_;
  size_t i_ = 0;
};

float half_to_float(uint16_t h) {
  uint32_t sign = (h & 0x8000u) << 16, exp = (h >> 10) & 0x1F, mant = h & 0x3FF;
  uint32_t bits;
  if (exp == 0) {
    if (mant == 0) {
      bits = sign;
    } else {  // subnormal
      exp = 127 - 15 + 1;
      while (!(mant & 0x400)) {
        mant <<= 1;
        --exp;
      }
      mant &= 0x3FF;
      bits = sign | (exp << 23) | (mant << 13);
    }
  } else if (exp == 31) {
    bits = sign | 0x7F800000u | (mant << 13);
  } else {
    bits = sign | ((exp + 127 - 15) << 23) | (mant << 13);
  }
  float f;
  std::memcpy(&f, &bits, 4);
  return f;
}

}  // namespace

// ============================================================================
// safetensors
// ============================================================================

void save_safetensors(const std::vector<NamedTensor>& tensors, const std::string& path,
                      const std::map<std::string, std::string>& metadata) {
  std::string header = "{";
  bool first = true;
  if (!metadata.empty()) {
    header += "\"__metadata__\":{";
    bool f2 = true;
    for (const auto& kv : metadata) {
      header += (f2 ? "" : ",") + json_str(kv.first) + ":" + json_str(kv.second);
      f2 = false;
    }
    header += "}";
    first = false;
  }
  std::set<std::string> names;
  uint64_t offset = 0;
  for (const auto& t : tensors) {
    if (!names.insert(t.name).second) throw std::invalid_argument("save_safetensors: duplicate name " + t.name);
    uint64_t bytes = static_cast<uint64_t>(t.tensor.numel()) * 4;
    header += (first ? "" : ",") + json_str(t.name) + ":{\"dtype\":\"F32\",\"shape\":" + json_shape(t.tensor.shape()) +
              ",\"data_offsets\":[" + std::to_string(offset) + "," + std::to_string(offset + bytes) + "]}";
    first = false;
    offset += bytes;
  }
  header += "}";
  while ((8 + header.size()) % 8 != 0) header += ' ';
  std::ofstream f(path, std::ios::binary);
  if (!f) throw std::runtime_error("save_safetensors: cannot open " + path);
  uint64_t n = header.size();
  unsigned char len[8];
  for (int i = 0; i < 8; ++i) len[i] = static_cast<unsigned char>((n >> (8 * i)) & 0xFF);
  f.write(reinterpret_cast<const char*>(len), 8);
  f.write(header.data(), static_cast<std::streamsize>(header.size()));
  std::vector<unsigned char> buf;
  for (const auto& t : tensors) {
    buf.resize(t.tensor.numel() * 4);
    for (size_t i = 0; i < t.tensor.numel(); ++i) {
      uint32_t bits;
      std::memcpy(&bits, &t.tensor.data()[i], 4);
      for (int k = 0; k < 4; ++k) buf[i * 4 + k] = static_cast<unsigned char>((bits >> (8 * k)) & 0xFF);
    }
    f.write(reinterpret_cast<const char*>(buf.data()), static_cast<std::streamsize>(buf.size()));
  }
  if (!f) throw std::runtime_error("save_safetensors: write failed for " + path);
}

std::vector<NamedTensor> load_safetensors(const std::string& path, std::map<std::string, std::string>* metadata) {
  std::ifstream f(path, std::ios::binary);
  if (!f) throw std::runtime_error("load_safetensors: cannot open " + path);
  std::string raw((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
  if (raw.size() < 8) throw std::runtime_error("load_safetensors: file too small");
  uint64_t n = 0;
  for (int i = 0; i < 8; ++i) n |= static_cast<uint64_t>(static_cast<unsigned char>(raw[i])) << (8 * i);
  if (8 + n > raw.size()) throw std::runtime_error("load_safetensors: bad header length");
  Json header = JsonParser(raw.substr(8, n)).parse();
  if (header.type != Json::Obj) throw std::runtime_error("load_safetensors: header is not an object");
  const unsigned char* data = reinterpret_cast<const unsigned char*>(raw.data()) + 8 + n;
  size_t data_size = raw.size() - 8 - n;
  std::vector<NamedTensor> out;
  for (const auto& kv : header.obj) {
    if (kv.first == "__metadata__") {
      if (metadata)
        for (const auto& m : kv.second.obj) (*metadata)[m.first] = m.second.str;
      continue;
    }
    const Json* dt = kv.second.get("dtype");
    const Json* sh = kv.second.get("shape");
    const Json* off = kv.second.get("data_offsets");
    if (!dt || !sh || !off || off->arr.size() != 2) throw std::runtime_error("load_safetensors: bad entry " + kv.first);
    Shape shape;
    for (const auto& d : sh->arr) shape.push_back(static_cast<int>(d.num));
    size_t begin = static_cast<size_t>(off->arr[0].num), end = static_cast<size_t>(off->arr[1].num);
    size_t count = shape_numel(shape);
    size_t width = dt->str == "F32" ? 4 : dt->str == "F64" ? 8 : (dt->str == "F16" || dt->str == "BF16") ? 2 : 0;
    if (!width) throw std::runtime_error("load_safetensors: unsupported dtype " + dt->str + " for " + kv.first);
    if (end < begin || end > data_size || end - begin != count * width)
      throw std::runtime_error("load_safetensors: bad data offsets for " + kv.first);
    std::vector<float> v(count);
    const unsigned char* p = data + begin;
    for (size_t i = 0; i < count; ++i) {
      uint64_t bits = 0;
      for (size_t k = 0; k < width; ++k) bits |= static_cast<uint64_t>(p[i * width + k]) << (8 * k);
      if (width == 4) {
        uint32_t b32 = static_cast<uint32_t>(bits);
        std::memcpy(&v[i], &b32, 4);
      } else if (width == 8) {
        double d;
        std::memcpy(&d, &bits, 8);
        v[i] = static_cast<float>(d);
      } else if (dt->str == "BF16") {
        uint32_t b32 = static_cast<uint32_t>(bits) << 16;
        std::memcpy(&v[i], &b32, 4);
      } else {
        v[i] = half_to_float(static_cast<uint16_t>(bits));
      }
    }
    out.push_back({kv.first, Tensor(shape, std::move(v))});
  }
  return out;
}

// ============================================================================
// Export plan: node keys, parameter mapping and the JSON spec
// ============================================================================

namespace {

struct ExportPlan {
  std::map<std::string, std::string> params;  // C++ name -> torch key
  std::string spec_json;
};

std::string sanitize(const std::string& name, std::set<std::string>& used) {
  static const std::set<std::string> reserved = {
      "keys",    "items",     "values", "update",   "pop",      "clear",      "training", "forward", "to",
      "cuda",    "cpu",       "type",   "half",     "float",    "double",     "eval",     "train",   "apply",
      "modules", "children",  "parameters", "buffers", "state_dict", "add_module", "zero_grad", "extra_repr"};
  std::string k;
  for (char c : name) k += (std::isalnum(static_cast<unsigned char>(c)) || c == '_') ? c : '_';
  if (k.empty() || std::isdigit(static_cast<unsigned char>(k[0]))) k = "n" + k;
  if (reserved.count(k)) k += "_";
  std::string base = k;
  for (int i = 2; used.count(k); ++i) k = base + "_" + std::to_string(i);
  used.insert(k);
  return k;
}

ExportPlan make_plan(const Network& net) {
  ExportPlan plan;
  std::map<const TensorImpl*, std::string> cpp_name;
  for (const auto& p : net.named_parameters()) cpp_name[p.tensor.impl()] = p.name;
  auto map_param = [&](const Tensor& t, const std::string& key) {
    if (!t.defined()) return;
    auto it = cpp_name.find(t.impl());
    if (it == cpp_name.end()) throw std::logic_error("export_torch: unregistered parameter for " + key);
    plan.params[it->second] = key;
  };

  std::set<std::string> used;
  std::map<std::string, std::string> key_of;  // node name -> key
  for (const auto& n : net.nodes()) key_of[n.name] = sanitize(n.name, used);

  std::ostringstream js;
  js << "{\"format\":\"codeepneat-torch\",\"version\":1,\"inputs\":[";
  for (size_t i = 0; i < net.input_names().size(); ++i)
    js << (i ? "," : "") << "{\"name\":" << json_str(net.input_names()[i])
       << ",\"shape\":" << json_shape(net.input_shapes()[i]) << "}";
  js << "],\"outputs\":[";
  for (size_t i = 0; i < net.output_names().size(); ++i) js << (i ? "," : "") << json_str(net.output_names()[i]);
  js << "],\"output_keys\":[";
  for (size_t i = 0; i < net.output_names().size(); ++i)
    js << (i ? "," : "") << json_str(key_of.at(net.output_names()[i]));
  js << "],\"nodes\":[";

  for (size_t idx = 0; idx < net.nodes().size(); ++idx) {
    const NodeInfo& n = net.nodes()[idx];
    const std::string& key = key_of.at(n.name);
    const std::string prefix = "nodes." + key + ".";
    js << (idx ? "," : "") << "{\"name\":" << json_str(n.name) << ",\"key\":" << json_str(key);
    js << ",\"out_shape\":" << json_shape(n.output_shape);
    js << ",\"origin\":{\"role\":" << json_str(n.origin.role) << ",\"blueprint_node\":" << n.origin.blueprint_node
       << ",\"species\":" << n.origin.species << ",\"module_id\":" << n.origin.module_id
       << ",\"module_node\":" << n.origin.module_node << "}";
    js << ",\"hp\":" << json_str(n.hp.to_string());
    if (n.kind == NodeKind::Input) {
      int slot = net.input_index(n.name);
      js << ",\"kind\":\"input\",\"cls\":\"input\",\"slot\":" << slot << ",\"shape\":" << json_shape(n.output_shape)
         << "}";
      continue;
    }
    js << ",\"kind\":" << (n.kind == NodeKind::Output ? "\"output\"" : "\"layer\"") << ",\"inputs\":[";
    for (size_t i = 0; i < n.input_names.size(); ++i) js << (i ? "," : "") << json_str(key_of.at(n.input_names[i]));
    js << "]";

    // Merge.
    js << ",\"merge\":";
    if (const Merge* m = net.merge(n.name)) {
      js << "{\"method\":" << json_str(m->method()) << ",\"mode\":" << (m->spatial() ? "\"spatial\"" : "\"flat\"");
      js << ",\"pool\":[" << m->pool_size().first << "," << m->pool_size().second << "],\"proj\":{";
      bool first = true;
      for (size_t k = 0; k < m->projections().size(); ++k) {
        const Tensor& w = m->projections()[k];
        if (!w.defined()) continue;
        js << (first ? "" : ",") << "\"" << k << "\":[" << w.shape()[1] << "," << w.shape()[0] << "]";
        first = false;
        map_param(w, prefix + "merge.proj." + std::to_string(k) + ".weight");
      }
      js << "}}";
    } else {
      js << "null";
    }

    const Layer* layer = net.layer(n.name);
    if (auto* d = dynamic_cast<const DenseLayer*>(layer)) {
      js << ",\"cls\":\"dense\",\"dense\":{\"in\":" << d->weight().shape()[1] << ",\"units\":" << d->weight().shape()[0]
         << ",\"bias\":" << json_bool(d->bias().defined()) << ",\"activation\":" << json_str(d->activation_name())
         << ",\"dropout\":" << json_num(d->dropout_rate()) << "}";
      map_param(d->weight(), prefix + "linear.weight");
      map_param(d->bias(), prefix + "linear.bias");
    } else if (auto* c = dynamic_cast<const Conv2DLayer*>(layer)) {
      const Shape& ws = c->weight().shape();
      js << ",\"cls\":\"conv2d\",\"conv\":{\"image_shape\":" << json_shape(c->image_shape())
         << ",\"in_channels\":" << ws[1] << ",\"filters\":" << ws[0] << ",\"kernel\":[" << ws[2] << "," << ws[3]
         << "],\"stride\":" << c->stride() << ",\"padding\":" << c->padding()
         << ",\"bias\":" << json_bool(c->bias().defined()) << ",\"pool\":" << json_bool(c->max_pool())
         << ",\"activation\":" << json_str(c->activation_name()) << ",\"dropout\":" << json_num(c->dropout_rate())
         << "}";
      map_param(c->weight(), prefix + "conv.weight");
      map_param(c->bias(), prefix + "conv.bias");
    } else if (auto* l = dynamic_cast<const LSTMLayer*>(layer)) {
      js << ",\"cls\":\"lstm\",\"lstm\":{\"in\":" << l->input_features() << ",\"units\":" << l->units()
         << ",\"dropout\":" << json_num(l->dropout_rate()) << ",\"skips\":[";
      map_param(l->weight(), prefix + "weight");
      map_param(l->bias(), prefix + "bias");
      auto skips = l->cell_skips();
      for (size_t k = 0; k < skips.size(); ++k) {
        const auto& s = skips[k];
        js << (k ? "," : "") << "{\"source\":" << json_str(key_of.at(s.source->name()))
           << ",\"source_units\":" << s.source->units() << ",\"proj\":" << json_bool(s.projection.defined()) << "}";
        std::string sp = prefix + "skips." + std::to_string(k) + ".";
        map_param(s.gate_weight, sp + "gate.weight");
        map_param(s.gate_bias, sp + "gate.bias");
        map_param(s.projection, sp + "proj.weight");
      }
      js << "]}";
    } else if (dynamic_cast<const IdentityLayer*>(layer)) {
      js << ",\"cls\":\"identity\"";
    } else {
      throw std::invalid_argument("export_torch: node '" + n.name + "' has layer type '" +
                                  (layer ? layer->type() : std::string("?")) +
                                  "', which has no PyTorch equivalent in the exporter");
    }
    js << "}";
  }
  js << "],\"params\":{";
  bool first = true;
  for (const auto& kv : plan.params) {
    js << (first ? "" : ",") << json_str(kv.first) << ":" << json_str(kv.second);
    first = false;
  }
  js << "}}";
  for (const auto& p : net.named_parameters())
    if (!plan.params.count(p.name))
      throw std::logic_error("export_torch: parameter '" + p.name + "' was not mapped to PyTorch");
  plan.spec_json = js.str();
  return plan;
}

// The PyTorch runtime emitted into every generated module.
const char* kTorchRuntime = R"PY(# Generated by codeepneat::export_torch from a C++ CoDeepNEAT network.
# Architecture lives in SPEC below; weights are loaded from a .safetensors file.
import json
import struct

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["__CLASS__", "SPEC", "load_safetensors", "save_safetensors"]


def _activation(x, name):
    if name in ("linear", "identity", ""):
        return x
    if name == "relu":
        return F.relu(x)
    if name == "leaky_relu":
        return F.leaky_relu(x, 0.01)
    if name == "elu":
        return F.elu(x, 1.0)
    if name == "tanh":
        return torch.tanh(x)
    if name == "sigmoid":
        return torch.sigmoid(x)
    if name == "softplus":
        return F.softplus(x)
    raise ValueError("unknown activation %r" % name)


_DTYPES = {"F32": torch.float32, "F64": torch.float64, "F16": torch.float16, "BF16": torch.bfloat16}


def load_safetensors(path):
    """Read a .safetensors file into {name: tensor} (no `safetensors` package needed)."""
    with open(path, "rb") as f:
        raw = f.read()
    (n,) = struct.unpack("<Q", raw[:8])
    header = json.loads(raw[8:8 + n].decode("utf-8"))
    base = 8 + n
    out = {}
    for name, info in header.items():
        if name == "__metadata__":
            continue
        dtype = _DTYPES[info["dtype"]]
        begin, end = info["data_offsets"]
        buf = bytearray(raw[base + begin:base + end])
        if len(buf) == 0:
            out[name] = torch.zeros(info["shape"], dtype=dtype)
        else:
            out[name] = torch.frombuffer(buf, dtype=dtype).reshape(info["shape"]).clone()
    return out


def save_safetensors(tensors, path, metadata=None):
    """Write {name: tensor} as float32 safetensors (readable by cdn::import_torch_weights)."""
    header, blobs, offset = {}, [], 0
    if metadata:
        header["__metadata__"] = {str(k): str(v) for k, v in metadata.items()}
    for name in sorted(tensors):
        t = tensors[name].detach().to("cpu", torch.float32).contiguous()
        try:
            blob = t.numpy().tobytes()
        except Exception:  # numpy unavailable
            blob = struct.pack("<%df" % t.numel(), *t.flatten().tolist())
        header[name] = {"dtype": "F32", "shape": list(t.shape), "data_offsets": [offset, offset + len(blob)]}
        blobs.append(blob)
        offset += len(blob)
    h = json.dumps(header, separators=(",", ":")).encode("utf-8")
    h += b" " * ((8 - (8 + len(h)) % 8) % 8)
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(h)))
        f.write(h)
        for blob in blobs:
            f.write(blob)


class _Merge(nn.Module):
    """Merges several parent outputs (concat or sum) after bringing them to a common size."""

    def __init__(self, spec):
        super().__init__()
        self.method = spec["method"]
        self.mode = spec["mode"]
        self.pool = tuple(spec["pool"])
        self.proj = nn.ModuleDict()
        for k, (fan_in, fan_out) in spec["proj"].items():
            if self.mode == "spatial":
                self.proj[k] = nn.Conv2d(fan_in, fan_out, 1, bias=False)
            else:
                self.proj[k] = nn.Linear(fan_in, fan_out, bias=False)

    def forward(self, xs):
        ys = []
        for k, x in enumerate(xs):
            if self.mode == "spatial":
                if tuple(x.shape[2:]) != self.pool:
                    x = F.adaptive_max_pool2d(x, self.pool)
            else:
                x = torch.flatten(x, 1)
            if str(k) in self.proj:
                x = self.proj[str(k)](x)
            ys.append(x)
        if self.method == "concat":
            return torch.cat(ys, 1)
        out = ys[0]
        for y in ys[1:]:
            out = out + y
        return out


class _Node(nn.Module):
    """One node of the evolved graph. forward(list_of_parent_outputs) -> output."""

    def __init__(self, spec):
        super().__init__()
        self.cpp_name = spec["name"]
        self.layer_type = spec["cls"]
        self.origin = spec.get("origin", {})
        self.hparams = spec.get("hp", "")
        self.merge = _Merge(spec["merge"]) if spec.get("merge") else None

    def forward(self, inputs):
        x = self.merge(inputs) if self.merge is not None else inputs[0]
        return self.op(x)

    def extra_repr(self):
        return "cpp_name=%r" % self.cpp_name


class _Identity(_Node):
    def op(self, x):
        return x


class _Dense(_Node):
    def __init__(self, spec):
        super().__init__(spec)
        d = spec["dense"]
        self.linear = nn.Linear(d["in"], d["units"], bias=d["bias"])
        self.activation = d["activation"]
        self.dropout = d["dropout"]

    def op(self, x):
        y = _activation(self.linear(torch.flatten(x, 1)), self.activation)
        return F.dropout(y, self.dropout, self.training) if self.dropout > 0 else y


class _Conv2d(_Node):
    def __init__(self, spec):
        super().__init__(spec)
        c = spec["conv"]
        self.image_shape = tuple(c["image_shape"])
        self.conv = nn.Conv2d(c["in_channels"], c["filters"], tuple(c["kernel"]), stride=c["stride"],
                              padding=c["padding"], bias=c["bias"])
        self.pool = c["pool"]
        self.activation = c["activation"]
        self.dropout = c["dropout"]

    def op(self, x):
        if x.dim() == 2:
            x = x.reshape(x.shape[0], *self.image_shape)
        y = _activation(self.conv(x), self.activation)
        if self.pool:
            y = F.max_pool2d(y, 2, 2)
        return F.dropout(y, self.dropout, self.training) if self.dropout > 0 else y


class _CellSkip(nn.Module):
    """Gated memory-cell connection: c += sigmoid(gate([x, h_prev])) * proj(c_source)."""

    def __init__(self, in_features, units, source_units, proj):
        super().__init__()
        self.gate = nn.Linear(in_features, units)
        self.proj = nn.Linear(source_units, units, bias=False) if proj else None


class _LSTM(_Node):
    """LSTM stepped once per forward call; state kept in .h / .c (gate order i, f, g, o)."""

    def __init__(self, spec):
        super().__init__(spec)
        s = spec["lstm"]
        self.units = s["units"]
        self.in_features = s["in"]
        self.dropout = s["dropout"]
        z = self.in_features + self.units
        self.weight = nn.Parameter(torch.empty(4 * self.units, z).uniform_(-0.1, 0.1))
        self.bias = nn.Parameter(torch.zeros(4 * self.units))
        self.skip_sources = [k["source"] for k in s["skips"]]
        self.skips = nn.ModuleList([_CellSkip(z, self.units, k["source_units"], k["proj"]) for k in s["skips"]])
        self.h = None
        self.c = None
        self._sources = []

    def bind(self, nodes):
        self._sources = [nodes[k] for k in self.skip_sources]

    def reset_state(self, batch, device=None, dtype=None):
        self.h = torch.zeros(batch, self.units, device=device, dtype=dtype or self.weight.dtype)
        self.c = torch.zeros_like(self.h)

    def op(self, x):
        x = torch.flatten(x, 1)
        b = x.shape[0]
        if self.h is None or self.h.shape[0] != b or self.h.device != x.device:
            self.reset_state(b, x.device, x.dtype)
        xh = torch.cat([x, self.h], 1)
        i, f, g, o = F.linear(xh, self.weight, self.bias).split(self.units, 1)
        i, f, g, o = torch.sigmoid(i), torch.sigmoid(f), torch.tanh(g), torch.sigmoid(o)
        c = f * self.c + i * g
        for src, skip in zip(self._sources, self.skips):
            s = src.c  # current step if the source already ran, else previous step (feedback)
            if s is None or s.shape[0] != b:
                continue
            if skip.proj is not None:
                s = skip.proj(s)
            c = c + torch.sigmoid(skip.gate(xh)) * s
        h = o * torch.tanh(c)
        self.h, self.c = h, c
        return F.dropout(h, self.dropout, self.training) if self.dropout > 0 else h


_CLASSES = {"identity": _Identity, "dense": _Dense, "conv2d": _Conv2d, "lstm": _LSTM}


class CoDeepNEATModule(nn.Module):
    """Executable PyTorch version of an evolved network.

    forward(*inputs) runs ONE time step (inputs [B, ...] or unbatched) and returns
    the output heads (a tensor if there is one head, else a tuple). Recurrent
    state persists between calls: reset_state / detach_state / mask_state /
    get_state / set_state mirror the C++ Network API. Every graph node is a
    submodule in self.nodes, so standard PyTorch hooks work on any node.
    """

    SPEC = None

    def __init__(self, spec=None):
        super().__init__()
        spec = spec if spec is not None else type(self).SPEC
        self.spec = spec
        self.input_names = [i["name"] for i in spec["inputs"]]
        self.input_shapes = [tuple(i["shape"]) for i in spec["inputs"]]
        self.output_names = [o[5:] if o.startswith("head:") else o for o in spec["outputs"]]
        self.nodes = nn.ModuleDict()
        self._plan = []
        for n in spec["nodes"]:
            if n["kind"] == "input":
                self._plan.append((n["key"], n["slot"], None))
            else:
                self.nodes[n["key"]] = _CLASSES[n["cls"]](n)
                self._plan.append((n["key"], None, list(n["inputs"])))
        self._output_keys = list(spec["output_keys"])
        self._recurrent = [m for m in self.nodes.values() if isinstance(m, _LSTM)]
        for m in self._recurrent:
            m.bind(self.nodes)
        self.cpp_names = {n["key"]: n["name"] for n in spec["nodes"]}
        self.cpp_parameter_names = dict(spec["params"])  # C++ name -> state_dict key
        self._state_batch = None

    # ---- construction / weights -------------------------------------------
    @classmethod
    def from_pretrained(cls, weights_path, strict=True, map_location=None):
        model = cls()
        state = load_safetensors(weights_path)
        if map_location is not None:
            state = {k: v.to(map_location) for k, v in state.items()}
        model.load_state_dict(state, strict=strict)
        return model

    def save_pretrained(self, weights_path):
        """Save weights so cdn::import_torch_weights can load them back into C++."""
        save_safetensors(self.state_dict(), weights_path, {"format": "codeepneat-torch"})

    # ---- forward ------------------------------------------------------------
    def forward(self, *inputs):
        if len(inputs) == 1 and isinstance(inputs[0], dict):
            d = inputs[0]
            inputs = tuple(d[n] if n in d else d[n.split(":", 1)[-1]] for n in self.input_names)
        if len(inputs) != len(self.input_names):
            raise ValueError("expected %d inputs, got %d" % (len(self.input_names), len(inputs)))
        xs = []
        for x, shape, name in zip(inputs, self.input_shapes, self.input_names):
            if not torch.is_tensor(x):
                x = torch.as_tensor(x, dtype=torch.float32)
            if tuple(x.shape) == shape:
                x = x.unsqueeze(0)
            if tuple(x.shape[1:]) != shape:
                raise ValueError("input %s has shape %s, expected [B]+%s" % (name, tuple(x.shape), list(shape)))
            xs.append(x)
        batch = xs[0].shape[0]
        if self._recurrent and self._state_batch != batch:
            self.reset_state(batch, xs[0].device, xs[0].dtype)
        values = {}
        for key, slot, parents in self._plan:
            if slot is not None:
                values[key] = xs[slot]
            else:
                values[key] = self.nodes[key]([values[k] for k in parents])
        outs = tuple(values[k] for k in self._output_keys)
        return outs[0] if len(outs) == 1 else outs

    def named_forward(self, *inputs):
        outs = self.forward(*inputs)
        if not isinstance(outs, tuple):
            outs = (outs,)
        return dict(zip(self.output_names, outs))

    # ---- recurrent state ----------------------------------------------------
    @property
    def is_recurrent(self):
        return bool(self._recurrent)

    def reset_state(self, batch=1, device=None, dtype=None):
        for m in self._recurrent:
            m.reset_state(batch, device, dtype)
        self._state_batch = batch

    def detach_state(self):
        for m in self._recurrent:
            if m.h is not None:
                m.h, m.c = m.h.detach(), m.c.detach()

    def mask_state(self, keep):
        """Multiply state row b by keep[b] (0 resets that sequence)."""
        for m in self._recurrent:
            if m.h is None:
                continue
            k = torch.as_tensor(keep, dtype=m.h.dtype, device=m.h.device).reshape(-1, 1)
            m.h, m.c = m.h * k, m.c * k

    def get_state(self):
        state = {}
        for m in self._recurrent:
            state[m.cpp_name + ".h"] = m.h
            state[m.cpp_name + ".c"] = m.c
        return state

    def set_state(self, state):
        for m in self._recurrent:
            m.h, m.c = state[m.cpp_name + ".h"], state[m.cpp_name + ".c"]
            self._state_batch = m.h.shape[0]

    # ---- inspection -----------------------------------------------------------
    def node(self, cpp_name):
        """Submodule for a node given its C++ name (e.g. 'bp3/m17/n5')."""
        for key, name in self.cpp_names.items():
            if name == cpp_name and key in self.nodes:
                return self.nodes[key]
        raise KeyError(cpp_name)

    def summary(self):
        lines = ["%s: %d parameters%s" % (type(self).__name__, sum(p.numel() for p in self.parameters()),
                                          ", recurrent" if self.is_recurrent else "")]
        for n in self.spec["nodes"]:
            mod = self.nodes[n["key"]] if n["key"] in self.nodes else None
            count = sum(p.numel() for p in mod.parameters()) if mod is not None else 0
            lines.append("  %-28s %-9s %-14s %8d  %s" % (n["name"], n["cls"], n["out_shape"], count, n.get("hp", "")))
        return "\n".join(lines)
)PY";

const char* kTorchTail = R"PY(

class __CLASS__(CoDeepNEATModule):
    """__DOC__"""

    SPEC = SPEC


if __name__ == "__main__":
    import os
    import sys

    path = sys.argv[1] if len(sys.argv) > 1 else os.path.splitext(os.path.abspath(__file__))[0] + ".safetensors"
    model = __CLASS__.from_pretrained(path) if os.path.exists(path) else __CLASS__()
    model.eval()
    print(model.summary())
    with torch.no_grad():
        out = model(*[torch.zeros(1, *s) for s in model.input_shapes])
    print("outputs:", out)
)PY";

void replace_all(std::string& s, const std::string& from, const std::string& to) {
  for (size_t pos = 0; (pos = s.find(from, pos)) != std::string::npos; pos += to.size()) s.replace(pos, from.size(), to);
}

bool valid_identifier(const std::string& s) {
  if (s.empty() || std::isdigit(static_cast<unsigned char>(s[0]))) return false;
  for (char c : s)
    if (!std::isalnum(static_cast<unsigned char>(c)) && c != '_') return false;
  return true;
}

}  // namespace

// ============================================================================
// Public API
// ============================================================================

std::map<std::string, std::string> torch_parameter_names(const Network& net) { return make_plan(net).params; }

std::string torch_module_source(const Network& net, const TorchExportOptions& options) {
  if (!valid_identifier(options.class_name))
    throw std::invalid_argument("export_torch: class_name must be a Python identifier");
  ExportPlan plan = make_plan(net);
  std::string doc = "Evolved network exported from C++ CoDeepNEAT (" + std::to_string(net.num_parameters()) +
                    " parameters, " + std::to_string(net.nodes().size()) + " nodes" +
                    (net.is_recurrent() ? ", recurrent" : "") + ").";
  if (!options.description.empty()) doc += "\n\n    " + options.description;
  replace_all(doc, "\\", "\\\\");
  replace_all(doc, "\"\"\"", "\\\"\\\"\\\"");
  std::string src = "\"\"\"" + doc + "\"\"\"\n";
  src += kTorchRuntime;
  src += "\n\nSPEC = json.loads(r'''" + plan.spec_json + "''')\n";
  std::string tail = kTorchTail;
  src += tail;
  replace_all(src, "__CLASS__", options.class_name);
  replace_all(src, "__DOC__", doc);
  return src;
}

void save_torch_weights(const Network& net, const std::string& path) {
  ExportPlan plan = make_plan(net);
  std::vector<NamedTensor> tensors;
  for (const auto& p : net.named_parameters()) tensors.push_back({plan.params.at(p.name), p.tensor});
  save_safetensors(tensors, path, {{"format", "codeepneat-torch"}});
}

void export_torch(const Network& net, const std::string& path_prefix, const TorchExportOptions& options) {
  std::string src = torch_module_source(net, options);
  std::ofstream f(path_prefix + ".py", std::ios::binary);
  if (!f) throw std::runtime_error("export_torch: cannot open " + path_prefix + ".py");
  f << src;
  if (!f) throw std::runtime_error("export_torch: write failed for " + path_prefix + ".py");
  save_torch_weights(net, path_prefix + ".safetensors");
}

void import_torch_weights(Network& net, const std::string& path, bool strict) {
  std::map<std::string, std::string> to_torch = torch_parameter_names(net);
  std::map<std::string, Tensor> by_key;
  for (auto& t : load_safetensors(path)) by_key[t.name] = t.tensor;
  for (auto p : net.named_parameters()) {
    auto it = by_key.find(to_torch.at(p.name));
    if (it == by_key.end()) {
      if (strict) throw std::invalid_argument("import_torch_weights: missing " + to_torch.at(p.name));
      continue;
    }
    if (it->second.shape() != p.tensor.shape()) {
      if (strict)
        throw std::invalid_argument("import_torch_weights: shape mismatch for " + it->first + ": " +
                                    shape_str(it->second.shape()) + " vs " + shape_str(p.tensor.shape()));
      continue;
    }
    p.tensor.copy_from(it->second);
  }
}

void export_torch_reference(Network& net, const std::vector<std::vector<Tensor>>& steps, const std::string& path) {
  if (steps.empty()) throw std::invalid_argument("export_torch_reference: no steps");
  ExportPlan plan = make_plan(net);
  bool was_training = net.is_training();
  net.eval();
  net.zero_grad();
  const Tensor& x0 = steps[0].at(0);
  int batch = x0.dim() == static_cast<int>(net.input_shapes()[0].size()) ? 1 : x0.shape()[0];
  net.reset_state(batch);
  std::vector<NamedTensor> rec;
  Tensor loss = Tensor::scalar(0.0f);
  for (size_t t = 0; t < steps.size(); ++t) {
    std::vector<Tensor> outs = net.forward(steps[t]);
    for (size_t k = 0; k < steps[t].size(); ++k)
      rec.push_back({"step" + std::to_string(t) + ".input" + std::to_string(k), steps[t][k].detach()});
    for (size_t k = 0; k < outs.size(); ++k) {
      rec.push_back({"step" + std::to_string(t) + ".output" + std::to_string(k), outs[k].detach()});
      loss = add(loss, mul(sum(square(outs[k])), 0.5f));
    }
  }
  if (loss.requires_grad()) loss.backward();
  for (const auto& p : net.named_parameters())
    rec.push_back({"grad." + plan.params.at(p.name), p.tensor.has_grad() ? p.tensor.grad() : Tensor::zeros(p.tensor.shape())});
  std::map<std::string, std::string> meta = {{"format", "codeepneat-torch-reference"},
                                             {"steps", std::to_string(steps.size())},
                                             {"inputs", std::to_string(steps[0].size())},
                                             {"outputs", std::to_string(net.output_names().size())},
                                             {"loss", json_num(loss.item())}};
  save_safetensors(rec, path, meta);
  net.zero_grad();
  net.reset_state(batch);
  net.train(was_training);
}

}  // namespace cdn
