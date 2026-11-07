"""Check that PyTorch modules written by codeepneat::export_torch reproduce the
C++ networks they came from.

For every case <prefix> it needs <prefix>.py, <prefix>.safetensors and
<prefix>.reference.safetensors (from cdn::export_torch_reference). It replays the
recorded time steps from a zero recurrent state and compares the outputs, and
the parameter gradients of loss = sum_t sum_heads 0.5*|y|^2, against C++.

With --write-back, it also takes one SGD step in PyTorch and saves
<prefix>.torch_trained.safetensors (via model.save_pretrained) and
<prefix>.torch_outputs.safetensors (outputs of the updated model on step 0), so
the C++ side can check cdn::import_torch_weights.

usage: python verify_torch_export.py PREFIX [PREFIX ...] [--write-back] [--rtol R] [--atol A]
       python verify_torch_export.py --dir DIR [--write-back]
"""
import argparse
import glob
import importlib.util
import json
import os
import struct
import sys

import torch


def load_module(path):
    name = "cdn_export_" + os.path.splitext(os.path.basename(path))[0]
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def read_metadata(path):
    with open(path, "rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        header = json.loads(f.read(n).decode("utf-8"))
    return header.get("__metadata__", {})


def model_class(mod):
    for value in vars(mod).values():
        if isinstance(value, type) and issubclass(value, mod.CoDeepNEATModule) and value is not mod.CoDeepNEATModule:
            return value
    raise RuntimeError("no exported model class found")


def as_tuple(x):
    return x if isinstance(x, tuple) else (x,)


def compare(name, got, want, rtol, atol):
    if tuple(got.shape) != tuple(want.shape):
        return "%s: shape %s vs %s" % (name, tuple(got.shape), tuple(want.shape)), float("inf")
    diff = (got - want).abs()
    worst = diff.max().item() if diff.numel() else 0.0
    if not torch.allclose(got, want, rtol=rtol, atol=atol):
        return "%s: max abs diff %.3g (max |ref| %.3g)" % (name, worst, want.abs().max().item()), worst
    return None, worst


def run_steps(model, ref, steps, n_in):
    first = ref["step0.input0"]
    batch = 1 if first.dim() == len(model.input_shapes[0]) else first.shape[0]
    model.reset_state(batch)
    outputs = []
    for t in range(steps):
        outputs.append(as_tuple(model(*[ref["step%d.input%d" % (t, k)] for k in range(n_in)])))
    return outputs, batch


def verify(prefix, rtol, atol, write_back):
    mod = load_module(prefix + ".py")
    cls = model_class(mod)
    model = cls.from_pretrained(prefix + ".safetensors")
    model.eval()
    ref_path = prefix + ".reference.safetensors"
    ref = mod.load_safetensors(ref_path)
    meta = read_metadata(ref_path)
    steps, n_in, n_out = int(meta["steps"]), int(meta["inputs"]), int(meta["outputs"])

    problems, worst_out, worst_grad = [], 0.0, 0.0
    outputs, batch = run_steps(model, ref, steps, n_in)
    loss = 0.0
    for t, outs in enumerate(outputs):
        if len(outs) != n_out:
            problems.append("step %d: %d outputs, expected %d" % (t, len(outs), n_out))
            continue
        for k, y in enumerate(outs):
            p, w = compare("step%d.output%d" % (t, k), y.detach(), ref["step%d.output%d" % (t, k)], rtol, atol)
            worst_out = max(worst_out, w)
            if p:
                problems.append(p)
            loss = loss + 0.5 * (y ** 2).sum()
    model.zero_grad()
    loss.backward()
    for name, param in model.named_parameters():
        key = "grad." + name
        if key not in ref:
            problems.append("missing reference gradient for " + name)
            continue
        got = param.grad if param.grad is not None else torch.zeros_like(param)
        p, w = compare(key, got, ref[key], rtol * 10, atol * 10)
        worst_grad = max(worst_grad, w)
        if p:
            problems.append(p)
    expected = set(k[5:] for k in ref if k.startswith("grad."))
    missing = expected - set(n for n, _ in model.named_parameters())
    if missing:
        problems.append("parameters missing in PyTorch module: %s" % sorted(missing))

    if write_back:
        with torch.no_grad():
            for p in model.parameters():
                if p.grad is not None:
                    p -= 0.05 * p.grad
        model.save_pretrained(prefix + ".torch_trained.safetensors")
        model.reset_state(batch)
        with torch.no_grad():
            outs = as_tuple(model(*[ref["step0.input%d" % k] for k in range(n_in)]))
        mod.save_safetensors({"output%d" % k: y for k, y in enumerate(outs)}, prefix + ".torch_outputs.safetensors")

    status = "OK  " if not problems else "FAIL"
    n_params = sum(p.numel() for p in model.parameters())
    print("%s %-40s %7d params  %d steps  max|dy|=%.2e  max|dgrad|=%.2e%s"
          % (status, os.path.basename(prefix), n_params, steps, worst_out, worst_grad,
             "  (recurrent)" if model.is_recurrent else ""))
    for p in problems[:10]:
        print("       " + p)
    return not problems


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("prefixes", nargs="*")
    ap.add_argument("--dir", help="verify every <prefix>.reference.safetensors in this directory")
    ap.add_argument("--write-back", action="store_true")
    ap.add_argument("--rtol", type=float, default=1e-4)
    ap.add_argument("--atol", type=float, default=1e-5)
    args = ap.parse_args()
    prefixes = list(args.prefixes)
    if args.dir:
        prefixes += sorted(p[: -len(".reference.safetensors")]
                           for p in glob.glob(os.path.join(args.dir, "*.reference.safetensors")))
    if not prefixes:
        ap.error("nothing to verify")
    torch.manual_seed(0)
    ok = all([verify(p, args.rtol, args.atol, args.write_back) for p in prefixes])
    print("all exports match" if ok else "MISMATCH")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
