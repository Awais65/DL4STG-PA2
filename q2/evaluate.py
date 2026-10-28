"""Test-split evaluation: per-horizon MSE, E^(5), E^(2), C_T, E_T and inference time.

    python -m q2.evaluate --table 2          # model A vs Blank and Copy-last
    python -m q2.evaluate --table 3          # mirror / rotation   (needs predictions/Q2.5.md committed)
    python -m q2.evaluate --table 4          # 2x / 3x speed       (needs predictions/Q2.5.md committed)
    python -m q2.evaluate --table 5          # models A-D          (needs predictions/Q2.11.md committed)
    python -m q2.evaluate --table 6          # symmetrised B and D (needs predictions/Q2.11.md committed)

Every metric uses t=1 (handout, eq. 5 and 6). A forecaster is any callable X (B, 5, 1, 64, 64)
-> Y_hat (B, 5, 1, 64, 64); learned models, the two reference predictors and the symmetrised
wrapper all share that interface.
"""
import argparse
import json
import subprocess
import time
from pathlib import Path

import numpy as np
import torch

from q2.data import G, M, R, batch_window, g_apply, g_inverse, load_moving_mnist
from q2.models import ConvLSTMForecaster, count_params

SEEDS = (0, 1)
TRANSFORMS = {"M": M, "R": R}


# ---------------------------------------------------------------- forecasters

def blank(X):
    return torch.zeros_like(X)


def copy_last(X):
    return X[:, -1:].expand_as(X).clone()


class Symmetrised(torch.nn.Module):
    """f_bar(X) = 1/8 sum_g g^-1(f(g(X))) (eq. 7)."""

    def __init__(self, f):
        super().__init__()
        self.f = f

    def forward(self, X):
        return torch.stack([g_inverse(self.f(g_apply(X, a, m)), a, m) for a, m in G]).mean(0)


def load_model(path, dev):
    ck = torch.load(path, map_location=dev)
    model = ConvLSTMForecaster(**ck["arch"]).to(dev)
    model.load_state_dict(ck["state_dict"])
    return model.eval()


# ---------------------------------------------------------------- metrics

@torch.no_grad()
def predict(f, X, bs=100):
    return torch.cat([f(X[i : i + bs]) for i in range(0, len(X), bs)])


def horizon_mse(pred, Y):
    """(N, 5, 1, H, W) pair -> (N, 5): MSE over the pixels of each video and horizon."""
    return ((pred - Y) ** 2).mean(dim=(2, 3, 4))


@torch.no_grad()
def speed_errors(f, test, k):
    """Per-video, per-horizon MSE at speed k with t=1. Horizons whose target frame does not exist are NaN."""
    X, Y, mask = batch_window(test, 1, k)
    mse = horizon_mse(predict(f, X), Y)
    mse[:, mask == 0] = float("nan")
    return mse


@torch.no_grad()
def transform_errors(f, test, T):
    """Per-video C_T = MSE(f(TX), T f(X)) and E_T = MSE(f(TX), T Y), at 1x with t=1 (eq. 6)."""
    X, Y, _ = batch_window(test, 1, 1)
    fX, fTX = predict(f, X), predict(f, T(X))
    C = horizon_mse(fTX, T(fX)).mean(1)
    E = horizon_mse(fTX, T(Y)).mean(1)
    return C, E


@torch.no_grad()
def inference_time(f, test_cpu, dev, bs=32):
    """Wall-clock time to predict the 1,000 1x, t=1 test windows at batch size 32.

    The clock runs only from when each batch is on the device until its prediction is finished,
    so the host-to-device copy is excluded. One untimed warm-up batch runs first (CUDA kernel
    selection and memory allocation would otherwise be charged to the first batch).
    """
    X, _, _ = batch_window(test_cpu, 1, 1)
    sync = torch.cuda.synchronize if dev.type == "cuda" else (lambda: None)
    f(X[:bs].to(dev))
    sync()
    total = 0.0
    for i in range(0, len(X), bs):
        xb = X[i : i + bs].to(dev)
        sync()
        t0 = time.perf_counter()
        f(xb)
        sync()
        total += time.perf_counter() - t0
    return total


def evaluate(f, test, test_cpu, dev, cols):
    """The requested Table 2-6 metrics for one forecaster, as means over the 1,000 test videos.

    Only what `cols` asks for is computed, so Table 2 never runs a transformed or faster test.
    """
    out = {}
    for k in (1, 2, 3):
        if k == 1 or any(c.endswith(f"_{k}x") for c in cols):
            e = speed_errors(f, test, k)
            out[f"E2_{k}x"] = e[:, :2].mean().item()
            if k == 1:
                out.update({f"MSE{h + 1}": e[:, h].mean().item() for h in range(5)})
            if not torch.isnan(e).any():
                out[f"E5_{k}x"] = e.mean().item()
    out["E5"] = out["E5_1x"]
    for name, T in TRANSFORMS.items():
        if f"C_{name}" in cols or f"E_{name}" in cols:
            C, E = transform_errors(f, test, T)
            out[f"C_{name}"], out[f"E_{name}"] = C.mean().item(), E.mean().item()
    if "inference_time" in cols:
        out["inference_time"] = inference_time(f, test_cpu, dev)
    return {c: out[c] for c in cols}


# ---------------------------------------------------------------- tables

def committed(path):
    """True if `path` is in a commit (so its commit time predates any experiment run after it)."""
    r = subprocess.run(["git", "log", "-1", "--format=%cI", "--", path], capture_output=True, text=True)
    return bool(r.stdout.strip())


def require_prediction(q):
    path = f"predictions/{q}.md"
    if not committed(path):
        raise SystemExit(f"{path} is not committed. Write and commit your {q} prediction before this experiment.")


def fmt(x):
    return f"{x:.4e}"


def mean_pair(vals):
    """'mean (seed 0, seed 1)' as the handout asks for every learned-model entry."""
    return f"{fmt(np.mean(vals))} ({fmt(vals[0])}, {fmt(vals[1])})"


def learned_rows(names, cols, test, test_cpu, dev, ckpt_dir, wrap=False, label=None):
    rows, raw = [], {}
    for name in names:
        res, n_params = [], None
        for s in SEEDS:
            model = load_model(Path(ckpt_dir) / f"{name}_s{s}.pt", dev)
            n_params = count_params(model)
            f = Symmetrised(model) if wrap else model
            res.append(evaluate(f, test, test_cpu, dev, cols))
        shown = (label or "{}").format(name)
        raw[shown] = dict(n_params=n_params, seeds=res)
        cells = [mean_pair([r[c] for r in res]) for c in cols]
        rows.append([shown, *cells, f"{n_params:,}"])
    return rows, raw


def reference_rows(cols, test, test_cpu, dev):
    rows, raw = [], {}
    for shown, f in (("Blank", blank), ("Copy-last", copy_last)):
        r = evaluate(f, test, test_cpu, dev, cols)
        raw[shown] = r
        rows.append([shown, *[fmt(r[c]) for c in cols], "0"])
    return rows, raw


def train_times(names, rows):
    """Append mean training time per epoch (mean over seeds) from results/train/*.json."""
    for row, name in zip(rows, names):
        t = [json.loads(Path(f"results/train/{name}_s{s}.json").read_text())["mean_epoch_time"] for s in SEEDS]
        row.append(f"{np.mean(t):.1f} s ({t[0]:.1f}, {t[1]:.1f})")


def markdown(header, rows):
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    return "\n".join(lines + ["| " + " | ".join(map(str, r)) + " |" for r in rows]) + "\n"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--table", type=int, required=True, choices=[2, 3, 4, 5, 6])
    p.add_argument("--ckpt", default="checkpoints")
    p.add_argument("--final", default=None, help="Table 6: 'DB' if the final model wraps D+B instead of D")
    args = p.parse_args()

    if args.table in (3, 4):
        require_prediction("Q2.5")
    if args.table in (5, 6):
        require_prediction("Q2.11")

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, _, test_u8 = load_moving_mnist()
    test_cpu = test_u8.float() / 255.0
    test = test_cpu.to(dev)
    hw = torch.cuda.get_device_name(dev) if dev.type == "cuda" else "cpu"

    t = args.table
    if t == 2:
        cols = ["MSE1", "MSE2", "MSE3", "MSE4", "MSE5", "E5"]
        rows, raw = learned_rows(["A"], cols, test, test_cpu, dev, args.ckpt)
        ref, raw_ref = reference_rows(cols, test, test_cpu, dev)
        header = ["Model", *[f"MSE_{h}" for h in range(1, 6)], "E^(5)", "Params"]
        rows += ref
    elif t == 3:
        cols = ["E5", "C_M", "E_M", "C_R", "E_R"]
        rows, raw = learned_rows(["A"], cols, test, test_cpu, dev, args.ckpt)
        ref, raw_ref = reference_rows(cols, test, test_cpu, dev)
        header = ["Model", "E^(5)", "C_M", "E_M", "C_R", "E_R", "Params"]
        rows += ref
    elif t == 4:
        cols = ["E2_1x", "E2_2x", "E2_3x", "E5_1x", "E5_2x"]
        rows, raw = learned_rows(["A"], cols, test, test_cpu, dev, args.ckpt)
        ref, raw_ref = reference_rows(cols, test, test_cpu, dev)
        rows += [r for r in ref if r[0] == "Copy-last"]
        header = ["Model", "E^(2) 1x", "E^(2) 2x", "E^(2) 3x", "E^(5) 1x", "E^(5) 2x", "Params"]
    else:
        cols = ["E5", "C_M", "E_M", "C_R", "E_R", "E2_1x", "E2_2x", "E2_3x", "inference_time"]
        if t == 5:
            names, wrap, label = ["A", "B", "C", "D"], False, None
        else:
            names, wrap, label = ["B", "D"] + ([args.final] if args.final else []), True, "sym. {}"
        rows, raw = learned_rows(names, cols, test, test_cpu, dev, args.ckpt, wrap=wrap, label=label)
        train_times(names, rows)
        raw_ref = {}
        header = ["Model", "E^(5)", "C_M", "E_M", "C_R", "E_R", "E^(2) 1x", "E^(2) 2x", "E^(2) 3x",
                  "Inference (s)", "Params", "Train (s/epoch)"]

    note = (f"Test split (videos 9000-9999), t=1. Learned models: mean (seed 0, seed 1). Hardware: {hw}.\n\n")
    md = f"## Table {t}\n\n{note}{markdown(header, rows)}"
    Path("results").mkdir(exist_ok=True)
    Path(f"results/table{t}.md").write_text(md)
    Path(f"results/table{t}.json").write_text(json.dumps(dict(hardware=hw, learned=raw, reference=raw_ref), indent=2))
    print(md)


if __name__ == "__main__":
    main()
