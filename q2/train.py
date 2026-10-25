"""Train models A-D under the handout's protocol.

    python -m q2.train --model A --seed 0
    python -m q2.train --model D --seed 0 --k 5          # model D candidates: --k / --dilation / --patch

Adam 1e-3, no schedule, batch 32, 30 epochs, final-epoch weights kept. One epoch is a
pass over the 8,000 training videos in shuffled order with one random window each.
The seed controls initialisation, shuffling, window sampling and augmentation sampling.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from q2.data import G, MAX_T_NORMAL, MAX_T_SPEED, N_FRAMES, N_IN, N_OUT, g_apply, load_moving_mnist
from q2.models import ConvLSTMForecaster, count_params

# Training speeds and symmetry augmentation per model (handout, "What Can Data Fix?").
MODELS = {
    "A": dict(speed_aug=False, sym_aug=False),
    "B": dict(speed_aug=False, sym_aug=True),
    "C": dict(speed_aug=True, sym_aug=False),
    "D": dict(speed_aug=True, sym_aug=False),
    "DB": dict(speed_aug=True, sym_aug=True),
}


def build_model(name, k=3, dilation=1, patch=4):
    if name in ("A", "B", "C"):
        return ConvLSTMForecaster()
    return ConvLSTMForecaster(k=k, dilation=dilation, patch=patch)


def sample_windows(n, speed_aug, gen):
    """One (t, k) per video, drawn under the start-frame rules. Returns 1-based t and k as int64 tensors."""
    if not speed_aug:
        k = torch.ones(n, dtype=torch.long)
        t = torch.randint(1, MAX_T_NORMAL + 1, (n,), generator=gen)
        return t, k
    k = torch.randint(1, 4, (n,), generator=gen)
    t_max = torch.tensor([MAX_T_SPEED[1], MAX_T_SPEED[2], MAX_T_SPEED[3]])[k - 1]
    t = 1 + (torch.rand(n, generator=gen) * t_max).long()
    return t, k


def gather_windows(videos, t, k):
    """videos (B, 20, 1, H, W) float, per-video t, k (B,) -> X, Y (B, 5, 1, H, W), mask (B, 5)."""
    B = videos.shape[0]
    steps = torch.arange(N_IN + N_OUT, device=videos.device)
    idx = (t - 1).to(videos.device)[:, None] + k.to(videos.device)[:, None] * steps  # 0-based frame indices
    mask = (idx[:, N_IN:] < N_FRAMES).float()
    assert (idx[:, :N_IN] < N_FRAMES).all()
    rows = torch.arange(B, device=videos.device)[:, None]
    X = videos[rows, idx[:, :N_IN]]
    Y = videos[rows, idx[:, N_IN:].clamp(max=N_FRAMES - 1)] * mask[:, :, None, None, None]
    return X, Y, mask


def apply_per_sample_g(X, Y, g_idx):
    """Apply symmetry G[g_idx[b]] to every frame of X[b] and Y[b]."""
    X, Y = X.clone(), Y.clone()
    for gi, (a, m) in enumerate(G):
        sel = (g_idx == gi).nonzero(as_tuple=True)[0]
        if len(sel):
            X[sel] = g_apply(X[sel], a, m)
            Y[sel] = g_apply(Y[sel], a, m)
    return X, Y


def masked_mse(pred, Y, mask):
    """Squared error averaged over pixels and available target frames of each video, then over videos."""
    per_frame = ((pred - Y) ** 2).mean(dim=(2, 3, 4))               # (B, 5)
    per_video = (per_frame * mask).sum(1) / mask.sum(1)
    return per_video.mean()


@torch.no_grad()
def val_e5(model, val, bs=100):
    """E^(5) at 1x, t=1 on the validation split (monitoring only)."""
    model.eval()
    errs = []
    for i in range(0, len(val), bs):
        v = val[i : i + bs].float() / 255.0
        X, Y = v[:, :N_IN], v[:, N_IN : N_IN + N_OUT]
        errs.append(((model(X) - Y) ** 2).mean(dim=(1, 2, 3, 4)))
    model.train()
    return torch.cat(errs).mean().item()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, choices=list(MODELS))
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--bs", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--k", type=int, default=3, help="model D kernel size")
    p.add_argument("--dilation", type=int, default=1, help="model D dilation")
    p.add_argument("--patch", type=int, default=4, help="model D patch size")
    p.add_argument("--tag", default="", help="suffix for checkpoint name (model D candidates)")
    p.add_argument("--out", default="checkpoints")
    args = p.parse_args()

    cfg = MODELS[args.model]
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    gen = torch.Generator().manual_seed(args.seed)

    train, val, _ = load_moving_mnist()
    train, val = train.to(dev), val.to(dev)  # uint8 on device, ~0.7 GB

    model = build_model(args.model, args.k, args.dilation, args.patch).to(dev)
    n_params = count_params(model)
    if args.model in ("D", "DB"):
        assert n_params <= 721_176, f"model D has {n_params:,} parameters (max 721,176)"
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    name = f"{args.model}{args.tag}_s{args.seed}"
    hw = torch.cuda.get_device_name(dev) if dev.type == "cuda" else "cpu"
    print(f"{name}: {n_params:,} params on {hw}, cfg={cfg}")

    history = []
    for epoch in range(1, args.epochs + 1):
        if dev.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        perm = torch.randperm(len(train), generator=gen)
        t_all, k_all = sample_windows(len(train), cfg["speed_aug"], gen)
        g_all = torch.randint(0, len(G), (len(train),), generator=gen)
        running, n_seen = 0.0, 0
        for i in range(0, len(train), args.bs):
            b = perm[i : i + args.bs]
            v = train[b.to(dev)].float() / 255.0
            X, Y, mask = gather_windows(v, t_all[b], k_all[b])
            if cfg["sym_aug"]:
                X, Y = apply_per_sample_g(X, Y, g_all[b].to(dev))
            loss = masked_mse(model(X), Y, mask)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            running += loss.item() * len(b)
            n_seen += len(b)
        if dev.type == "cuda":
            torch.cuda.synchronize()
        epoch_time = time.perf_counter() - t0
        ve = val_e5(model, val)
        history.append(dict(epoch=epoch, train_loss=running / n_seen, val_e5=ve, epoch_time=epoch_time))
        print(f"epoch {epoch:2d}  loss {running / n_seen:.4e}  val E5 {ve:.4e}  {epoch_time:.1f}s", flush=True)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    arch = dict(k=args.k, dilation=args.dilation, patch=args.patch) if args.model in ("D", "DB") else {}
    torch.save(dict(state_dict=model.state_dict(), model=args.model, arch=arch, seed=args.seed), out / f"{name}.pt")
    log = dict(
        model=args.model, tag=args.tag, seed=args.seed, arch=arch, n_params=n_params, hardware=hw,
        mean_epoch_time=float(np.mean([h["epoch_time"] for h in history])), history=history,
    )
    Path("results/train").mkdir(parents=True, exist_ok=True)
    Path(f"results/train/{name}.json").write_text(json.dumps(log, indent=2))
    print(f"saved {out / f'{name}.pt'}; mean epoch time {log['mean_epoch_time']:.1f}s")


if __name__ == "__main__":
    main()
