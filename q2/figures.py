"""Figures 2-6 and GIFs (seed-0 models, test split, t=1).

    python -m q2.figures --fig 2
    python -m q2.figures --fig 3      # also Figure 4 and the Figure 3 GIFs   (needs predictions/Q2.5.md committed)
    python -m q2.figures --fig 5      #                                       (needs predictions/Q2.5.md committed)
    python -m q2.figures --fig 6      # final model on the Figure 3/5 videos  (needs predictions/Q2.11.md committed)
    python -m q2.figures --fig 6 --final DB    # if the final model wraps D+B
"""
import argparse
import json
from pathlib import Path

import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy import ndimage

from q2.data import batch_window, load_moving_mnist
from q2.evaluate import TRANSFORMS, Symmetrised, load_model, predict, require_prediction, transform_errors

INTENSITY_MIN = 0.2


def overlap_score(frames):
    """Smallest bright-pixel area over the frames, relative to the largest (low = digits overlap)."""
    area = (frames > INTENSITY_MIN).flatten(1).sum(1).float()
    return (area.min() / area.max()).item()


def frame_grid(rows, labels, path, title=None, col_labels=None):
    """rows: list of (10, 64, 64) arrays (or (n, 64, 64)) drawn one per row; NaN frames are left blank."""
    n_cols = max(len(r) for r in rows)
    fig, axes = plt.subplots(len(rows), n_cols, figsize=(1.2 * n_cols, 1.25 * len(rows)), squeeze=False)
    for i, (frames, lab) in enumerate(zip(rows, labels)):
        for j in range(n_cols):
            ax = axes[i, j]
            ax.set_xticks([]), ax.set_yticks([])
            if j < len(frames) and frames[j] is not None:
                ax.imshow(frames[j], cmap="gray", vmin=0, vmax=1)
            else:
                for sp in ax.spines.values():
                    sp.set_visible(False)
            if j == 0:
                ax.set_ylabel(lab, fontsize=8)
            if i == 0 and col_labels:
                ax.set_title(col_labels[j], fontsize=8)
    if title:
        fig.suptitle(title, fontsize=9)
    fig.tight_layout()
    fig.savefig(f"{path}.png", dpi=150)
    fig.savefig(f"{path}.pdf")
    plt.close(fig)


def fig2(test, dev):
    """Input, prediction and ground truth for three test videos, one with overlapping digits."""
    model = load_model("checkpoints/A_s0.pt", dev)
    X, Y, _ = batch_window(test, 1, 1)
    scores = [overlap_score(torch.cat([X[i], Y[i]])[:, 0]) for i in range(len(test))]
    overlap = int(np.argmin(scores))
    clean = [i for i in range(len(test)) if scores[i] > 0.95][:2]
    ids = [clean[0], clean[1], overlap]
    pred = predict(model, X[ids].to(dev)).cpu()

    rows, labels = [], []
    for n, i in enumerate(ids):
        tag = f"video {9000 + i}" + (" (overlap)" if i == overlap else "")
        rows += [list(X[i, :, 0].numpy()) + [None] * 5,
                 [None] * 5 + list(pred[n, :, 0].numpy()),
                 [None] * 5 + list(Y[i, :, 0].numpy())]
        labels += [f"{tag}\ninput", "prediction", "ground truth"]
    cols = [f"x{j}" for j in range(1, 6)] + [f"h={h}" for h in range(1, 6)]
    frame_grid(rows, labels, "figures/fig2_examples", "Model A (seed 0), 1x, t=1", cols)
    info = dict(videos=[9000 + i for i in ids], overlap_video=9000 + overlap,
                overlap_score={9000 + i: scores[i] for i in ids})
    Path("results/fig2_videos.json").write_text(json.dumps(info, indent=2))
    print(info)


def commuting_rows(f, X, Y, T, dev):
    """The four Figure 3 rows for one video: X | f(X), T(X) | T(Y), T(X) | f(T(X)), T(X) | T(f(X))."""
    X, Y = X[None].to(dev), Y[None].to(dev)
    fX, fTX = f(X), f(T(X))
    pairs = [(X, fX), (T(X), T(Y)), (T(X), fTX), (T(X), T(fX))]
    return [list(torch.cat([a, b], 1)[0, :, 0].cpu().numpy()) for a, b in pairs]


def commuting_gif(rows, T_name, path):
    """Animated GIF stepping through the five forecast horizons, keeping the same four rows."""
    labels = ["f(X)", f"{T_name}(Y)", f"f({T_name}(X))", f"{T_name}(f(X))"]
    frames = []
    for h in range(5):
        fig, axes = plt.subplots(4, 1, figsize=(1.7, 6))
        for ax, r, lab in zip(axes, rows, labels):
            ax.imshow(r[5 + h], cmap="gray", vmin=0, vmax=1)
            ax.set_xticks([]), ax.set_yticks([])
            ax.set_ylabel(lab, fontsize=8)
        fig.suptitle(f"horizon h={h + 1}", fontsize=8)
        fig.tight_layout()
        fig.canvas.draw()
        frames.append(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy())
        plt.close(fig)
    imageio.mimsave(path, frames, duration=0.6, loop=0)


def fig3_panels(f, test, dev, ids_by_T, prefix, title):
    """One Figure 3 panel (and GIF) per (transformation, video). Returns the GIF paths."""
    X, Y, _ = batch_window(test, 1, 1)
    gifs = []
    for T_name, ids in ids_by_T.items():
        T = TRANSFORMS[T_name]
        for rank, i in enumerate(ids, 1):
            rows = commuting_rows(f, X[i], Y[i], T, dev)
            labels = ["(1) X | f(X)", f"(2) {T_name}(X) | {T_name}(Y)",
                      f"(3) {T_name}(X) | f({T_name}(X))", f"(4) {T_name}(X) | {T_name}(f(X))"]
            name = f"{prefix}_{T_name}{rank}_v{9000 + i}"
            cols = [f"x{j}" for j in range(1, 6)] + [f"h={h}" for h in range(1, 6)]
            frame_grid(rows, labels, f"figures/{name}", f"{title}, T={T_name}, video {9000 + i}", cols)
            Path("figures/gifs").mkdir(parents=True, exist_ok=True)
            gif = f"figures/gifs/{name}.gif"
            commuting_gif(rows, T_name, gif)
            gifs.append(gif)
    return gifs


def fig3_and_4(test, dev):
    """Figure 3 (top-2 per-video C_M and C_R, seed-0 model A), its GIFs, and Figure 4 (per-video histograms)."""
    model = load_model("checkpoints/A_s0.pt", dev)
    per_video, ids_by_T = {}, {}
    for T_name, T in TRANSFORMS.items():
        C, _ = transform_errors(model, test.to(dev), T)
        per_video[T_name] = C.cpu().numpy()
        ids_by_T[T_name] = [int(i) for i in np.argsort(-per_video[T_name])[:2]]
    gifs = fig3_panels(model, test, dev, ids_by_T, "fig3", "Model A (seed 0)")

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
    for panel, (ax, (T_name, C)) in enumerate(zip(axes, per_video.items())):
        bins = np.logspace(np.log10(C.min()), np.log10(C.max()), 50)
        ax.hist(C, bins=bins, edgecolor="white")
        ax.set_xscale("log")
        top = ax.get_ylim()[1]
        for i in ids_by_T[T_name]:
            ax.axvline(C[i], color="C3", ls="--", lw=1)
            ax.annotate(f"{9000 + i}", (C[i], top * 0.95), rotation=90, fontsize=7, ha="right", va="top", color="C3")
        ax.set(xlabel=f"per-video $C_{T_name}$ (log scale)", ylabel="number of test videos",
               title=f"({'ab'[panel]}) $C_{T_name}$, model A seed 0; dashed = Figure 3 videos")
    fig.tight_layout()
    fig.savefig("figures/fig4_consistency_hist.png", dpi=200)
    fig.savefig("figures/fig4_consistency_hist.pdf")
    plt.close(fig)

    info = dict(ids={T: [9000 + i for i in ids] for T, ids in ids_by_T.items()},
                per_video_C={T: {9000 + i: float(per_video[T][i]) for i in ids} for T, ids in ids_by_T.items()},
                gifs=gifs)
    Path("results/fig3_videos.json").write_text(json.dumps(info, indent=2))
    print(json.dumps(info, indent=2))


def speed_rows(f, video, dev):
    """Figure 5 rows: for k = 1, 2, 3, five inputs, first two predictions and first two true future frames."""
    rows, labels = [], []
    for k in (1, 2, 3):
        X, Y, _ = batch_window(video[None], 1, k)
        pred = f(X.to(dev)).cpu()
        rows.append(list(X[0, :, 0].numpy()) + list(pred[0, :2, 0].numpy()) + list(Y[0, :2, 0].numpy()))
        labels.append(f"{k}x")
    return rows, labels


def n_blobs(frame, min_size=15):
    """Number of 8-connected bright regions of at least min_size pixels (two separate digits -> 2)."""
    lab, n = ndimage.label(frame > INTENSITY_MIN, structure=np.ones((3, 3)))
    sizes = ndimage.sum(np.ones_like(lab), lab, range(1, n + 1))
    return int((sizes >= min_size).sum())


def fig5_video(test):
    """First test video whose digits do not overlap (or touch) in any displayed true frame, at all speeds.

    Displayed true frames are the five inputs and first two targets at 1x, 2x and 3x; each must
    show two separate bright regions and no drop in bright area.
    """
    for i in range(len(test)):
        shown = torch.cat([torch.cat([X[0], Y[0, :2]]) for X, Y, _ in
                           (batch_window(test[i : i + 1], 1, k) for k in (1, 2, 3))])[:, 0]
        if overlap_score(shown) > 0.95 and all(n_blobs(fr.numpy()) == 2 for fr in shown):
            return i
    raise RuntimeError("no non-overlapping video found")


def fig5_plot(f, test, dev, i, path, title):
    rows, labels = speed_rows(f, test[i], dev)
    cols = [f"x{j}" for j in range(1, 6)] + ["pred h=1", "pred h=2", "true h=1", "true h=2"]
    frame_grid(rows, labels, path, f"{title}, video {9000 + i}, t=1", cols)


def fig5(test, dev):
    model = load_model("checkpoints/A_s0.pt", dev)
    i = fig5_video(test)
    fig5_plot(model, test, dev, i, "figures/fig5_speed", "Model A (seed 0)")
    Path("results/fig5_video.json").write_text(json.dumps(dict(video=9000 + i), indent=2))
    print(f"Figure 5 video: {9000 + i}")


def fig6(test, dev, final):
    """Figures 3 and 5 repeated with the final model (symmetrised D, or D+B) on exactly the same videos."""
    f = Symmetrised(load_model(f"checkpoints/{final}_s0.pt", dev))
    name = f"symmetrised {'D+B' if final == 'DB' else 'D'} (seed 0)"
    fig3_ids = json.loads(Path("results/fig3_videos.json").read_text())["ids"]
    ids_by_T = {T: [v - 9000 for v in vs] for T, vs in fig3_ids.items()}
    gifs = fig3_panels(f, test, dev, ids_by_T, "fig6a", f"Final model: {name}")
    i = json.loads(Path("results/fig5_video.json").read_text())["video"] - 9000
    fig5_plot(f, test, dev, i, "figures/fig6b_speed", f"Final model: {name}")
    Path("results/fig6_gifs.json").write_text(json.dumps(dict(final=final, gifs=gifs), indent=2))
    print("\n".join(gifs))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--fig", type=int, required=True, choices=[2, 3, 5, 6])
    p.add_argument("--final", default="D", choices=["D", "DB"])
    args = p.parse_args()
    if args.fig in (3, 5):
        require_prediction("Q2.5")
    if args.fig == 6:
        require_prediction("Q2.11")
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, _, test_u8 = load_moving_mnist()
    test = test_u8.float() / 255.0
    Path("figures").mkdir(exist_ok=True)
    with torch.no_grad():
        if args.fig == 2:
            fig2(test, dev)
        elif args.fig == 3:
            fig3_and_4(test, dev)
        elif args.fig == 5:
            fig5(test, dev)
        else:
            fig6(test, dev, args.final)


if __name__ == "__main__":
    main()
