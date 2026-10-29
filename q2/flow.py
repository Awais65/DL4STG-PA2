"""Figure 1: Farneback optical-flow direction and speed on the first 50 training videos.

Run: python -m q2.flow
"""
import json
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

from q2.data import load_moving_mnist

N_VIDEOS = 50
INTENSITY_MIN = 51     # 0.2 on the [0, 1] scale, i.e. > 51 on uint8
MAG_MIN = 0.5          # pixels


def pooled_flow(videos):
    """Directions (deg, [0, 360)) and speeds (px/frame) of kept pixels, pooled over all consecutive pairs."""
    angles, speeds, n_pairs = [], [], 0
    for vid in videos:
        frames = vid[:, 0].numpy()  # (20, 64, 64) uint8 (= float frame * 255)
        for s in range(len(frames) - 1):
            prev, nxt = frames[s], frames[s + 1]
            flow = cv2.calcOpticalFlowFarneback(prev, nxt, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            dx, dy = flow[..., 0], flow[..., 1]   # dx along columns, dy along rows (down)
            mag = np.hypot(dx, dy)
            keep = (prev > INTENSITY_MIN) & (nxt > INTENSITY_MIN) & (mag > MAG_MIN)
            angles.append(np.degrees(np.arctan2(-dy[keep], dx[keep])) % 360.0)
            speeds.append(mag[keep])
            n_pairs += 1
    return np.concatenate(angles), np.concatenate(speeds), n_pairs


def run():
    train, _, _ = load_moving_mnist()
    theta, speed, n_pairs = pooled_flow(train[:N_VIDEOS])
    assert n_pairs == N_VIDEOS * 19
    n, med = len(speed), float(np.median(speed))
    w = np.full(n, 1.0 / n)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 3.8))
    ax1.hist(theta, bins=np.arange(0, 361, 20), weights=w, edgecolor="white")
    ax1.axhline(1 / 18, color="k", ls="--", lw=1, label="uniform (1/18)")
    ax1.set(xlabel=r"flow direction $\theta$ (deg; 0 = right, 90 = up)", ylabel="fraction of kept pixels",
            title="(a) direction", xticks=range(0, 361, 45))
    ax1.legend(loc="upper right")

    ax2.hist(speed, bins=60, weights=w, edgecolor="white")
    ax2.axvline(med, color="k", ls="--", lw=1, label=f"median = {med:.3g} px/frame")
    ax2.set(xlabel="flow speed (pixels / frame)", ylabel="fraction of kept pixels", title="(b) speed")
    ax2.legend(loc="upper right")
    fig.tight_layout()

    Path("figures").mkdir(exist_ok=True)
    Path("results").mkdir(exist_ok=True)
    fig.savefig("figures/fig1_flow.png", dpi=200)
    fig.savefig("figures/fig1_flow.pdf")

    stats = {"n_pairs": n_pairs, "n_pixels_kept": n, "median_speed": med,
             "direction_fraction_per_bin": np.histogram(theta, bins=np.arange(0, 361, 20))[0].tolist()}
    stats["direction_fraction_per_bin"] = [c / n for c in stats["direction_fraction_per_bin"]]
    # Q2.1: how much the direction histogram moves if every video is mirrored (theta -> 180 - theta)
    # or rotated 90 deg anticlockwise (theta -> theta + 90), as total variation distance.
    hist = lambda a: np.histogram(a % 360.0, bins=np.arange(0, 361, 20))[0] / n
    h = hist(theta)
    stats["tv_distance_mirror"] = float(0.5 * np.abs(h - hist(180.0 - theta)).sum())
    stats["tv_distance_rot90"] = float(0.5 * np.abs(h - hist(theta + 90.0)).sum())
    stats["speed_percentiles"] = dict(zip(["p10", "p25", "p50", "p75", "p90"],
                                          np.percentile(speed, [10, 25, 50, 75, 90]).tolist()))
    stats["speed_max"] = float(speed.max())
    stats["speed_p99_9"] = float(np.percentile(speed, 99.9))
    stats["n_pixels_faster_than_2x_median"] = int((speed > 2 * med).sum())   # about 6.6 px/frame
    Path("results/fig1_flow_stats.json").write_text(json.dumps(stats, indent=2))
    print(f"pairs={n_pairs}  pixels kept={n:,}  median speed={med:.3g} px/frame")
    print("caption: Farneback flow on the first 50 training videos (950 frame pairs); "
          f"{n:,} pixels kept; median speed {med:.3g} px/frame.")


if __name__ == "__main__":
    run()
