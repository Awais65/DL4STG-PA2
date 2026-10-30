"""Extra numbers cited in the Q2.6-Q2.10 answers (seed-0 model A, test split, t=1).

    python -m q2.answer_stats      # -> results/answer_stats.json

- Blank's E^(2) at each speed (Table 4 lists only Copy-last);
- per-video C_M vs C_R (the Figure 4 comparison);
- brightness-centroid tracks for the Figure 3 panels and the Figure 5 video, and for the final
  model (symmetrised D) on the same videos (Figure 6). Centroids are taken
  over all bright mass in a frame, so with two digits they follow the pair, not each digit;
- which input cells can reach one output cell (model A vs model D coupling).
"""
import json
from pathlib import Path

import numpy as np
import torch

from q2.data import batch_window, load_moving_mnist
from q2.evaluate import TRANSFORMS, Symmetrised, horizon_mse, load_model, transform_errors

INTENSITY_MIN = 0.2


def frame_stats(fr):
    """Brightness centroid (row, col), peak value and bright-pixel count of one 64x64 frame."""
    fr = fr.detach().cpu()
    w = fr.clamp(min=0)
    r, c = torch.meshgrid(torch.arange(64.0), torch.arange(64.0), indexing="ij")
    return dict(row=float((w * r).sum() / w.sum()), col=float((w * c).sum() / w.sum()),
                peak=float(fr.max()), bright=int((fr > INTENSITY_MIN).sum()))


@torch.no_grad()
def main():
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, _, test_u8 = load_moving_mnist()
    test = (test_u8.float() / 255.0).to(dev)
    A = load_model("checkpoints/A_s0.pt", dev)
    out = {}

    out["blank_E2"] = {}
    for k in (1, 2, 3):
        _, Y, _ = batch_window(test, 1, k)
        out["blank_E2"][f"{k}x"] = horizon_mse(torch.zeros_like(Y), Y)[:, :2].mean().item()

    CM, _ = transform_errors(A, test, TRANSFORMS["M"])
    CR, _ = transform_errors(A, test, TRANSFORMS["R"])
    CM, CR = CM.cpu().numpy(), CR.cpu().numpy()
    out["per_video_C"] = dict(median_C_M=float(np.median(CM)), median_C_R=float(np.median(CR)),
                              frac_videos_C_R_gt_C_M=float((CR > CM).mean()),
                              corr_C_M_C_R=float(np.corrcoef(CM, CR)[0, 1]))

    out["fig3_tracks"] = {}
    for T_name, vids in json.loads(Path("results/fig3_videos.json").read_text())["ids"].items():
        T = TRANSFORMS[T_name]
        for v in vids:
            X, Y, _ = batch_window(test[v - 9000 : v - 8999], 1, 1)
            fX, fTX = A(X), A(T(X))
            out["fig3_tracks"][f"{T_name}_{v}"] = {
                "per_horizon_C": horizon_mse(fTX, T(fX))[0].tolist(),
                "last_input": frame_stats(T(X)[0, 4, 0]),
                **{name: [frame_stats(seq[0, h, 0]) for h in range(5)]
                   for name, seq in (("row2_T(Y)", T(Y)), ("row3_f(T(X))", fTX), ("row4_T(f(X))", T(fX)))},
            }

    v = json.loads(Path("results/fig5_video.json").read_text())["video"]
    out["fig5_tracks"] = {"video": v}
    for k in (1, 2, 3):
        X, Y, _ = batch_window(test[v - 9000 : v - 8999], 1, k)
        pred = A(X)
        out["fig5_tracks"][f"{k}x"] = dict(last_input=frame_stats(X[0, 4, 0]),
                                           pred=[frame_stats(pred[0, h, 0]) for h in range(2)],
                                           true=[frame_stats(Y[0, h, 0]) for h in range(2)])

    # Q2.16: the final model (symmetrised D, seed 0) on the same Figure 3 / Figure 5 videos.
    Dbar = Symmetrised(load_model("checkpoints/D_s0.pt", dev))
    out["fig6_tracks"] = {}
    for key in out["fig3_tracks"]:
        T_name, v = key.split("_")
        T = TRANSFORMS[T_name]
        X, Y, _ = batch_window(test[int(v) - 9000 : int(v) - 8999], 1, 1)
        fX, fTX = Dbar(X), Dbar(T(X))
        out["fig6_tracks"][key] = {
            "per_horizon_C": horizon_mse(fTX, T(fX))[0].tolist(),
            "per_horizon_E_T": horizon_mse(fTX, T(Y))[0].tolist(),
            "per_horizon_E_T_model_A": horizon_mse(A(T(X)), T(Y))[0].tolist(),
            "row3_f(T(X))": [frame_stats(fTX[0, h, 0]) for h in range(5)],
        }
    v = out["fig5_tracks"]["video"]
    for k in (1, 2, 3):
        X, Y, _ = batch_window(test[v - 9000 : v - 8999], 1, k)
        out["fig5_tracks"][f"{k}x"]["E2_model_A"] = horizon_mse(A(X), Y)[0, :2].mean().item()
        out["fig5_tracks"][f"{k}x"]["E2_final"] = horizon_mse(Dbar(X), Y)[0, :2].mean().item()

    # Q2.12 / Q2.14: which folded-grid input cells can affect one output cell (8, 8) at h=1.
    # A dilation-2 convolution only links cells two apart, so model D should split by cell parity.
    out["coupling"] = {}
    for name in ("A", "D"):
        model = load_model(f"checkpoints/{name}_s0.pt", dev)
        with torch.enable_grad():
            x = torch.zeros(1, 5, 1, 64, 64, device=dev, requires_grad=True)
            cell = model(x)[0, 0, 0].reshape(16, 4, 16, 4)[8, :, 8, :].sum()
            g = torch.autograd.grad(cell, x)[0][0, :, 0].abs().sum(0).reshape(16, 4, 16, 4).sum((1, 3))
        nz = (g > 0).nonzero().tolist()
        out["coupling"][name] = dict(n_input_cells=len(nz),
                                     parities=sorted({(r % 2, c % 2) for r, c in nz}))

    Path("results/answer_stats.json").write_text(json.dumps(out, indent=2))
    print(json.dumps({k: out[k] for k in ("blank_E2", "per_video_C")}, indent=2))


if __name__ == "__main__":
    main()
