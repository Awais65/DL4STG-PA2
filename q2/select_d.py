"""Compare model-D candidates on the validation split (final-epoch metrics logged by q2.train).

    python -m q2.select_d

The test split is never touched here. Candidates are ranked by validation E^(2) at 3x, the
speed model D is meant to fix; E^(5) at 1x and E^(2) at 2x are shown to check nothing else got worse.
"""
import json
from pathlib import Path

from q2.evaluate import fmt, markdown

ref = {n: json.loads(Path(f"results/train/{n}_s0.json").read_text()) for n in ("A", "C")}
cands = {p.stem: json.loads(p.read_text()) for p in sorted(Path("results/train").glob("Dcand_*_s0.json"))}

rows = []
for name, log in {**ref, **cands}.items():
    last = log["history"][-1]
    arch = log["arch"] or dict(k=3, dilation=1, patch=4)
    rows.append([name, f"k={arch['k']}, dil={arch['dilation']}, patch={arch['patch']}", f"{log['n_params']:,}",
                 # model A was trained before the 2x/3x validation monitor existed
                 *[fmt(last[c]) if c in last else "–" for c in ("val_e5", "val_e2_1x", "val_e2_2x", "val_e2_3x")],
                 f"{log['mean_epoch_time']:.1f}"])
best = min(cands, key=lambda n: cands[n]["history"][-1]["val_e2_3x"])
header = ["Run (seed 0)", "Architecture", "Params", "val E^(5) 1x", "val E^(2) 1x", "val E^(2) 2x", "val E^(2) 3x", "s/epoch"]
md = ("## Model D selection (validation split, videos 8000-8999, t=1, final epoch)\n\n"
      + markdown(header, rows) + f"\nLowest validation E^(2) at 3x: **{best}**.\n")
Path("results/model_d_selection.md").write_text(md, encoding="utf-8")
print(md)
