"""Q1 architecture diagram -> figures/q1_architecture.png / .pdf (picked up by build_report.py).

Run from the repo root:  python report/q1_diagram.py
"""
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch

OUT = Path(__file__).resolve().parent.parent / "figures" / "q1_architecture"
SRC, PROC, ENC, HEAD, LAT = "#e8f0fa", "#f3f3f3", "#fdf3e1", "#e9f6ec", "#ffffff"


def box(ax, x0, y0, x1, y1, text, fc, size=8.2, weight="normal", ls="-", va="center"):
    ax.add_patch(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0, boxstyle="round,pad=0.02,rounding_size=0.08",
                                fc=fc, ec="#555", lw=0.9, ls=ls))
    ty = (y0 + y1) / 2 if va == "center" else y1 - 0.12
    ax.text((x0 + x1) / 2, ty, text, ha="center", va=va, fontsize=size, weight=weight, linespacing=1.3)


def arrow(ax, pts, ls="-", color="#333"):
    for (xa, ya), (xb, yb) in zip(pts[:-2], pts[1:-1]):
        ax.plot([xa, xb], [ya, yb], color=color, lw=1.0, ls=ls)
    ax.add_patch(FancyArrowPatch(pts[-2], pts[-1], arrowstyle="-|>", mutation_scale=10, color=color,
                                 lw=1.0, ls=ls, shrinkA=0, shrinkB=1))


R = 0.17  # node radius in data units (the axes use an equal aspect, so circles stay round)


def node(ax, x, y, name, fc, ls="-"):
    ax.add_patch(Circle((x, y), R, fc=fc, ec="#333", lw=0.9, ls=ls, zorder=3))
    ax.text(x, y, name, ha="center", va="center", fontsize=7, weight="bold", zorder=4)


def edge(ax, a, b):
    # run from circle edge to circle edge, so the arrowhead is never hidden under a node
    dx, dy = b[0] - a[0], b[1] - a[1]
    n = (dx * dx + dy * dy) ** 0.5
    ux, uy = dx / n, dy / n
    ax.add_patch(FancyArrowPatch((a[0] + R * ux, a[1] + R * uy), (b[0] - R * ux, b[1] - R * uy),
                                 arrowstyle="-|>", mutation_scale=8, color="#333", lw=0.9,
                                 shrinkA=0, shrinkB=0, zorder=2))


def draw():
    fig, ax = plt.subplots(figsize=(11.6, 6.55))
    ax.set_xlim(0, 14)
    ax.set_ylim(0.3, 8.2)
    ax.set_aspect("equal")
    ax.axis("off")

    for x, title in [(1.4, "Data sources (as of t₀)"), (4.1, "Align"), (6.7, "Embed"),
                     (9.65, "Spatio-temporal encoder"), (12.65, "Forecast head")]:
        ax.text(x, 8.0, title, ha="center", fontsize=10, weight="bold")

    # sources
    hist = [(7.15, "Levels S1–S4, T1\n(B, 72, 5, 3)"), (6.25, "Rain R1, R2\n(B, 72, 2, 3)"),
            (5.35, "Rain totals 24 h / 72 h\n(B, 2, 2)"), (4.45, "Reservoir release, past\n(B, 72, 1, 3)")]
    for y, t in hist:
        box(ax, 0.2, y - 0.36, 2.6, y + 0.36, t, SRC)
        arrow(ax, [(2.6, y), (3.1, y)])
    box(ax, 0.2, 2.24, 2.6, 2.96, "Announced release, future\n(B, 6, 1, 3)", SRC)
    box(ax, 0.2, 1.44, 2.6, 2.04, "Calendar X_cal (B, 4)", SRC)
    box(ax, 0.2, 0.5, 2.6, 1.2, "Static graph G: 7 nodes, 6 edges,\ndistances, gauge→node map", SRC, size=7.8)

    # align and embed
    box(ax, 3.1, 4.0, 5.1, 7.6, "10-min grid, t₀ − 710 … t₀\n(H = 72 steps)\n\ninputs taken as of t₀\nby arrival time\n\n"
        "forward-fill missing,\nmask + age per source\n\nrain summed per 10 min", PROC)
    arrow(ax, [(5.1, 5.8), (5.6, 5.8)])
    box(ax, 5.6, 4.0, 7.8, 7.6, "Linear projection to d = 64\nonto graph nodes\n\nRes ← release\nS1–S4, T1 ← own level\n"
        "T1, S2 ← R1    U ← R2\n(+ rain totals c_g)\n\n+ node embedding s_v\n+ time encoding p_τ\n\n"
        "node tensor (B, 72, 7, 64)", PROC)
    arrow(ax, [(7.8, 5.8), (8.3, 5.8)])

    # encoder: K = 4 x (T block -> routing block), with the river graph
    box(ax, 8.3, 1.9, 11.0, 7.6, "", ENC)
    ax.text(9.65, 7.38, "× K = 4", ha="center", fontsize=9, weight="bold")
    box(ax, 8.5, 6.45, 10.8, 7.12, "T block: causal dilated TCN\nkernel 3, dilations 1–16 (63 steps)", LAT, size=7.6)
    arrow(ax, [(9.65, 6.45), (9.65, 6.1)])
    box(ax, 8.5, 5.4, 10.8, 6.1, "Routing block: lag attention\non each edge, ℓ = 0…36 (6 h)", LAT, size=7.6)
    ax.text(9.65, 5.05, "river graph (flow →)", ha="center", fontsize=7.6, style="italic")
    main = [("Res", 8.6, "#f6c99a"), ("S1", 9.125, "#bcd6f2"), ("S2", 9.65, "#bcd6f2"),
            ("S3", 10.175, "#bcd6f2"), ("S4", 10.7, "#bcd6f2")]
    y0 = 3.75
    for (_, xa, _), (_, xb, _) in zip(main[:-1], main[1:]):
        edge(ax, (xa, y0), (xb, y0))
    edge(ax, (9.125, 4.55), (9.65, y0))
    edge(ax, (9.65, 2.95), (10.175, y0))
    for name, x, fc in main:
        node(ax, x, y0, name, fc)
    node(ax, 9.125, 4.55, "T1", "#bcd6f2")
    node(ax, 9.65, 2.95, "U", LAT, ls="--")
    ax.text(9.65, 2.3, "U = unmonitored tributary (latent, no data)", ha="center", fontsize=6.4)
    ax.text(9.65, 2.02, "longest path Res → S4: 4 hops", ha="center", fontsize=6.6)

    # head
    arrow(ax, [(11.0, 6.8), (11.5, 6.8)])
    box(ax, 11.5, 6.4, 13.8, 7.2, "take τ = H (t₀) slice of S1–S4\n(B, 4, 64)", HEAD, size=7.8)
    arrow(ax, [(12.65, 6.4), (12.65, 6.05)])
    box(ax, 11.5, 5.35, 13.8, 6.05, "concat [h_s ; c_rel ; c_cal]", HEAD, size=7.8)
    arrow(ax, [(12.65, 5.35), (12.65, 5.0)])
    box(ax, 11.5, 4.2, 13.8, 5.0, "Quantile head g_s per station\nŷ₀.₅ = y(t₀) + g_s(·), ± softplus", HEAD, size=7.6)
    arrow(ax, [(12.65, 4.2), (12.65, 3.85)])
    box(ax, 11.5, 2.3, 13.8, 3.85, "OUTPUT Ŷ (B, 4, 3)\nlevel at t₀ + 60 min,\nS1–S4, q = 0.1 / 0.5 / 0.9\n\n"
        "aux path (B, 6, 4, 3),\nt₀+10 … t₀+60 (training only)", HEAD, size=7.6, weight="normal")

    # future release + calendar -> c_rel, c_cal -> concat
    box(ax, 3.1, 1.5, 7.8, 2.9, "future-release encoder  c_rel = MLP(X_rel_future, actual − scheduled)\n"
        "calendar features  c_cal = (sin, cos time of day; sin, cos day of year)", PROC, size=7.6)
    arrow(ax, [(2.6, 2.6), (3.1, 2.6)])
    arrow(ax, [(2.6, 1.74), (3.1, 1.74)])
    arrow(ax, [(7.8, 1.65), (11.25, 1.65), (11.25, 5.7), (11.5, 5.7)])

    # static graph -> embedding (gauge map) and routing (edges)
    arrow(ax, [(2.6, 0.85), (8.05, 0.85), (8.05, 4.4), (7.8, 4.4)], ls="--", color="#777")
    arrow(ax, [(8.05, 0.85), (9.0, 0.85), (9.0, 1.9)], ls="--", color="#777")
    ax.text(5.3, 0.97, "gauge→node map (embed), edges + distances (routing)", ha="center", fontsize=6.8, color="#555")

    fig.tight_layout()
    fig.savefig(OUT.with_suffix(".png"), dpi=200)
    fig.savefig(OUT.with_suffix(".pdf"))
    print("wrote", OUT.with_suffix(".png"))


if __name__ == "__main__":
    draw()
