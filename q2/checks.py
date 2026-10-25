"""Table 1: sanity checks for the transforms and windows. Run: python -m q2.checks"""
from pathlib import Path

import torch

from q2.data import M, R, window


def bright_pixel(v):
    """(frame, row, column) of the single nonzero entry; indices from 0, as in the handout's marker."""
    nz = v.nonzero()
    assert len(nz) == 1, f"expected one bright pixel, found {len(nz)}"
    f, _, i, j = nz[0].tolist()
    return (f, i, j)


def frames_returned(k):
    v = torch.arange(1, 21, dtype=torch.float32).view(20, 1, 1, 1).expand(20, 1, 64, 64).clone()
    X, Y, mask = window(v, t=1, k=k)
    ins = [int(x) for x in X[:, 0, 0, 0]]
    tgs = [int(y) for y, m in zip(Y[:, 0, 0, 0], mask) if m]
    return ins, tgs


def fmt_frames(ins, tgs):
    return f"{', '.join(map(str, ins))} / {', '.join(map(str, tgs))}"


def run():
    torch.manual_seed(0)
    x = torch.rand(20, 1, 64, 64)
    p = torch.zeros(20, 1, 64, 64)
    p[7, 0, 10, 20] = 1

    rows = []
    for name, lhs, rhs in [
        ("(a) max|M(M(x)) - x|", M(M(x)), x),
        ("(b) max|R^4(x) - x|", R(x, 4), x),
        ("(c) max|M R M(x) - R^-1(x)|", M(R(M(x))), R(x, -1)),
    ]:
        diff = (lhs - rhs).abs().max().item()
        rows.append((name, "0", f"{diff:g} (torch.equal: {torch.equal(lhs, rhs)})",
                     diff == 0 and torch.equal(lhs, rhs)))

    for label, T, exp in [("(d) bright pixel of M(p)", M, (7, 10, 43)), ("(e) bright pixel of R(p)", R, (7, 43, 10))]:
        got = bright_pixel(T(p))
        rows.append((label, str(exp), str(got), got == exp))

    expected = {
        1: ([1, 2, 3, 4, 5], [6, 7, 8, 9, 10]),
        2: ([1, 3, 5, 7, 9], [11, 13, 15, 17, 19]),
        3: ([1, 4, 7, 10, 13], [16, 19]),
    }
    for label, k in zip("fgh", (1, 2, 3)):
        got = frames_returned(k)
        rows.append((f"({label}) {k}x input / target frames", fmt_frames(*expected[k]), fmt_frames(*got),
                     got == expected[k]))

    lines = ["| Check | Expected | Yours |", "|---|---|---|"]
    lines += [f"| {n.replace('|', chr(92) + '|')} | {e} | {y} |" for n, e, y, _ in rows]
    table = "\n".join(lines)
    print(table)
    ok = all(r[3] for r in rows)
    print("\nALL CHECKS PASS" if ok else "\nSOME CHECKS FAIL — fix before continuing")

    out = Path("results")
    out.mkdir(exist_ok=True)
    (out / "table1.md").write_text("**Table 1.** Sanity checks of transforms and windows.\n\n" + table + "\n")
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if run() else 1)
