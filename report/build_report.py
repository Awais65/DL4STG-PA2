"""Assemble the PA2 report: answers + tables + figures + AI log -> report/<roll>_PA2.html and .pdf.

Run from the repo root:  python report/build_report.py
The PDF is printed with headless Chrome (or Edge). Edit the answer files, not the HTML.
"""
import html
import math
import re
import subprocess
from pathlib import Path

import markdown

ROOT = Path(__file__).resolve().parent.parent
REPORT = ROOT / "report"
REPO_URL = "https://github.com/Awais65/DL4STG-PA2"
AUTHOR = "Muhammad Awais Ismail"
ROLL = "24030007"
HARDWARE = ("NVIDIA GeForce RTX 4070 Laptop GPU (8 GB), Intel Core i9-14900HX, 32 GB RAM; "
            "Windows 11, PyTorch 2.6.0, CUDA 12.4")
BROWSERS = [r"C:\Program Files\Google\Chrome\Application\chrome.exe",
            r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"]


LIST = re.compile(r"^\s*([-*]|\d+\.|\(\w+\)) ")


def subscripts(text):
    """x_{ab} and C_M -> HTML subscripts, outside code spans and code blocks."""
    parts = re.split(r"(```.*?```|`[^`]*`)", text, flags=re.S)
    for i in range(0, len(parts), 2):
        t = re.sub(r"(?<=\S)_\{([^{}]*)\}", r"<sub>\1</sub>", parts[i])
        parts[i] = re.sub(r"(?<=[^\W\d_])_([^\W_]+)(?![\w{])", r"<sub>\1</sub>", t)
    return "".join(parts)


def md(text):
    lines = text.splitlines()
    out = []
    for i, l in enumerate(lines):
        # Python-Markdown needs a blank line before a list and 4-space nesting (the files use 2).
        if l.lstrip().startswith(("- ", "1. ")) and i and lines[i - 1].strip() \
                and not LIST.match(lines[i - 1]) and not lines[i - 1].startswith(" "):
            out.append("")
        out.append(re.sub(r"^  (?=\S)", "    ", l))
    return markdown.markdown(subscripts("\n".join(out)), extensions=["tables", "fenced_code"])


def sig3(x):
    """3 significant figures; scientific notation unless 0.1 <= |x| < 1000."""
    x = float(x)
    if x == 0:
        return "0"
    if 0.1 <= abs(x) < 1000:
        return f"{x:.{max(2 - math.floor(math.log10(abs(x))), 0)}f}"
    m, e = f"{x:.2e}".split("e")
    return f"{m}e{int(e)}"


def symbols(s):
    s = re.sub(r"\^(\([^)]*\)|-?\w+)", r"<sup>\1</sup>", s)
    s = re.sub(r"_(\w+)", r"<sub>\1</sub>", s)
    s = re.sub(r"(\d)x\b", "\\1×", s)
    return s.replace("\\|", "|")


def cell(s):
    s = re.sub(r"\d\.\d+e[+-]\d+", lambda m: sig3(m.group()), s.strip())
    # "mean (seed 0, seed 1)": put the seed values on their own line to keep wide tables narrow.
    s = re.sub(r"^(\S+(?: s)?) (\(.*\))$", r"\1<br><span class=seeds>\2</span>", s)
    return symbols(s)


def table(n, caption, cls=""):
    lines = [l for l in (ROOT / f"results/table{n}.md").read_text(encoding="utf-8").splitlines()
             if l.startswith("|")]
    rows = [[c for c in re.split(r"(?<!\\)\|", l)[1:-1]] for l in lines]
    head, body = rows[0], rows[2:]
    out = [f'<table class="{cls}"><caption><b>Table {n}.</b> {caption}</caption><thead><tr>']
    out += [f"<th>{symbols(h.strip())}</th>" for h in head] + ["</tr></thead><tbody>"]
    for r in body:
        out += ["<tr>"] + [f"<td>{cell(c)}</td>" for c in r] + ["</tr>"]
    return "".join(out + ["</tbody></table>"])


def figure(n, files, caption, width="100%"):
    imgs = "".join(f'<img src="../figures/{f}" style="width:{width}">' for f in files)
    return f'<figure>{imgs}<figcaption><b>Figure {n}.</b> {caption}</figcaption></figure>'


def answers_q2():
    text = (REPORT / "answers_q2.md").read_text(encoding="utf-8")
    parts = re.split(r"(?m)^## (Q2\.\d+)\s*$", text)[1:]
    return {k: f'<h3>{k}</h3>{md(v)}' for k, v in zip(parts[::2], parts[1::2])}


def answers_q1():
    text = (REPORT / "answers_q1_draft.md").read_text(encoding="utf-8")
    text = text.split("\n", 1)[1]  # drop the file's own title line
    text = re.sub(r"(?m)^\*\(Redraw this.*\)\*\s*$", "", text)
    diagram = ROOT / "figures/q1_architecture.png"
    if diagram.exists():  # once the diagram is redrawn, it replaces the ASCII sketch
        text = re.sub(r"```\n.*?```", "![Architecture](../figures/q1_architecture.png)",
                      text, count=1, flags=re.S)
    return md(text.replace("\n## ", "\n### "))


def ai_log():
    text = (ROOT / "AI_USAGE_LOG.md").read_text(encoding="utf-8")
    text = re.sub(r"<!--.*?-->", "", text, flags=re.S)
    text = text[text.index("| # |"):]
    intro = ("All generative-AI use for this assignment, in order: each prompt (verbatim), a summary of the "
             "output and the files it touched, and the edits I made. Tool: Claude Code (Claude Opus 5.5). "
             "The full log is <code>AI_USAGE_LOG.md</code> in the repository.")
    return f"<p>{intro}</p>" + md(text)


def build():
    a = answers_q2()
    fig3_caption = (
        "Seed-0 model A on the two test videos with the largest per-video C<sub>M</sub> (9269, 9818; "
        "T = M) and the two with the largest per-video C<sub>R</sub> (9928, 9391; T = R). "
        "Rows: (1) X | f(X), (2) T(X) | T(Y), (3) T(X) | f(T(X)), (4) T(X) | T(f(X)); columns 1–5 "
        "are the inputs, 6–10 the five forecast horizons. GIFs: {gifs}.")
    gifs3 = ", ".join(f"<code>figures/gifs/fig3_{p}.gif</code>"
                      for p in ["M1_v9269", "M2_v9818", "R1_v9928", "R2_v9391"])
    gifs6 = ", ".join(f"<code>figures/gifs/fig6a_{p}.gif</code>"
                      for p in ["M1_v9269", "M2_v9818", "R1_v9928", "R2_v9391"])
    fig3_files = [f"fig3_{p}.png" for p in ["M1_v9269", "M2_v9818", "R1_v9928", "R2_v9391"]]
    fig6_files = [f"fig6a_{p}.png" for p in ["M1_v9269", "M2_v9818", "R1_v9928", "R2_v9391"]]
    learned = "Learned models: mean (seed 0, seed 1); Blank and Copy-last are deterministic."

    body = f"""
<header class=title>
  <div class=course>AI651 — Deep Learning for Space, Time and Graphs · Fall 2026 · LUMS</div>
  <h1>Assignment 2 Report</h1>
  <div class=who>{AUTHOR} · Roll number {ROLL}</div>
  <div class=repo>Code repository: <a href="{REPO_URL}">{REPO_URL}</a></div>
  <div class=hw>Hardware for all timings: {HARDWARE}.</div>
</header>

<h1>Question 1 — River Forecasting Proposal</h1>
{answers_q1()}

<h1 class=newpage>Question 2 — Motion or Memory?</h1>
<p>All numbers use the 1,000 test videos (9000–9999) at t = 1 except Table 1 (synthetic tensors),
Figure 1 (training videos), parameter counts and timings. In every table, learned models are reported
as mean (seed 0, seed 1); figures, GIFs and histograms use seed 0. Errors are given to three
significant figures.</p>

<h2>1. Setup</h2>
{table(1, "Sanity checks of the transforms and speed windows.")}
{figure(1, ["fig1_flow.png"], "Farneback optical flow on the first 50 training videos (950 consecutive-frame pairs), "
        "pooled over the 103,050 kept pixels (intensity &gt; 0.2 in both frames and flow magnitude &gt; 0.5 px). "
        "(a) Direction histogram in 20° bins, 0° rightward and 90° upward, with the uniform level 1/18 marked. "
        "(b) Speed histogram; median 3.32 px/frame (vertical line). Both as fractions of kept pixels.")}
{a['Q2.1']}{a['Q2.2']}

<h2>2. Baseline: ConvLSTM</h2>
{table(2, "Per-horizon test MSE and E<sup>(5)</sup> at 1×. " + learned)}
{figure(2, ["fig2_examples.png"], "Seed-0 model A at 1×: inputs, predictions and ground truth for test videos "
        "9002, 9007 and 9191 (digits overlapping).", "72%")}
{a['Q2.3']}{a['Q2.4']}{a['Q2.5']}

<h2>3. Testing the Baseline</h2>
{table(3, "Accuracy and mirror/rotation consistency at 1×. " + learned)}
{figure(3, fig3_files, fig3_caption.format(gifs=gifs3), "74%")}
{figure(4, ["fig4_consistency_hist.png"], "Histograms of per-video C<sub>M</sub> (a) and C<sub>R</sub> (b) "
        "for seed-0 model A over the 1,000 test videos (log scale); the Figure 3 videos are marked with dashed lines.")}
{a['Q2.6']}{a['Q2.7']}{a['Q2.8']}
{table(4, "Model A at 1×, 2× and 3×, with Copy-last for comparison. " + learned)}
{figure(5, ["fig5_speed.png"], "Seed-0 model A on test video 9021 (digits never overlap in the displayed true "
        "frames): the five input frames and the first two predicted and true future frames at 1×, 2× and 3×.")}
{a['Q2.9']}{a['Q2.10']}

<h2>4. What Can Data Fix?</h2>
{a['Q2.11']}
{table(5, "Models A–D. Inference: wall-clock time for the 1,000 test videos at batch size 32. "
       "Training: mean wall-clock time per epoch over the 30 epochs. Learned models: mean (seed 0, seed 1).", "wide")}
<p class=note>Model D was chosen on the validation split only (videos 8000–8999): dilation 2 had the lowest
validation E<sup>(2)</sup> at 3× (3.12e-2) among the candidates, against 3.28e-2 for C; dilation 3 and patch 8
stayed near a blank prediction (3.74e-2, 3.95e-2). Full comparison: <code>results/model_d_selection.md</code>.</p>
{a['Q2.12']}{a['Q2.13']}{a['Q2.14']}

<h2>5. The Final Model</h2>
{table(6, "Symmetrised models (average over all eight transformations). The training-time column is that "
       "of the underlying learned model; symmetrisation needs no extra training. The final model is "
       "symmetrised D. Mean (seed 0, seed 1).", "wide")}
<p class=note>Final model: <b>symmetrised D</b>. D+B was also trained and wrapped, but Table 6 shows no gain over
symmetrised D (E<sup>(5)</sup> 2.86e-2 for both; E<sup>(2)</sup> at 3× 3.15e-2 against 3.14e-2), so the simpler
model D was kept.</p>
{figure("6(a)", fig6_files, "Figure 3 repeated with the final model (symmetrised D, seed 0) "
        f"on the same four videos and transformations, same row layout. GIFs: {gifs6}.", "74%")}
{figure("6(b)", ["fig6b_speed.png"], "Figure 5 repeated with the final model (symmetrised D, seed 0) on test video 9021.")}
{a['Q2.15']}{a['Q2.16']}{a['Q2.17']}

<h2>Notes on the experiments</h2>
<ul>
<li><b>Predictions.</b> Q2.5 was committed in <code>ff9a9db</code> before the experiments in Testing the Baseline,
and Q2.11 in <code>f6fecaa</code> before models B–D were trained (<code>predictions/</code>). Both are reproduced
above unchanged. Before writing Q2.5 I had seen model A's 2× and 3× validation errors from the training monitor.
The wording of both predictions started from an AI draft that I edited (Appendix).</li>
<li><b>Timing.</b> Epoch times on the laptop GPU varied widely for the same architecture (6.6–23.9 s per epoch),
so the training-time columns compare models only roughly.</li>
<li><b>Re-run.</b> The first seed-0 run of model B had one epoch that stalled for 2,148 s. It was re-run with the
same settings; the old log is kept in <code>logs/superseded/</code>.</li>
</ul>

<h1 class=newpage>Appendix — Generative AI usage</h1>
<div class=ailog>{ai_log()}</div>
"""
    css = """
@page { size: A4; margin: 16mm 15mm 18mm; }
body { font-family: "Segoe UI", "Cambria Math", sans-serif; font-size: 10pt; line-height: 1.42; color: #111; }
h1 { font-size: 16pt; margin: 18px 0 8px; border-bottom: 1.5px solid #333; padding-bottom: 3px; }
h2 { font-size: 13pt; margin: 18px 0 6px; }
h3 { font-size: 11pt; margin: 14px 0 4px; }
.newpage { break-before: page; }
.title { text-align: center; margin: 10px 0 24px; }
.title h1 { border: none; font-size: 22pt; margin: 6px 0; }
.title .course { font-size: 10.5pt; color: #444; }
.title .who { font-size: 12pt; margin: 6px 0; }
.title .repo, .title .hw { font-size: 10pt; margin: 3px 0; }
table { border-collapse: collapse; margin: 10px auto; font-size: 8.5pt; break-inside: avoid; }
th, td { border: 1px solid #bbb; padding: 2px 5px; text-align: left; vertical-align: top; }
th { background: #f0f0f0; }
caption { caption-side: top; text-align: left; font-size: 9pt; padding-bottom: 4px; }
table.wide { font-size: 7pt; width: 100%; }
table.wide th, table.wide td { padding: 2px 3px; }
.seeds { color: #555; white-space: nowrap; }
figure { margin: 12px 0; text-align: center; break-inside: avoid; }
figure img { display: block; margin: 0 auto 4px; }
figcaption { text-align: left; font-size: 9pt; }
pre { font-size: 6.6pt; line-height: 1.25; background: #f7f7f7; padding: 6px; overflow: hidden; break-inside: avoid; }
code { font-size: 0.92em; }
.note { font-size: 9pt; background: #f6f6f6; padding: 5px 8px; }
.ailog table { font-size: 7pt; width: 100%; break-inside: auto; }
.ailog tr { break-inside: avoid; }
"""
    page = (f"<!doctype html><html><head><meta charset=utf-8><title>PA2 Report — {html.escape(AUTHOR)}</title>"
            f"<style>{css}</style></head><body>{body}</body></html>")
    out_html = REPORT / f"{ROLL}_PA2.html"
    out_html.write_text(page, encoding="utf-8")
    browser = next(b for b in BROWSERS if Path(b).exists())
    out_pdf = REPORT / f"{ROLL}_PA2.pdf"
    subprocess.run([browser, "--headless", "--disable-gpu", "--no-pdf-header-footer",
                    f"--print-to-pdf={out_pdf}", out_html.as_uri()], check=True, capture_output=True)
    print("wrote", out_html, "and", out_pdf)


if __name__ == "__main__":
    build()
