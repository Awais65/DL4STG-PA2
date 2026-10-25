# Generative AI Usage Log — PA2

The course AI policy requires the report to state, at the end, **each prompt, its output, and the edits I made**.
This log is the source for that appendix. Add an entry for every prompt as it happens, not afterwards.

- **Tool:** Claude Code (model: Claude Opus 5.5) unless stated otherwise
- **Output** = a short summary of what the AI produced, plus the files or commits it touched
- **My edits** = what I changed, rejected or verified myself. Fill this in honestly; "none yet" is fine until it is true.

| # | Date | Prompt (verbatim) | Output (summary + files) | My edits / verification |
|---|------|-------------------|--------------------------|-------------------------|
| 1 | 2026-10-14 | "Analyze the PA2 and what are the deliverables" | Read `DL4STG_PA2.pdf` and the Question1 folder. Summarised the deadline, submission rules (PDF report, public repo link, AI appendix, committed predictions for Q2.5/Q2.11), the six required Q1 headings, and the Q2 models A–D, Tables 1–6, Figures 1–6 and Q2.1–Q2.17. Suggested a 12-day plan. No files changed. | Read for orientation only; nothing copied into the report. |
| 2 | 2026-10-15 | "Walk me through this so that I can implement this? where to start?" | Step-by-step implementation guide for Q2: environment, repo layout, order of work, and starter code for transforms/windows, the Table 1 checks, the Farneback flow (Figure 1), the ConvLSTM (model A, 480,784-parameter check), the masked loss/training loop and the evaluation harness. No files changed. | |
| 3 | 2026-10-15 | "You can use conda env ml310 for Step 0. Here is the Github repo for PA2 https://github.com/Awais65/DL4STG-PA2.git. Lets create a log book now for the prompts where we can add the prompts. Once done lets start with the next steps" | Installed `opencv-python` and `imageio` into `ml310`. Cloned the repo. Created this log, `README.md`, `.gitignore`, `requirements.txt`, `q2/data.py` (loading, transforms, windows), `q2/checks.py` (Table 1), `q2/flow.py` (Figure 1) and `q2/models.py` (ConvLSTM forecaster). Ran them: all Table 1 checks pass (`results/table1.md`), model A has 480,784 parameters, and Figure 1 was generated (`figures/fig1_flow.png`; 103,050 pixels kept, median 3.32 px/frame). | |
| 4 | 2026-10-25 | "check PA2 progress and whats remaining" / "yes, commit and start the training code" | Status review against the handout. Made the first git commit. Wrote `q2/train.py` (training protocol for models A, B, C, D, D+B: masked loss, speed/symmetry augmentation, seeding, epoch timing, validation monitoring); checked window sampling and timed one epoch (18.6 s on RTX 4070 Laptop). Started model A training, seeds 0 and 1. | |
| 5 | 2026-10-25 | "was anything missing?" / "Yes" | Audited `q2/train.py` and the model A runs against the training protocol (all requirements met). Flagged gaps: no 2x/3x validation metric for choosing model D, GPU nondeterminism, unstable epoch timings on the laptop GPU, uncommitted results. Replaced the validation monitor with `val_metrics` (val E^(5) at 1x and E^(2) at 1x/2x/3x, t=1). | |

<!-- Template for new rows:
| N | YYYY-MM-DD | "exact prompt" | what it produced; files | what I changed / checked |
-->
