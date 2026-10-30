# Q2 answers (working draft for the report)

## Q2.1

Figure 1(a) shows motion in every direction. Each 20° bin holds between 0.47× and 1.48× the uniform share of 1/18, and no bin is empty. Mirroring every video maps a direction θ to 180° − θ, and rotating by 90° maps it to θ + 90°. Either way the histogram is reshuffled, but every new direction already occurs in training. The mirrored histogram differs from the original by 13% of its mass (total variation distance) and the rotated one by 20%. Most of that difference is likely sampling noise, since the 50 videos contain only about 100 digit trajectories. So mirrored and rotated trajectories should already look familiar as motion. Figure 1(b) gives a typical speed of 3.3 px/frame (median), with 80% of kept pixels between 2.1 and 4.1 px/frame. At 2× a digit moves about 6.6 px/frame and at 3× about 10 px/frame. Only 21 of the 103,050 kept pixels (0.02%) move faster than 6.6 px/frame, and 99.9% move slower than 5.7 px/frame, so 2× and 3× motion is essentially absent from the data. Optical flow is unreliable where digits overlap: two digits moving differently in one region produce mixed or averaged vectors, and the brightness of a digit that's being occluded changes. That noise mostly affects a small fraction of pixels and is not tied to any one direction, so it could blur the histograms but not create or remove whole directions. The few pixels above 6.6 px/frame are more likely such flow errors than real fast motion. My conclusion holds.

## Q2.2

- Mirror changes both: it maps direction θ to 180° − θ, and it produces mirrored digits (a backwards 2, 3, 4 or 7) that never occur in training.
- Rotation by 90° also changes both: directions shift by 90°, and the digits appear sideways.
- Speed-up changes only the motion: it skips frames, so the displacement per frame doubles or triples, but each frame shows ordinary upright digits.

By Q2.1, mirror and rotation keep the motion statistics familiar while making the appearance unfamiliar. Speed-up does the opposite: familiar appearance, unfamiliar and larger motion. All three keep the physics valid (constant velocity, elastic bounces) and change no other property of the data. A failure under mirror or rotation therefore points at appearance. A failure under speed-up points at motion, either unseen speeds or the architecture's reach. That makes them a controlled experiment: each transformation isolates one cause.

## Q2.3

Reach is measured centre-to-centre in original pixels. On the folded 16×16 grid, one cell is 4 px. Only the 3×3 convolutions inside the two ConvLSTM layers move information. With stride 1 and no dilation, each such convolution moves it at most one cell, which is 4 px. Pixel-unshuffle, pixel-shuffle, the 1×1 output convolution and the elementwise gate and cell updates move nothing. Within one step τ, the possible paths to the prediction x̂_{τ+1} are:

- (i) from the input u_τ through layer 1's convolution (one cell), then through layer 2's convolution (one cell), to the output;
- (ii) from the previous state h⁽¹⁾_{τ−1}, which enters the same layer-1 convolution, through the same two convolutions;
- (iii) from h⁽²⁾_{τ−1} through layer 2's convolution only (one cell).

The cell states c⁽ℓ⁾ are updated elementwise, so they add no distance. Paths (i) and (ii) are the longest, at two cells, so model A's reach is 8 px per time step. That is above the typical training speed of 3.3 px/frame, close to 2× (about 6.6 px/frame) and below 3× (about 10 px/frame).

## Q2.4

Model A beats both reference predictors at every horizon. Its E⁽⁵⁾ is 2.18e-2, against 4.26e-2 for Blank and 6.41e-2 for Copy-last (Table 2), so it has clearly learned more than "predict nothing" or "nothing moves". Its error grows steadily with horizon, from MSE₁ = 1.66e-2 to MSE₅ = 2.58e-2 (+55%). Blank's error is flat at about 4.26e-2, because it ignores the input. Copy-last is worse than Blank at every horizon and grows from 4.92e-2 to 7.32e-2. A copied digit sits in the wrong place, so it is penalised twice: once for the missing digit at its true position and once for the bright digit where nothing is. Figure 2 shows why model A's error grows. Its predictions put the digits in roughly the right place, but they become blurrier and dimmer with each step. Under MSE, an uncertain model hedges by spreading intensity over the positions it thinks are plausible, and the uncertainty compounds because each prediction is fed back as the next input. When digits overlap (video 9191), the merged input gives the model no clear separation. Its first predictions are a single blurred blob; at later horizons that blob splits into two blobs roughly where the digits really go, but neither is recognisable as a "5".

## Q2.5

*Prediction committed before the baseline tests (`ff9a9db`, 29 Oct 2026 11:29), reproduced unchanged:*

Mirror. I expect model A to mostly pass: C_M should be small relative to E⁽⁵⁾, and E_M close to E⁽⁵⁾. Figure 1(a) shows training motion in every direction (no empty bins; roughly 0.5–1.5× the uniform level), so mirrored trajectories are familiar, and because convolutions apply the same local rule everywhere, the motion rule should transfer. The main risk is appearance: mirrored 2, 3, 4 and 7 never appear in training. Any C_M should therefore come from blurred or distorted digits, not wrong motion.

Rotation. I expect rotation to be harder: C_R and E_R larger than C_M and E_M. The rotated directions are still familiar, but a 90°-rotated digit is further from the training data than a mirrored one: sideways digits never occur, and some become ambiguous (e.g. a sideways 1 resembles a dash). If the motion rule depends on recognising digit shape, rotation should expose this more than mirroring.

Speed. Figure 1(b) gives a median speed of about 3.3 px/frame, so 2× ≈ 6.6 and 3× ≈ 10 px/frame. I estimate model A’s reach at 8 px per step (Q2.3). 2× is just within reach, so predictions should degrade (as 2× motion is unseen in training) but stay recognisable. 3× exceeds the reach, so I expect breakdown: digits that stall, smear or vanish, with E⁽²⁾ approaching or exceeding Copy-last. I predict breakdown at 3×, with clear degradation already at 2×.

*Outcome (written afterwards):* the mirror and rotation parts held (Table 3). The speed part was too optimistic: model A already breaks down at 2× (Table 4; see Q2.9).

## Q2.6

Blank and Copy-last commute with mirror and rotation by construction. Blank outputs zeros, and a transformed zero frame is still zero. Copy-last repeats the last input frame, and transforming the input and then copying gives the same result as copying and then transforming. So C_M = C_R = 0 exactly. Yet they are the two worst predictors: E⁽⁵⁾ is 4.26e-2 for Blank and 6.41e-2 for Copy-last, against 2.18e-2 for model A (Table 3). C_T only asks whether the model treats a transformed clip the same way it treats the original. It says nothing about whether either prediction is right, so a trivial or badly wrong model can be perfectly consistent. Blurry predictions also lower C_T, because hedged outputs are similar to each other whatever the input. E_T supplies what C_T misses: accuracy on the transformed clips. A model has learned a rule that carries over only if C_T is small and E_T is close to its untransformed E⁽⁵⁾. For model A, C_M = 1.86e-3 is small next to E_M = 2.25e-2, and E_M is only 3% above E⁽⁵⁾.

## Q2.7

Video 9928 under rotation (Figure 3, T = R). The video shows two overlapping 9s; they overlap in all 20 frames.

- Row 2, the true rotated future: the digits move straight up at about 3 px per frame. Their centre of brightness rises from row 43 to row 32 over the five horizons and stays near column 17.
- Row 4, R(f(X)), the model's prediction on the original clip, rotated afterwards: this follows the truth closely. It rises about 3 px per frame, from row 42 to row 30, drifts only about 1 px sideways, and stays fairly compact: the brightest pixel falls from 0.78 to 0.67.
- Row 3, f(R(X)), the model's prediction on the rotated clip: this tells a different story.
  - The upward motion slows to about 2 px per frame, ending near row 36 instead of 32.
  - The digits drift about 4 px to the right, a direction the true digits never move.
  - The digits smear. The bright area grows from 325 to 434 pixels and the brightest pixel falls from 0.80 to 0.58, so by h = 5 the two 9s are an unrecognisable blob.

The per-horizon consistency gap grows from 2.1e-3 at h = 1 to 1.2e-2 at h = 5. So on the rotated clip the model loses the digits' speed and direction, and the error compounds as its own blurred predictions are fed back.

## Q2.8

Yes. In Table 3, C_R is 3.04e-3 against 1.86e-3 for C_M, about 1.6× higher, and C_R is higher for both seeds. E_R is 8% above E⁽⁵⁾, while E_M is only 3% above. Figure 4 shows this isn't driven by a few outliers. The whole per-video C_R histogram sits to the right of C_M. C_R exceeds C_M in 89% of test videos, with medians of 2.58e-3 and 1.66e-3. By Q2.1, the difference is unlikely to come from the motion. After either transformation every direction still occurs in the training data. The rotated direction histogram differs from the original by 20% against 13% for the mirrored one, and most of that is sampling noise from about 100 trajectories. By Q2.2, the appearance does differ. A mirrored 0, 1 or 8 looks almost unchanged, and other mirrored digits keep their upright stroke structure. A 90° rotation turns every digit sideways, which is further from anything in training. So the extra difficulty most likely comes from appearance: the motion rule model A learned appears tied to upright digit shapes. Model A alone can't rule out a small motion contribution. Comparing it with model B, which saw every symmetry during training, tests this (Q2.13).

## Q2.9

Model A falls apart at 2×. Its E⁽²⁾ rises from 1.84e-2 at 1× to 4.45e-2 at 2×, which is 2.4 times higher and already worse than predicting a blank frame. Blank's E⁽²⁾ is about 4.26e-2 at every speed. At 3× it reaches 5.14e-2. It stays below Copy-last (6.61e-2 and 7.03e-2), so it still extracts something from the input (Table 4). This only partly agrees with my Q2.3 reach of 8 px per step. 3× (about 10 px/frame) is beyond that reach, so failure there is expected. 2× (about 6.6 px/frame) is within it, yet the model still fails. What my estimate missed is that reach is only an upper bound on how far information can move in one step. It says nothing about what the model learned to do. Every training speed was below about 5.7 px/frame, so the learned update moves digits only about 3 px per step. Figure 5 (video 9021) shows this:

- At 2×, the predicted digits lag behind. In the first predicted frame, the brightness moves only about a third of the true displacement (1.4 px against 4.6 px).
- At 3×, the predictions don't follow the digits at all. They stay near where the digits were in the last input frame and smear out.

A digit that stalls and blurs suggests the model is applying the slow motion rule it learned and hedging over the plausible positions near that slow guess. It is not tracking the fast digit.

## Q2.10

No. Two explanations predict the same failure:

- Unfamiliar motion: model A never saw motion above about 5.7 px/frame.
- Reach: 3× exceeds its 8 px reach.

Both say A should fail at 3×, and A failing at 2× even though 2× is within reach hints at unfamiliarity, but with only model A the two causes can't be separated. The experiment is to vary one factor at a time:

1. Train the same architecture with speed augmentation, i.e. 1×, 2× and 3× clips. This is model C.
2. If C fixes both 2× and 3×, the failure was unfamiliar motion.
3. If C fixes 2× but still fails at 3×, the 3× failure is reach.
4. Confirm by keeping the speed-augmented data and enlarging only the reach (model D). D should then succeed at 3× where C did not.
