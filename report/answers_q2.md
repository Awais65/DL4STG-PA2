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

## Q2.11

*Prediction committed before training models B–D (`f6fecaa`, 29 Oct 2026 18:59), reproduced unchanged:*

Model B (symmetry augmentation) should shrink model A's mirror and rotation gaps, with C_R falling more than C_M because rotated digits were the less familiar appearance; it should leave the speed failure unchanged. Model C (speed augmentation) should fix most of the 2× degradation, since 2× motion (≈6.6 px/frame) is within the 8 px reach and was only unfamiliar, but it should still fail at 3× (≈10 px/frame), which exceeds the reach. Model D (larger reach, also speed-augmented) should be the only model that does well at 3×, which would show the 3× failure is architectural reach, not missing data.

## Q2.12

Model D is model A with a dilation of 2 in the 3×3 convolutions of both ConvLSTM layers. The padding is 2, so the grid stays at 16×16. Everything else matches model A: patch size 4, 64 hidden channels, the 1×1 output, the 9-step rollout, and the same training protocol with speed augmentation as C. Dilation doesn't add any weights, so D has 480,784 parameters. That is the same as A and well below the 721,176 cap.

A dilated 3×3 kernel connects cells up to 2 apart (8 px) instead of 1. Following the same paths as in Q2.3:

- (i) from u_τ through layer 1 (2 cells) and then layer 2 (2 cells): 4 cells, or 16 px;
- (ii) from h⁽¹⁾_{τ−1} through the same two convolutions: also 16 px;
- (iii) from h⁽²⁾_{τ−1} through layer 2 only: 8 px.

D's reach is therefore 16 px per step. That is strictly more than A's 8 px and above the roughly 10 px/frame motion at 3×. I picked D using the validation split (`results/model_d_selection.md`):

- **Dilation 2:** lowest validation E⁽²⁾ at 3×, 3.12e-2, compared with 3.28e-2 for C.
- **Dilation 3** (24 px reach) and **patch 8** (16 px reach, 594,496 parameters): both stayed at the error level of a near-blank prediction for most of the 30 epochs.
- **5×5 kernel:** 1,332,752 parameters, which is over the cap.

This design has a cost. A dilation-2 kernel only connects cells two apart, so the folded grid splits into four separate 8×8 sub-grids (odd/even rows × odd/even columns). These never share information. I confirmed this directly. Each output cell of D depends on 64 input cells, all in its own sub-grid. Each output cell of A depends on 169 input cells spread across all four (`results/answer_stats.json`, "coupling"). D gains reach by giving up spatial resolution.

## Q2.13

- **Mirror and rotation failure: appearance.** Model B has the same motion as A but sees every orientation during training, and the penalty disappears. Its E_M and E_R both equal its E⁽⁵⁾ (2.24e-2 each), whereas A had +3% and +8%. C_R drops from 3.04e-3 to 1.60e-3 (−47%) and C_M from 1.86e-3 to 1.28e-3 (−31%). Speed augmentation on its own doesn't remove the rotation penalty, since C's E_R is still 4% above its E⁽⁵⁾. Part of the lower C_M for C and D comes from blurrier predictions (Q2.6). For those models, E_T − E⁽⁵⁾ is a better comparison than C_T.
- **2× failure: unfamiliar motion.** C uses the same architecture with speed augmentation, and it reduces E⁽²⁾ at 2× from 4.45e-2 to 2.88e-2 (−35%). That is well below Blank's 4.26e-2. B, which never saw fast clips, doesn't help (4.47e-2).
- **3× failure: mostly unfamiliar motion, with a small reach effect.** C alone reduces E⁽²⁾ at 3× from 5.14e-2 to 3.34e-2 (−35%). Doubling the reach in D only brings it down to 3.17e-2 (a further −5%), although the gap holds across seeds (C 3.33e-2 / 3.35e-2, D 3.17e-2 / 3.16e-2). C probably copes at 3× despite its 8 px reach because a digit is about 20 px wide. After a 10 px jump, part of the digit is still within 8 px of where it was.

The clean result the question describes (C fails at 3×, D succeeds) didn't happen here. If it had, the 3× failure would be architectural. C and D see the same data and differ only in reach, and more data can't fix a reach limit. In a single step, each cell's update only sees cells within its reach. If a digit jumps further than that, the cell at its new position gets no information from its old position, and no choice of weights can connect the two. Data sets the weights, but it doesn't change which cells are connected.

## Q2.14

Yes, speed augmentation costs accuracy at 1×:

- E⁽⁵⁾: 2.18e-2 for A, 2.72e-2 for C (+25%), 2.87e-2 for D (+32%).
- E⁽²⁾ at 1×: 1.84e-2, 2.37e-2 and 2.64e-2.

A likely reason is that the same capacity now has to cover three speeds. Only a third of training clips are at 1×, and the model has to work out the speed from five frames. D loses a further 11% at 1× compared with C, probably because of its four decoupled sub-grids (Q2.12). Symmetry augmentation, by contrast, costs very little: B's E⁽⁵⁾ is 2.24e-2, only +3%.

Whether the trade-off is worth it depends on the inputs:

- **If clips can arrive at any speed,** speed augmentation clearly wins. Averaged over 1×, 2× and 3×, E⁽²⁾ is 3.81e-2 for A, 2.86e-2 for C and 2.91e-2 for D.
- **If only 1× clips matter,** A is still best.

D's extra reach gained 5% at 3× but lost 11% at 1×, so on these numbers it isn't worth choosing over C. B's consistency gain is worth its 3% cost.

## Q2.15

- **Exactness.** Built-in symmetry is exact, but learned symmetry isn't. Take
  f̄(gX) = (1/8) Σ_{h∈G} h⁻¹ f(hgX).
  Let k = hg. As h runs over G, so does k, and h⁻¹ = g k⁻¹. Each term becomes g k⁻¹ f(kX). g only rearranges pixels, so it can be moved outside the average:
  f̄(gX) = g · (1/8) Σ_{k∈G} k⁻¹ f(kX) = g f̄(X).
  This means C_M = C_R = 0 for any forecaster. Table 6 confirms it up to floating-point error: about 5e-17 for symmetrised B and 3e-17 for symmetrised D. Model B learned the symmetry from augmented data, and it only brings the gaps down to C_M = 1.28e-3 and C_R = 1.60e-3. Augmentation encourages consistency but doesn't guarantee it.
- **Accuracy.** Wrapping didn't cost any accuracy. Symmetrised B's E⁽⁵⁾ is 2.17e-2, compared with 2.24e-2 for B. Since the wrapper is exactly consistent, E_M = E_R = E⁽⁵⁾ = 2.17e-2. At 2× and 3×, E⁽²⁾ is 4.41e-2 and 5.01e-2, against 4.47e-2 and 5.10e-2 for B. Symmetrised D matches D: 2.86e-2 against 2.87e-2.
- **Inference cost.** It needs eight forward passes, about 2.0 s for the 1,000 test videos compared with 0.23 s for the unwrapped models, so roughly 8.7× slower. Training cost is unchanged.
- **When averaging should hurt.** Averaging eight predictions blurs the result wherever they disagree, for example when the model does well on upright clips but badly on rotated ones. The average then combines sharp, correct predictions with wrong ones. On average this didn't hurt here (see the accuracy point above), but the blur shows up on hard videos. In Q2.16, symmetrised D's brightest pixel on video 9928 is only 0.58, compared with 0.80 for model A's prediction on the same rotated clip.

## Q2.16

Only partly. In Figure 6(a), rows 3 and 4 of the final model for video 9928 under rotation are now identical, with a per-horizon C of about 3e-17. The disagreement between them is gone, and symmetrisation alone fixed it. But the shared prediction still goes wrong in the same way as model A's row 3:

- The two 9s still rise too slowly, at about 1.9 px per frame against the true 3. They end 3 px short of the truth.
- They still drift right, from column 22 to 25, while the true digits stay near column 17.
- They are dimmer from the first frame, with a brightest pixel of 0.58 falling to 0.50. This is blur from the averaging.

Its error on the rotated clip is 2.00e-2 at h = 1 and 2.99e-2 at h = 5. Model A's is 1.65e-2 and 2.91e-2, so the final model is no better on this video. Speed augmentation and reach don't come into it, because this is a 1× clip. The overlap between the two 9s probably accounts for some of the remaining failure. Symmetrisation made the model consistent, not correct.

The speed failure, on the other hand, is fixed. On the Figure 5 video (9021, Figure 6(b)), E⁽²⁾ at 2× drops from 3.70e-2 to 2.04e-2, and at 3× from 3.92e-2 to 2.25e-2. That improvement comes from speed augmentation and reach, at a small cost at 1× (1.96e-2 against 1.70e-2).

## Q2.17

Partly. The final model, symmetrised D, beats Blank and Copy-last at every speed and gets worse only slowly as clips speed up. Its E⁽²⁾ is 2.64e-2, 2.91e-2 and 3.14e-2 at 1×, 2× and 3×, compared with 1.84e-2, 4.45e-2 and 5.14e-2 for model A. This speed robustness is the best evidence that it learned a motion rule. A model that had memorised typical training clips would fail exactly here, as model A did. Its exact mirror and rotation consistency (C_M = C_R = 0) is not evidence, because symmetrisation guarantees that for any forecaster. But it is less accurate than model A on 1× clips (E⁽⁵⁾ 2.86e-2 against 2.18e-2), its predictions blur over the five horizons, and on video 9928 it still slows and drifts the two overlapping 9s. One test it hasn't passed is keeping overlapping digits sharp and on their true paths for all five horizons.
