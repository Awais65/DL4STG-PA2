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
