## Model D selection (validation split, videos 8000-8999, t=1, final epoch)

| Run (seed 0) | Architecture | Params | val E^(5) 1x | val E^(2) 1x | val E^(2) 2x | val E^(2) 3x | s/epoch |
|---|---|---|---|---|---|---|---|
| A | k=3, dil=1, patch=4 | 480,784 | 2.2092e-02 | – | – | – | 16.4 |
| C | k=3, dil=1, patch=4 | 480,784 | 2.6867e-02 | 2.3371e-02 | 2.8782e-02 | 3.2769e-02 | 18.6 |
| Dcand_dil2_s0 | k=3, dil=2, patch=4 | 480,784 | 2.8230e-02 | 2.5986e-02 | 2.8937e-02 | 3.1245e-02 | 7.0 |
| Dcand_dil3_s0 | k=3, dil=3, patch=4 | 480,784 | 3.7389e-02 | 3.7414e-02 | 3.7385e-02 | 3.7393e-02 | 7.1 |
| Dcand_patch8_s0 | k=3, dil=1, patch=8 | 594,496 | 3.9468e-02 | 3.9490e-02 | 3.9463e-02 | 3.9456e-02 | 4.2 |

Lowest validation E^(2) at 3x: **Dcand_dil2_s0**.
