# Q1 — River Forecasting Proposal

Service: every 10 minutes, at issuance time t₀, forecast the water level one hour ahead (t₀ + 60 min) at the four main-river stations S1–S4.

The design in one sentence: a **directed river graph** whose nodes are the reservoir, the stations and a latent node for the unmonitored tributary. Each node first runs its own causal **temporal encoder**. **Routing blocks** then pass information downstream along the river edges, using **flow-dependent travel-time attention** over past lags. Finally a **quantile head** predicts each station's level change from now to t₀ + 60 min.

---

## 1. Architecture diagram

In tensor shapes, B = batch, H = 72 past steps of 10 min and d = 64 features.

```
 DATA SOURCES (as available at t0)            ALIGN to 10-min grid, mask + age          ENCODE (per node)
 ---------------------------------            ---------------------------------         ------------------------
 Levels S1-S4, T1   (B,72,5,3) --------------> forward-fill, mask, age --------+
 Rain R1, R2        (B,72,2,3) --------------> 10-min sums, mask, age ---+     |
 Rain totals 24h/72h (B,2,2)  ---------------------------------------------+   |     node embedding  s_v
 Reservoir release, past  (B,72,1,3) ------> actual rate, mask, age ---+   |   |     + time encoding p_t
 Announced release, future (B,6,1,3) ---+                              |   |   |
 Static: graph, distances, gauge->node  |                              v   v   v
                                        |                  +---------------------------------+
                                        |                  | project each source to d=64     |
                                        |                  | onto its graph node             |
                                        |                  |  Res <- release                 |
                                        |                  |  S1..S4, T1 <- own level        |
                                        |                  |  T1, S2 <- R1   U <- R2         |
                                        |                  +---------------------------------+
                                        |                         node tensor (B, 72, 7, 64)
                                        |                                   |
                                        |          +------------------------v-------------------------+
                                        |          |  x K=4:  T block (causal dilated temporal conv)   |
                                        |          |          -> Routing block on river graph:         |
                                        |          |             Res->S1->S2->S3->S4,  T1->S2,  U->S3  |
                                        |          |             lag attention over l = 0..36 (6 h)    |
                                        |          +------------------------+-------------------------+
                                        |                                   |  take t0 slice of S1..S4
                                        |                                   v  (B, 4, 64)
                                        |        future-release encoder     |
                                        +-----> MLP -> (B, 64) ----------> concat <---- calendar X_cal (B, 4)
                                                                            |
                                                                  Quantile head per station
                                                                            v
                                         OUTPUT  Y_hat (B, 4, 3): level at t0+60 for S1..S4,
                                                 quantiles 0.1 / 0.5 / 0.9
                                                 (+ auxiliary path (B, 6, 4, 3) for t0+10..t0+60)
```

The graph has seven nodes: Res, S1, T1, S2, U (the unmonitored tributary, a latent node with no measurements), S3 and S4. Its edges are directed downstream.

---

## 2. Assumptions

These are assumptions, not facts. The supplied facts are the station layout, the 10-minute cadence, which tributary is monitored (T1, joining between S1 and S2) and which is not (joining between S2 and S3), the available sources, and that future rainfall and future water levels are unknown at issuance.

- **A1 — Common clock and datum.** All timestamps share one timezone (UTC) and the stations report levels on a common, stable datum. *Why reasonable:* this is standard for an authority-run network. *Effect:* records can be aligned on one 10-minute grid, and a level change at one station is comparable with one at another.
- **A2 — Record length.** At least 2–3 years of history, including several high-flow events. *Why:* the authority runs flood preparedness, so archives should exist. *Effect:* enough data to learn rainfall response and flow-dependent travel times. With a shorter record, the design leans more on the physics-based prior (A5).
- **A3 — Reporting delays.** Levels and rainfall normally arrive within 10–20 minutes of measurement, with occasional outages of hours. Each record carries both its measurement time and its arrival time. *Effect:* inputs are built "as of" t₀ from arrival times (§5), and each source gets an age feature.
- **A4 — Response times.** Travel time from the reservoir to S4 and the rainfall-to-river response are both under about 12 hours. Antecedent wetness is summarised by 24-hour and 72-hour rainfall totals. *Effect:* a history window of H = 72 steps (12 h) is long enough, and each routing edge needs lags of at most 6 hours.
- **A5 — Fixed flow direction, monotone response.** Water flows only downstream, with no significant backwater from below S4. More release or more rainfall never lowers downstream levels. *Effect:* the graph's edges are directed, and the lag-attention prior is built from the distances along the river. Monotonicity is not built into the network; it is used as a check in verification (a larger release must not lower the forecast).
- **A6 — Gauge-to-catchment mapping.** R1 mostly represents the catchment draining between S1 and S2, including T1's basin. R2 represents the unmonitored tributary's catchment. *Why:* the gauge locations are supplied, but which areas drain where is assumed. *Effect:* R1 feeds nodes T1 and S2, and R2 feeds the latent node U. If the mapping is wrong, a learned gauge-to-node projection can partly correct it.
- **A7 — Release schedules.** Schedules are announced before issuance and cover at least the next 60 minutes. Actual releases are recorded within the reporting delay. Past announcements are archived *as announced*, not overwritten later by the actual values.
- **A8 — Stable level–flow relation.** Each station's rating curve is stable: no ice, and no large changes to the channel within the record. *Effect:* the model can work in level directly, without converting to discharge.
- **A9 — Measurement quality.** Sensor noise is a few centimetres, smaller than forecast-relevant changes, and occasional spikes are filtered out by simple quality-control rules.
- **A10 — Stationarity.** Apart from seasonal patterns, the system's behaviour doesn't change over the record. Reservoir operating rules may change; this is handled by retraining (§6).

---

## 3. Forecasting task and data specification

- **Issuance time:** t₀ = every 10 minutes, on the 10-minute grid (:00, :10, …, UTC). An input counts only if it **arrived** by t₀.
- **History window:** H = 72 steps, covering t₀ − 710 min up to t₀ (12 h, A4). This is supplemented by 24-hour and 72-hour rainfall totals.
- **Horizon:** the level at t₀ + 60 min at S1–S4. An auxiliary output predicts the full path t₀ + 10 … t₀ + 60, which is used only as a training aid.

| Name | Type | Shape | Axes (in order) | Time coverage |
|---|---|---|---|---|
| `X_level` | input | (B, 72, 5, 3) | batch; time step; station [S1, S2, S3, S4, T1]; [level (forward-filled, normalised), observed mask, age in min] | t₀−710 … t₀ |
| `X_rain` | input | (B, 72, 2, 3) | batch; time; gauge [R1, R2]; [rain depth per 10 min, mask, age] | t₀−710 … t₀ |
| `X_rainctx` | input | (B, 2, 2) | batch; gauge; [24 h total, 72 h total] | to t₀ |
| `X_rel_past` | input | (B, 72, 1, 3) | batch; time; reservoir; [actual release rate, mask, age] | t₀−710 … t₀ |
| `X_rel_future` | input | (B, 6, 1, 3) | batch; future step; reservoir; [scheduled rate, announced flag, min since announcement] | t₀+10 … t₀+60 |
| `X_cal` | input | (B, 4) | batch; [sin, cos time of day; sin, cos day of year] | t₀ |
| `G` | static | 7 nodes, 6 edges + distances; gauge→node map | — | fixed |
| `Y_hat` | output | (B, 4, 3) | batch; station [S1…S4]; quantile [0.1, 0.5, 0.9] | t₀+60 |
| `Y_aux` | output (training only) | (B, 6, 4, 3) | batch; horizon; station; quantile | t₀+10 … t₀+60 |

**Alignment.**
- **Levels:** placed on the 10-minute grid by their measurement time. A missing slot is forward-filled, with mask = 0 and the age counting up.
- **Rainfall:** summed over each 10-minute interval ending at a grid time.
- **Releases:** treated as a step function sampled at grid times. Past values come from actual records, future values from the announced schedule.

Each source is mapped onto its graph node with its own learned linear projection to d = 64. The reservoir release goes to Res, each station's level to its own node, rainfall to the nodes given by A6, and nothing to U except R2. This turns sources with different numbers of locations and features into one node tensor of shape (B, 72, 7, 64).

**Available at issuance:** past levels, rainfall and actual releases that arrived by t₀, plus release schedules announced by t₀. **Not available:** future rainfall, future levels, any record that arrives after t₀, and any measurement of the unmonitored tributary.

---

## 4. Model computations and design justification

**Notation.** The graph has nodes V = {Res, S1, T1, S2, U, S3, S4} and directed edges E = {Res→S1, S1→S2, T1→S2, S2→S3, U→S3, S3→S4}. Time steps τ = 1…H, with τ = H corresponding to t₀. h_{v,τ} ∈ ℝᵈ is node v's representation at time τ.

**(1) Input embedding.** For each node v,

h⁽⁰⁾_{v,τ} = W_{src(v)} [x_{v,τ}; m_{v,τ}; a_{v,τ}] + Σ_g B_{gv} W_rain [r_{g,τ}; m_{g,τ}; a_{g,τ}; c_g] + s_v + p_τ,

where s_v is a learned node embedding, p_τ is a time encoding, B_{gv} ∈ {0,1} is the gauge-to-node map from A6, and c_g holds gauge g's 24 h and 72 h rainfall totals from `X_rainctx` (the same at every τ). The latent node U has no level term, so its embedding comes only from s_U, p_τ and R2's rainfall.

**(2) T block.** This is a causal dilated temporal convolution over each node's own history, with a residual connection: h ← h + TCN(h). It uses kernel 3 with dilations 1, 2, 4, 8 and 16, giving a temporal receptive field of 63 steps per block; the K = 4 stacked T blocks together cover far more than the 72-step window. Being causal, it never looks past t₀.

**(3) Routing block (flow-dependent lag attention).** For each edge u→v, the message is a weighted sum over the upstream node's recent past:

z_{u→v,τ} = Σ_{ℓ=0}^{L} α_{uv,ℓ}(τ) · W_e h_{u,τ−ℓ},   L = 36 steps (6 h),

α_{uv,·}(τ) = softmax_ℓ( q(h_{v,τ}, h_{u,τ})ᵀ k_ℓ + b_{uv,ℓ} ),

and the node is updated with h_{v,τ} ← h_{v,τ} + MLP([h_{v,τ}; Σ_{u∈in(v)} z_{u→v,τ}]). Here q is a small MLP that forms a query from the current states of both nodes, and k_ℓ is a learned embedding of lag ℓ. The bias b_{uv,ℓ} is initialised as a Gaussian centred on the expected travel lag, the reach's length divided by a typical wave speed. This is the physical prior. Because the attention query depends on the current state, a high-flow state can move the attention to shorter lags, matching flood waves that travel faster. The mapping from flow state to lag is learned.

**(4) Depth and reach.** The longest path in the graph is Res→S1→S2→S3→S4, which is 4 hops. So K = 4 routing blocks, each preceded by a T block, are needed for a release or upstream event to influence S4 within one forward pass. Temporal reach covers the 72-step window, and each hop adds up to 6 hours of lag.

**(5) Forecast head.** At issuance, for each main station s and quantile q ∈ {0.1, 0.5, 0.9}:

ŷ_{s,0.5} = y_{s,t₀} + g_s([h⁽ᴷ⁾_{s,H}; c_rel; c_cal]),   ŷ_{s,0.1} = ŷ_{s,0.5} − softplus(·),   ŷ_{s,0.9} = ŷ_{s,0.5} + softplus(·),

where c_rel = MLP(`X_rel_future`) encodes the announced releases and c_cal encodes the calendar features. The model predicts the *change* from the last observed level, a residual on persistence. The softplus offsets keep the quantiles in order.

**(6) Training objective.** The loss is a pinball (quantile) loss, masked where the true value is missing:

𝓛 = Σ_s Σ_q w_s(y) · m_s · ρ_q(y_{s,t₀+60} − ŷ_{s,q}) + λ · 𝓛_aux(t₀+10…t₀+50),   with ρ_q(e) = max(q·e, (q−1)·e).

Here λ (e.g. 0.1) weights the auxiliary path loss. The weight w_s(y) up-weights levels above the authority's alert thresholds, so the rare floods that matter most aren't swamped by normal-flow periods. The training data mimics real operations: sensors and intervals are dropped at random, and arrival delays are simulated (§5).

**Justification and influence paths.**
- **Why a graph, not a grid or full attention.** The river's connectivity is known, so the model's paths follow it: each station receives information only from what is physically upstream. Lag attention costs O(|E|·H·L·d) per block. That is small, and it avoids the cost of full space–time attention, which grows with the square of the number of tokens. A 4-hop graph covers the whole river, whereas a local convolution would need far more depth to reach that far.
- **Physical versus learned.** Physical assumptions: flow direction, connectivity, causality, the lag prior from distances, and predicting the change from persistence. Learned: the travel-time attention weights, the rainfall-to-level response, the latent tributary's inflow, how far actual releases deviate from schedules, and the uncertainty.
- **What drives each forecast.**
  - **S1:** reservoir releases (past actual values through Res→S1, plus announced future values through c_rel) and its own recent level. No rain gauge covers the catchment above S1 (A6).
  - **S2:** S1 routed downstream, T1 routed downstream (the monitored tributary), and R1 rainfall.
  - **S3:** S2 routed downstream, plus the latent U node, which is driven by R2 rainfall. The model learns how much S3 rises beyond the routed S2 signal.
  - **S4:** everything above, through four hops.

---

## 5. Operation and exceptional conditions

**(i) Missing observations.** Each missing slot is forward-filled, with mask = 0 and the age feature counting up. Because training drops sensors and stretches of time at random, the model learns to rely more on upstream stations and rainfall when a station's own record is stale.
- **If a station's own level is missing at t₀:** the forecast is still issued from its last value and the upstream information. It is flagged, and its quantile interval widens with the age feature.
- **If a key input is missing for more than about 6 hours:** for example, S1 offline while a release is under way. The system falls back to a simple routing-plus-persistence model and labels the forecast as degraded.

**(ii) Delayed observations.** Inputs are taken "as of" t₀ from arrival times (A3). A record that arrives late is never used for a forecast that was issued before it arrived, so training and operation see exactly the same information. The age features tell the model how fresh each source is, and training simulates realistic delays. Late records enter the next issuance 10 minutes later, so the forecast is revised then. Issued forecasts are archived unchanged, for verification.

**(iii) Announced future releases.** The schedule for t₀+10 … t₀+60 enters through `X_rel_future` and c_rel, with an "announced" flag and the time since announcement. Only schedules announced by t₀ are used. Releases that started earlier but are still travelling downstream are already in `X_rel_past` and reach later stations through the routing blocks.

**(iv) Scheduled versus actual releases.**
- **Past versus future:** past inputs always use the **actual** recorded release, and only the future uses the schedule.
- **Training:** training uses historical schedules *as they were announced* (A7). The model therefore learns how far actual releases typically deviate from the schedule and builds that into its uncertainty.
- **Detecting a mismatch:** the last actual release minus the last scheduled release is added as an extra input to c_rel. It exposes a current mismatch, which widens the S1 interval.
- **Correction:** once the actual release is recorded, the next issuance uses it.

**Feasibility every 10 minutes.** Each forecast processes one window of 7 nodes × 72 steps × 64 features through 4 T blocks and 4 routing blocks over 6 edges. That is about 400,000 parameters and on the order of 10⁸ multiply-adds per forecast, which takes tens of milliseconds on a CPU. The real work each cycle is in data ingestion and quality control, and both are routine. There are two levels of fallback: the routing-plus-persistence model, and persistence alone.

---

## 6. Uncertainty, limitations and additional information

Each limitation below gives its impact on the forecasts and the response.

- **The unmonitored tributary.** Its inflow between S2 and S3 is never measured. It is modelled only through the latent node U, driven by R2 rainfall, and learned from how much S3 rises beyond the routed S2 signal.
  - *Impact:* larger errors at S3 and S4 during local storms over that catchment, especially storms R2 doesn't capture.
  - *Response:* the quantile intervals widen with R2 rainfall and with recent unexplained rises at S3, and verification reports S3 and S4 performance separately.
  - *Unresolved:* storms that miss both gauges.
- **Changing travel times.** Travel times change with river conditions, which the flow-dependent lag attention handles.
  - *Impact:* in extreme floods outside the training range, the attention weights have to extrapolate and the predicted timing may be off.
  - *Response:* flood events are up-weighted in training, the lag prior is physics-based, and the intervals widen when levels go beyond the training range.
- **Sparse rainfall and unknown future rain.** Two gauges can miss convective cells, and future rainfall isn't available.
  - *Impact:* errors for rain that starts after t₀ and reaches a station within the hour. This matters mostly for fast local runoff near S2 and S3.
  - *Response:* this is accepted for a one-hour horizon, where most of the signal is already in the river and is reflected in wider intervals.
- **Reservoir operations.** Unannounced or changed releases are unknowable in advance. They are partly covered by learned schedule-deviation uncertainty (§5 iv), and the forecast corrects within one or two issuances.
- **Non-stationarity and sensor drift.** Channel changes, a new rating curve, a datum change or new operating rules all break A8 and A10. *Response:* retrain on a rolling schedule, monitor errors continuously, and alert operators to data-quality problems.
- **Unspecified data conditions.** The record length, reporting delays and quality flags aren't known (A2, A3, A9). If they turn out worse than assumed, the history window, the fallback thresholds and the training-time simulation all have to be retuned.

**Additional information to request.** The system remains usable without any of these.

| Request | How it helps | Usable without? |
|---|---|---|
| A temporary or permanent gauge on the unmonitored tributary, or its catchment area and land use | Turns U from a latent node into an observed one, or gives a physical prior for its inflow, cutting S3 and S4 errors | Yes, through the latent node and R2 |
| Rating curves and channel cross-sections at S1–S4 | Converts levels to discharge, gives physically based travel times, and helps extreme floods | Yes, the model learns in level space |
| Archive of past release schedules *as announced*, plus operating rules | Lets the model learn how schedules deviate from actual releases (§5 iv) | Partly. Without it, the deviation uncertainty is set by hand |
| Arrival-time logs and quality flags | Lets the "as-of" simulation match real delays and outages | Yes, with assumed delays (A3) |
| Rainfall radar or short-range rain nowcasts | Fills gaps between the two gauges and adds rain expected after t₀ | Yes |
| Temperature and snow data (for the mountain catchment) | Captures seasonal snowmelt, which the rainfall inputs miss | Yes, seasonality is partly captured by calendar features |
| Historical flood reports and alert thresholds | Sets the loss weights w_s, and serves as verification cases | Yes, with thresholds estimated from the data |

The model treats uncertainty through its quantile outputs, trained with the pinball loss. Their calibration is checked on held-out years by asking whether 80% of outcomes fall inside the 0.1–0.9 band, both overall and for high-flow periods.
