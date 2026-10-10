# Three certificates for skipping a patch, and which norm is the right currency

Setting. A fixed camera produces frames; a patch `Ω` (P×P pixels) has a reference `R` (its last kept copy, whose token the model holds) and a new value `F`. Write `d = F − R`. A *skip rule* is a statistic `σ(d, R)` and a threshold; a *certificate* is a theorem of the form "if `σ < θ` then nothing of the following kind happened in `Ω`". The original rule is `σ = range(d) = max d − min d` with the persistence-stability certificate. This note develops three alternatives, each from a different piece of mathematics, states their certificates with proofs, and says what each one buys.

Notation. `‖·‖∞` is the sup norm over `Ω`. For a function `g` on `Ω`, `Dgm(g)` is its sublevel-set H0 persistence diagram (dark blobs); superlevel sets give bright blobs; `d_B` is the bottleneck distance. **Stability** (Cohen-Steiner, Edelsbrunner, Harer 2007): `d_B(Dgm(g), Dgm(h)) ≤ ‖g − h‖∞`.

---

## 0. Why the range rule fails on real cameras

Every statistic below is a norm of the change, and the choice of norm is the whole story:

| norm of `d` | certifies | cost of i.i.d. noise σ | cost of a step edge jittering by δ px |
|---|---|---|---|
| `L∞` (range) | every object of contrast Δ, any size | ≈ 7.5 σ (max of 256 samples) | full contrast × δ, on a 1-px strip |
| `L1` / mean (EVS, TimeChat, RLT) | objects of *mass* (area×contrast) | ≈ 0.8 σ × area | contrast × δ × edge length |
| flat norm (transport) | mass created / destroyed / moved > ℓ | between the two | same as `L1`: a strip is single-signed mass |
| multi-scale box max | objects containing a (2r+1)-box, contrast Δ_r | ≈ 7.5 σ / (2r+1) at scale r | contrast × δ / (2r+1) |
| `L∞` *after quotienting translations* | objects of contrast Δ, up to a sub-pixel move | ≈ 7.5 σ | ≈ interpolation error only |

A sub-pixel jiggle of a high-contrast edge changes a whole one-pixel column by the full contrast. That is a real change of real mass; no norm of `d` alone makes it small. Two things do: dilution by area (multi-scale) and explanation by motion (quotient). Transport does not, and Section 1 says so honestly.

---

## 1. Transport certificate (flat norm)

**Definition.** For a signed mass `e` on the grid and a scale `ℓ > 0`,

```
‖e‖_ℓ = min { (1/ℓ) Σ_edges |J| + Σ_x |r(x)| :  div J = e − r }
```

mass may be moved along grid edges at cost (L1 distance)/ℓ per unit, or created/destroyed at cost 1 per unit. Equivalently (Kantorovich–Rubinstein duality, a finite LP so strong duality is exact)

```
‖e‖_ℓ = sup { ⟨f, e⟩ :  |f| ≤ 1,  |f(x) − f(y)| ≤ |x − y|₁ / ℓ }.
```

Rule: `c = mean(d)`, `e = d − c`, drop iff `UB(e) < τ` where `UB ≥ ‖e‖_ℓ` is the quadtree bound below.

**Theorem 1 (conservation).** If `‖e‖_ℓ < τ` then for every region `S ⊂ Ω`, with `f_S(x) = max(0, 1 − dist₁(x, S)/ℓ)`, `|⟨f_S, e⟩| < τ`.
*Proof.* `f_S` is feasible for the dual. ∎

*Reading.* `⟨f_S, e⟩` is the net mass change in `S` plus a tapered ring of width ℓ. An object of area `A` and contrast `Δ` appearing contributes `AΔ` to it; the only way to stay under τ is an opposite change of mass `≥ AΔ − τ` within ℓ pixels. So: nothing of mass ≥ τ is created or destroyed, and mass `m` that moves by `δ` costs `m·min(δ, 2ℓ)/ℓ`, hence motion of ≥ 2ℓ pixels by mass ≥ τ/2 is always kept.

**Theorem 2 (linear frontends).** For any filter `w` on `Ω`, `|⟨w, e⟩| ≤ ‖w‖*_ℓ ‖e‖_ℓ` with `‖w‖*_ℓ = max(‖w‖∞, ℓ·max_{x∼y}|w(x) − w(y)|)`.
*Proof.* `w/‖w‖*_ℓ` is dual-feasible. ∎
So a ViT patch embedding (a linear map with rows `w_k`) moves by at most `max_k ‖w_k‖*_ℓ · τ` on a dropped patch; the constant is computed from the weights.

**Tightness.** `e = Δ·1_S` on a flat patch (far from the boundary, mean removed) has `‖e‖_ℓ = AΔ` up to the cost of spreading the compensating mean, so a threshold τ cannot be raised without admitting a τ-mass creation.

**Computation (quadtree bound).** Build a quadtree over the (zero-padded, shifted) patch. Move every cell's net mass to its parent's centre at cost `2^(j−1)/ℓ` per unit at level `j` (the L1 distance between child and parent centres), cancel opposite masses that meet, destroy whatever remains once `2^(j−1)/ℓ ≥ 1`, and destroy the root remainder. The tree metric dominates the grid L1 metric, so this flow is feasible on the grid and its cost is an upper bound. Minimising over offsets `0..2^⌈log₂ℓ⌉−1` per axis covers every placement of the cell boundaries at the transporting levels. It is a few reshapes and sums, vectorised over all patches of a frame. An exact value needs a min-cost flow and is only used for audits (not implemented here; the bound is sound regardless).

**Verdict.** Sound, cheap, and the guarantee reads like a conservation law, which is attractive for tracking. But a jittering step edge is single-signed mass, so transport cannot cancel it, and in the experiments the flat norm behaves like a mean rule: robust to noise, blind to small objects, no better than the range rule on jitter. It is kept as a baseline with a different currency, not as the answer.

---

## 2. Multi-scale certificate (box averages)

**Definition.** For half-widths `r ∈ 𝓡` let `M_r(e) = max_B |mean_B e|` over all `(2r+1)×(2r+1)` boxes `B ⊂ Ω`. With `c = midrange(d)`, `e = d − c`, drop iff `M_r(e) < ε_r` for all `r ∈ 𝓡`. For `𝓡 = {0}`, `ε_0 = Δ/2`, this is exactly the range rule.

**Theorem 3.** Let `β_r` be the `(2r+1)`-box mean filter (valid region). If the patch is dropped then for every `r ∈ 𝓡`:
(a) `‖β_r F − β_r (R + c)‖∞ < ε_r`;
(b) `d_B(Dgm(β_r F), Dgm(β_r (R + c))) < ε_r`, for dark and for bright blobs;
(c) no feature of `β_r F` with persistence ≥ `2ε_r` is born from or dies into the diagonal relative to `β_r R`; and every such feature's contrast changes by less than `2ε_r`.
*Proof.* (a) is `M_r(e) < ε_r` restated since `β_r` is linear and `β_r c = c`. (b) is stability applied to the filtered images. (c) follows from (b) as in the original rule. ∎

*Reading.* "An object of contrast Δ_r = 2ε_r that contains a (2r+1)-box" is, after box filtering, a feature of persistence ≥ Δ_r; (c) says it cannot appear while dropped. Thin things (a 1-px strip) are not such objects at scale r ≥ 1: that is the point.

**Tightness.** A change of `±ε_r` on exactly one `(2r+1)`-box (midrange removed) has `M_r = ε_r` and creates a `2ε_r` feature of `β_r F`; no rule reading only the `M`-statistics can drop at `M_r ≥ ε_r`.

**Noise and schedules.** For i.i.d. noise of std `σ_d` on `d`, `mean_B e` has std `σ_d/(2r+1)`, and the max over ≈ (P−2r)² positions grows only logarithmically. A strip of width 1 and contrast C contributes `C/(2r+1)`. So with `ε_r ∝ 1/(2r+1)` nothing is gained (noise, strips and thresholds all scale alike); the gain comes from a *flatter* schedule `Δ_r = Δ_0 (2r+1)^(−γ)` with `γ < 1`: relative to objects, noise is suppressed by `(2r+1)^(1−γ)` and strips likewise. `γ = ½` is the midpoint and is what the benchmark sweeps.

**Computation.** Integral images: all boxes of all sizes in O(P²) per patch, vectorised over the frame.

---

## 3. Quotient certificate (certify modulo a motion group)

The group `G` of sub-pixel translations acts on patches; translating a scene does not create or destroy anything. Certify the change *modulo G*.

**Definition.** For each patch choose `g = (δ_y, δ_x) ∈ [−δ_max, δ_max]²` and let `W = T_g R` be the reference resampled (bilinearly, from the full reference frame) at `x + δ`. Let `d' = F − W`, `c = midrange(d')`, `e = d' − c`, `s = range(d')`. Drop iff `s < Δ`.

How `g` is chosen is a heuristic and does not affect soundness: one or two Gauss–Newton steps on brightness constancy `F(x) ≈ R(x + δ) + c`, i.e. the 3×3 normal equations with `R_x, R_y` the reference gradients,

```
[ΣR_x²  ΣR_xR_y  ΣR_x] [δ_x]   [ΣR_x d]
[ΣR_xR_y ΣR_y²   ΣR_y] [δ_y] = [ΣR_y d]
[ΣR_x   ΣR_y     n   ] [c  ]   [Σ d   ]
```

followed by clamping to `δ_max`. The certificate is evaluated against the `W` actually constructed.

**Theorem 4.** If the patch is dropped then `‖F − (T_g R + c)‖∞ < Δ/2` with `|g|∞ ≤ δ_max`, hence `d_B(Dgm F, Dgm(T_g R + c)) < Δ/2`: no feature of contrast ≥ Δ appears, vanishes, splits or merges in `F` relative to the translate `T_g R` of the reference.
*Proof.* Range of `d'` below Δ gives the sup bound after the midrange shift; stability gives the rest. ∎

*Reading.* The model holds the token of `R`. The certificate says `F` is, up to `Δ`-contrast topology and a brightness shift, the reference moved by less than `δ_max` pixels. Edge jitter is absorbed into `g` instead of being paid for in `s`; a person walking in is not a translation of the background and stays in the residual. The certificate is weaker than the range rule's by exactly the quotient: sub-pixel position is not certified. For an interpolated translate the features of `T_g R` are those of `R` up to interpolation attenuation of one-pixel-scale detail; this is the one approximation in the chain and it is stated, not hidden.

**Tightness.** Same construction as the range rule applied to the residual: any residual with spread Δ can create a Δ-object in some scene.

**Why this is the interesting one.** The same statement holds for any group of warps for which `T_g R` can be formed: per-patch affine motion, or a per-frame homography estimated once, turns this into a certificate for *moving* cameras: "F equals the registered reference up to Δ-contrast topology". The fixed-camera limitation of the original rule is a limitation of the trivial group.

**Computation.** Gradients of `R` once per kept update, a 3×3 solve per patch, one bilinear resampling of the frame, then the range rule. Everything vectorises across the frame.

---

## 4. Combining 2 and 3

Quotient first, then multi-scale on the residual: jitter is explained by motion, remaining noise is diluted by area, and the certificate is Theorem 3 applied to `F` versus `T_g R + c`. This is the rule to use when a camera is both noisy and jittery.

---

## 5. Relation to prior work, stated plainly

Motion-compensated prediction with a residual test is every video codec since H.261; conditional replenishment is 1969; box filtering and scale-space blob detection are classical; the flat norm is Whitney's, with the quadtree bound following Indyk–Thaper / Charikar style embeddings. None of this is new. What is new is the *certificate*: for each statistic a theorem naming exactly what cannot have happened in a dropped patch, a tightness construction showing the threshold cannot be moved, and the observation that the right thing to quotient out is a motion group so that stability only has to pay for what the group cannot explain. The experimental claim is in `RESULTS.md`, Part II: at equal 100 % recall of planted objects on real CCTV footage, quotient + multi-scale keeps 2.7–5.5 % of patches where the range rule keeps 15–19 %.

---

## 6. Sequential certificate: the time axis (quickest change detection)

Every rule above is memoryless: it compares one frame with the reference and cannot see a change fainter than one frame's noise. **Sequential analysis** is the branch of statistics built for exactly this: Page's CUSUM (1954), Lorden's minimax formulation (1971), Moustakides' exact optimality of CUSUM (1986), Lai's window-limited GLR for unknown post-change parameters (1998), Shiryaev–Roberts for the Bayesian version. Its theorems say how early a persistent change *can* be detected at a given false-alarm rate, and which statistic achieves it. It is essentially unused in computer vision and has never been applied to token skipping.

**Setting.** Per patch, after the last keep at time `t₀`, the quotient residuals `e_{t₀+1}, e_{t₀+2}, …` are observations; "no change" means mean zero (up to the reference's own fixed noise, which is static and cancels below); "change at ν" means a persistent mean shift `μ·1_S` on some region `S` for `t ≥ ν`. Time, place, size and magnitude of the change are unknown.

**Statistic (window-limited GLR as a space-time scan).** For window lengths `w ∈ W` (dyadic) and box half-widths `r ∈ 𝓡`, with `A` the mean of `e` over the last `w` frames and `B` over the `w` frames before those,

```
Z_{w,r} = max_B |mean_B (A − B)| · (2r+1) · √(w/2) / σ
```

and keep iff `max_{w,r} Z_{w,r} ≥ z`. For a Gaussian mean shift at a known time, place and size this *is* the log-likelihood ratio (up to a monotone map); maximising over `(w, r, B)` is the generalised likelihood ratio; restricting `w` to a dyadic window set is Lai's window limitation. Spatially it is the multiscale box scan of Section 2, and Arias-Castro, Donoho and Huo (2005) prove such scans are minimax-optimal for detecting geometric objects in noise. `σ` is estimated per patch from temporal differences of the residual, so the threshold adapts to each region's noise.

**Self-normalisation (what the footage forced).** Measured on three cameras, the per-pixel residual noise was white-looking but *spatially correlated* along scan lines (analog) or in codec blocks, and on the two codec clips its variance *grew with the window length* (temporal drift). Under such noise the Gaussian scale `σ√(2/w)/(2r+1)` is wrong by a factor 2–3 at long windows and the box statistic fired on 40–90 % of static patch-frames. The remedy is classical in sequential analysis: *self-normalise*. Let `M_{w,r} = max_B |mean_B(A − B)|` be the extreme statistic per patch. Its null location `m_{w,r}` and spread `s_{w,r}` are learned per patch from quiet frames (frames on which neither rule fires), and the test is

```
Z'_{w,r} = (M_{w,r} − m_{w,r}) / s_{w,r},     keep iff max_{w,r} Z'_{w,r} ≥ z'.
```

Standardising the *maximum* rather than the field is deliberate: a scale estimated from overlapping box positions is biased low by their correlation (measured: 15 %), whereas the empirical distribution of `M` over time is nonparametric in the noise's spatial correlation and temporal spectrum. A planar illumination term `(1, x, y)` is projected out per patch before accumulation so that smooth lighting drift is not mistaken for an object.

**Proposition 5 (false fires).** If the noise is stationary and the null of `M_{w,r}` is Gumbel-like with location `m` and spread `s` (the max of many weakly dependent variables), then for each patch, frame and test `P(Z' ≥ z') ≈ exp(−(1.14 z' + 0.4))`, and over the `|W|·|𝓡|` tests the union bound applies: `z' = 8` gives about `10⁻³` per patch-frame. Measured: `3·10⁻⁴` on white noise, `1.3·10⁻³` false fires on the hallway camera. (With independent Gaussian noise and the parametric scale one would instead have `P ≤ N(1 − Φ(z))`, `N = |W| Σ_r (P − 2r)²`, the a-contrario "number of false alarms" of Desolneux, Moisan and Morel; that bound is what the footage violated.)

**Proposition 6 (delay).** A persistent object of contrast `Δ` containing a `(2r+1)`-box that appears at frame `ν` is kept by frame `ν + w − 1` for the smallest `w ∈ W` with `Δ(1 − |S|/P²) ≥ m_{w,r} + (z' + z'')s_{w,r}`, with probability ≥ the null tail at `z''`. *Proof.* At `t = ν + w − 1` the recent window lies entirely after the change and the earlier window entirely before, so the box centred on the object has `mean_B(A − B) = Δ` less the planar-projection leakage, and `M_{w,r}` is at least that. ∎ On the hallway camera the learned null at `(w, r) = (4, 2)` has `m ≈ 1.0`, `s ≈ 0.18` grey levels, so `Δ ≈ 2.5` is detectable within four frames at `z' = 8`; the memoryless certificate at the same operating point needs `Δ ≥ 36` for a 5×5 box.

**Optimality.** No procedure can detect reliably while `Δ(2r+1)√w ≤ c·σ√(log N)` (the minimax lower bound for detecting a box of unknown position and scale in Gaussian noise, Arias-Castro–Donoho–Huo 2005; in time this is Lorden's bound), so the delay of Proposition 6 is within a constant factor, and the dyadic rounding within a factor 2, of the best possible.

**What is certified.** The sequential test only *adds* keeps to the memoryless rule, so every dropped patch still carries the worst-case certificate of Theorems 3–4. The sequential part adds a statistical guarantee that the memoryless rules cannot have: a persistent faint change is kept within a provable, near-optimal delay. Reference noise and any static bias cancel in `A − B`; independent noise averages down by `√w`; what does not cancel is real, persistent change, which is what the model should see.

**Two guards the breadth footage forced.** On codec-clean cameras static blocks repeat *exactly* between frames, so the learned spread `s_{w,r}` collapses to numerical zero and the codec's periodic requantisation (every 6–12 frames on the VIRAT 720p scenes) fires 94 % of the patches at once; one scene lost eleven points of drop rate to it. (i) A physical floor on the spread, `s ← max(s, s₀)` with `s₀ = 0.5` grey levels: a box-mean statistic cannot be resolved below a fraction of a quantisation step, so a null narrower than that is an artefact of the data, not of the scene. (ii) A global-coincidence veto: if more than a fraction `ν = 0.1` of all patches fire in the same frame, the event is scene-wide (keyframe, illumination) and not an object; no patch is kept and the frame is used to train the null, so the breathing enters `s`. Under the independence that Proposition 5 assumes, the expected number of simultaneous fires is `N·p ≪ νN`, so the veto never triggers on genuine objects; measured, it restores the lost drop rate exactly (0.883 → 0.998) and leaves faint-object recall unchanged (0.87–1.00 at contrast 8–16).

**Honesty about the model.** The first implementation used the parametric Gaussian scale and failed on all three cameras (drop rate 70 % instead of 95 %); the self-normalised version is what the results report. Its guarantees are statistical and conditional on stationarity of the camera noise over the learning window; illumination flicker that is neither smooth nor static is correctly treated as change. `RESULTS.md` Part III reports measured false-fire cost, delays, and the faint-object recall the memoryless rules cannot reach at any threshold.

---

## 7. From patches to pixels, in real time: tracking the minimiser of a time-varying energy

The skip rules decide per 16×16 patch. A segmentation mask is a decision per pixel, and the natural per-frame formulation is a Potts model: label each pixel foreground or background so as to balance the evidence against the length of the boundary. Its exact solution is available through a theorem of Chan, Esedoğlu and Nikolova (2006): the convex relaxation

```
min_{u ∈ [0,1]^Ω}  ⟨f, u⟩ + λ Σ_x w(x) |∇u(x)|
```

has, by the coarea formula, a global minimiser whose every superlevel set `{u > s}`, `0 < s < 1`, is a global minimiser of the binary problem. `f = γ(z₀ − z̃)` is the studentised, shadow-suppressed residual evidence (negative where the change is significant) and `w = exp(−|∇ẽ|²/2τ²)` makes the boundary cheap exactly where the residual has an edge, so silhouettes snap to objects. Chambolle–Pock solves it; sixty iterations per frame give the mask in `out/seg_hall_full.mp4`.

**The video problem.** Solving each frame from scratch treats a video as a pile of photographs. The energy at frame `t` differs from the energy at `t − 1` by a small drift of the evidence, and its minimiser differs by a small motion of the objects. Treat it as what it is, a **time-varying convex optimisation** (Simonetto, Dall'Anese, Paternain, Leus & Giannakis, *Proc. IEEE* 2020), and *track* the minimiser:

```
E_t(u) = ⟨f_t, u⟩ + λ Σ w_t |∇u| + (μ/2) ‖u − û_t‖²,     û_t = prediction from frame t−1.
```

The proximal term is the video prior (labels move continuously) and makes `E_t` `μ`-strongly convex, which is what buys the theorem below. The prediction advects each tracked object by its own velocity, estimated from the displacement of its centroid; the correction is `K` primal–dual steps started from the warm start.

**Proposition 7 (tracking error).** Let `u_t*` minimise `E_t`, let `ρ < 1` be the contraction factor of one primal–dual step on the strongly convex problem, and let `d_t = ‖u_t* − û_t‖` be the prediction error (the drift the predictor failed to foresee). Then the iterate after `K` steps satisfies `‖u_t − u_t*‖ ≤ ρ^K (‖u_{t−1} − u_{t−1}*‖ + d_t)`, hence in steady state `‖u_t − u_t*‖ ≤ ρ^K d / (1 − ρ^K)`: a fixed multiple of one frame's unforeseen drift, independent of the length of the video.
*Proof.* Contraction of the fixed-point map toward `u_t*`, triangle inequality through `û_t`, geometric series. ∎ (This is the prediction–correction bound of Simonetto et al. specialised to a fixed step count.)

**Event-driven correction.** Which pixels can have moved? Those where the evidence changed coherently (the Part-II multi-scale statistic on the residual, at half its certified threshold), those near an object's current or predicted position, those whose labels were not stationary last frame, and a one-patch halo. Only there are the primal and dual variables updated; elsewhere `u`, `p` and the mask are reused. This is the segmentation analogue of the skip rule: work scales with what changed, not with the frame.

**Proposition 8 (certificate for the reused labels).** `E_t` is convex, so a point is optimal iff the KKT residual vanishes everywhere: for each pixel `g = −div(w p) + f + μ(u − û)` must be `0` for `0 < u < 1`, `≥ 0` at `u = 0`, `≤ 0` at `u = 1`, with `|p| ≤ λ`. The residual is evaluated on the active set every frame and on each frozen patch when it was last touched; a frozen patch is reactivated as soon as its residual exceeds `η`. Hence at every frame the whole labelling is `η`-stationary for the current energy, frozen region included, and the gap to the exact minimiser is bounded by the residual times the diameter of `[0,1]^Ω`.

**What this buys, measured.** Against the fully converged per-frame solution of the same evidence, the tracked mask has mean IoU 0.94 (median 0.96) with three steps per frame; the KKT residual on the frozen region stays below `0.05` on 95 % of frames. Cost: 6 ms per frame at 352×288 and 29 ms at 1280×720 in single-threaded C++ (`cpp/rtseg.hpp`), 60 iterations replaced by 3 and full frames replaced by the 5 % of patches that are active. The numpy implementation runs the same mathematics at 25 ms per CIF frame.

**What is and is not new.** Background subtraction, Potts MRFs for segmentation, graph cuts and TV relaxations, morphological clean-up and overlap tracking are all standard. Prediction–correction tracking of a time-varying convex program is standard in control and signal processing and, to my knowledge, has not been used to make a segmentation energy follow a video; the certified-skip-driven active set and the KKT certificate for reused labels follow directly from this project's earlier parts. The mask quality is that of a good classical pipeline, not of a learned segmenter; what is new is that it is delivered at video rate with a provable relation to the per-frame optimum.

### 7.1 Crisper and faster: three additions, with the mathematics that justifies each

**Lagrangian evidence accumulation (the per-pixel sequential test, in the object's own frame).** Part III accumulated evidence per patch to see faint persistent objects. The same idea works per pixel once the accumulator moves with the object. Let `s_t(x) = e_t(x)/σ(x)` be the signed studentised residual and `v` the tracked velocity of the object covering `x`. Define

```
A_t(x) = ρ · A_{t−1}(x − v) + s_t(x),      z_acc(x) = |A_t(x)| · √(1 − ρ²).
```

*Lemma.* If `s_t` has unit variance and is independent across frames, `A_t` has variance `1/(1 − ρ²)`, so `z_acc` is again unit-variance under the null; if `s_t ≡ k` persists along the object's trajectory, `A_t → k/(1 − ρ)` and `z_acc → k·√((1+ρ)/(1−ρ))`, i.e. `2k` for `ρ = 0.6`. *Proof.* Geometric series for the mean; for the variance, `Var(A_t) = ρ² Var(A_{t−1}) + 1` has fixed point `1/(1−ρ²)`. ∎ The unary uses `max(z̃, z_acc)`: instantaneous strong evidence is kept, persistent weak evidence is doubled. Without the advection a fast object leaves a trail of accumulated evidence behind it, which is exactly what happened when the per-frame displacement bound was left at the CIF value of 8 pixels on 1080p footage; the bound is now 48 pixels.

**Boundary weights from both the residual and the image.** `w = min(w_res, w_img) + 0.05`, with `w_img = exp(−|∇F̃|²/2τ_img²)`: a boundary is cheap wherever either the residual or the image has an edge, which is where an object's contour lies; inside the background, where the data term is decisive, cheap edges are never used.

**Guided-filter snap.** After clean-up the binary mask is passed through the guided filter of He, Sun and Tang (2010) with the frame as guide, in a band of radius `r+1` around the contour, and re-thresholded at ½. The filter is the solution of a local linear model `q = aF + b` fitted in each window, so the contour moves onto the nearest intensity edge of strength above `√ε` and leaves textureless regions where they are. Being a weighted average of the mask, it cannot create foreground far from the contour, and it is O(1) per pixel.

**Ghosts (what high-resolution footage taught).** When an object is present during background initialisation, or sits still long enough to be absorbed, the spot it later vacates differs from the model: a ghost. A ghost and an object are distinguished by where their outline lives: a ghost's outline is in the background model `B` and absent from the frame `F`; a real object's is in `F` and absent from `B`. For each component the contour contrast is measured on both images (the largest studentised intensity step within 3 px of each boundary pixel, averaged along the contour); a component whose contrast in `F` is below a floor, or below 0.6 of its contrast in `B`, is a ghost: it is not reported and its pixels are absorbed at rate 0.25 per frame. The second source of ghosts is removed at the root: objects are no longer absorbed into the background while they are objects (`α_fg = 5·10⁻⁵`), which also means a parked car or an abandoned bag remains an object, as it should.

**Event-driven cost, measured at 1920×1080.** With the active set defined by the studentised multi-scale statistic (a box-mean of `z` has standard deviation `1/(2r+1)` under the noise model, so the threshold is in sigma units and foliage whose variance has been learned stops being "change"), the active fraction is 5–10 % of pixels; hole filling and the snap band are computed inside each object's bounding box; the per-patch stages run in parallel in a four-colour schedule (patches of one colour are never adjacent, so their one-pixel margins never overlap and no two threads write the same pixel). Per frame on one M4: building scene 35 ms (29 fps), street scene with wind in the trees 52 ms (19 fps). The remaining cost is the fixed per-frame work on the full frame (jitter fit, illumination, background and noise update, component labelling), about 24 ms, which is the next thing to make sparse.

---

## 8. Predictive look scheduling with a distribution-free guarantee, and why it does not beat uniform sampling

**Where prediction can pay.** At the patch level the certified rule already decides before any token is computed, so predicting a skip saves a few arithmetic operations and nothing else. One level up, a video model decides *when to look at the stream*, and it does so blind today (fixed 1–2 fps). A predictive scheduler that lengthens its horizon when the scene is quiet and shortens it when things happen saves decoding and inspection and lowers delay. Nothing deterministic can be certified about frames nobody looked at, so the only honest guarantee is distribution-free and statistical.

**Setting.** After a look at frame `t`, choose a horizon `h_t ∈ {1, …, H}`, skip `h_t − 1` frames, look again. At the next look the certified rule refreshes patches with its deterministic certificate. Define a *miss* as `err_t = 1` iff `h_t ≥ 2` and the refreshed set contains at least `m` patches outside the one-patch dilation of the set active at the previous look: an *onset* arrived in a region not being tracked while we were not looking. If no frame was skipped nothing can have been missed, `err_t = 0`. Ongoing activity is not a miss; its staleness is reported separately.

**Policy (risk equalisation with a safety net).** With `λ_b` the running estimate of the onset rate per frame in activity state `b ∈ {active, recent, quiet}`, `h_t = clip(round(e^{θ_b} α / λ_b), 1, H)`: the predicted onset probability per window is `α` in every state, so for a constant rate the policy is uniform at the `α`-calibrated rate. The multiplier `θ_b` is an online integrator on the miss rate, one per state: `θ_b ← min(log H, θ_b + γ_n (α − err))`, `γ_n = γ / √n` (Gibbs and Candès 2021 in the log-horizon domain; the 2024 critique that this is a control property rather than conformal inference is correct, and the name used here is the honest one).

**Proposition 9 (one-sided, deterministic, distribution-free).** For *any* frame sequence, within each state `b` with `N_b` decisions, `(1/N_b) Σ err ≤ α + (θ_1 + log H)/Σ_n γ_n`. *Proof.* The ceiling can only lower `θ`, so `θ_{n+1} ≤ θ_n + γ_n(α − err_n)` unless the floor `−log H` binds; the floor binds only when `err = 1` at a `θ` where the horizon is already 1, and there `err = 0` by definition, so it never binds. Summing, `Σ err ≤ N_b α + (θ_1 − θ_{N+1})/γ ≤ N_b α + (θ_1 + log H)/γ`; the decaying-step version replaces `1/γ` by `1/Σγ_n`. ∎ The overall miss rate obeys the `N_b`-weighted sum. No lower bound is claimed: in a quiet scene the horizon saturates at `H` and the miss rate stays below `α`, which is the desired behaviour. Every persistent Δ-change is seen within `H` frames regardless.

**Proposition 10 (the structural limit).** If onsets form a memoryless process with rate `λ` independent of the observable state, then for any schedule with mean horizon `h̄`, the expected miss rate is `≈ λ h̄` to first order and the mean staleness of motion is `E[h²]/2h̄ ≥ h̄/2`, with equality iff the schedule is uniform. Hence no adaptive schedule improves on uniform sampling in (looks, misses, staleness), and adaptivity can only pay through state-dependence of the onset rate. *Proof.* Linearity of the miss probability in `h` for small `λh`; Jensen for the staleness. ∎

**What the footage says** (`RESULTS.md` Part V). The guarantee held in all twelve runs. On continuously busy indoor cameras, new activity appears in 17–92 % of frames, so no schedule can skip under a 10 % onset-miss constraint and the scheduler correctly looks at every frame. On the quiet parking-lot camera the learned onset rates differ between quiet and active states by only a factor 2–3, so there is little for risk equalisation to exploit: at equal cost the adaptive scheduler ties uniform on misses and onset delay and is worse on tracking staleness, exactly as Proposition 10 predicts. The scheduler's genuine contributions are self-calibration (it finds the rate for a given `α` without a hand-set horizon) and the guarantee; it is not a better sampler than a well-tuned uniform rate on this footage. This is the result that justifies the main design: looking at every frame with the cheap certified statistic costs almost nothing and makes the deterministic certificate possible, which no predictive scheme can offer.

---

## 9. Which image is certified: the offset gap

**The gap.** Every certificate in Sections 2–4 is a statement about `T_g R + c`: the held copy, moved by a sub-pixel shift `g` and re-levelled by a brightness offset `c`, both fitted per patch on the frame being tested. A model that reuses tokens holds `R`. The shift is harmless for what is claimed, since a translate has the features of `R` at positions less than a pixel away (Section 3). The offset is not. Inside one patch it slides the persistence diagram along the diagonal and changes no contrast, which is why Sections 2–4 could drop it. Across patches it is a piecewise-constant field `c_p`, and

```
F − T_g R = c_p + e_p      on patch p,   with only e_p bounded by the rule.
```

The distance between the truth and the held copy is therefore not bounded by any threshold. Whatever change lives in the field `c_p` is never sent:

* a flat object on a flat surface, once it covers whole patches: its interior has `d ≡ contrast` and range 0;
* any change of lighting: on a flat patch `d ≈ (gain − 1)·mean(R)`, again with a small range.

The functions called `view()` return `T_g R + c`, so the overlays, the media in the README and the end-to-end experiment of `RESULTS.md` Part VII all showed the re-levelled image, not the held copy. Measured on the 48 blind-spot trials of Part VII, the square is present in the re-levelled view in 48 trials and in the held copy in 27; on the parking lot, where the square covers 26 whole patches, in 0 of 16 (`RESULTS.md`, Part VIII).

**Local and global symmetry.** A brightness shift is a symmetry of the topology of an image, and Sections 2–4 use it once per patch. An image has that symmetry once per frame at most, and so does a model looking at it: a frame that is brighter everywhere means the same, a frame that is brighter in some patches does not. The rules quotient by the product of one group per patch where the consumer is invariant only under the diagonal. In the language of gauge theory, a symmetry made local needs its gauge field carried along. Here that would be `c_p` and `g_p` for every dropped patch of every frame, and a model able to apply them to a cached token. Short of that, only the global part may be divided out.

**The rule without offset.** Take `c = 0`. With `e = F − T_g R`, drop iff `M_r(e) < ε_r` for all `r ∈ 𝓡`; in the range form, drop iff `2‖e‖∞ < Δ`. In the code this is `offset="none"` on the quotient and sequential pruners, in numpy and in C++; `offset="patch"` is the rule of Sections 3–6 and remains the default, so every earlier number is reproduced unchanged.

**Theorem 11 (certificate against the held copy, for the frame).** Let `V` be the image equal to `T_{g_p} R` on every dropped patch `p` and to `F` on every kept one. Under the rule without offset:
(a) `‖F − V‖∞ < ε_0` over the whole frame;
(b) `d_B(Dgm F, Dgm V) < ε_0` for the diagrams of the *frame*, dark and bright: no feature of contrast ≥ `Δ_0 = 2ε_0` is born, dies, splits or merges anywhere, features that extend over many patches included;
(c) for every `r ∈ 𝓡` and every `(2r+1)`-box `B` inside a dropped patch, `|mean_B (F − V)| < ε_r`. In particular the brightness of the held copy is within `ε_r` of the truth on every such box.
*Proof.* (a) On a dropped patch `M_0(e) < ε_0` is the sup bound, on a kept patch `F − V = 0`, and a bound that holds on every patch with no constant depending on the patch holds on their union. (b) is stability applied to the frame. (c) restates the rule. ∎

With the per-patch offset, (a) reads `‖F − (T_g R + c_p)‖∞ < ε_0`: true, about an image that exists only in the encoder. `V` differs from the held copy by the resampling of each patch at a shift of at most `δ_max`, the approximation already stated in Section 3; with `δ_max = 0` it is the held copy and (b) has no approximation in it.

**Corollary (what the offset hides).** Take a frame on which the rule with the per-patch offset drops every patch, with offsets `c_p`, and write `V = T_g R`. Then `‖F − (V + c)‖∞ < ε_0` for the patchwise-constant field `c`, and for every single level `m`

```
‖F − (V + m)‖∞ < ε_0 + ‖c − m‖∞,        best m:   ε_0 + ½·osc(c),     osc(c) = max_p c_p − min_p c_p.
```

Adding one constant to a whole image translates its persistence diagram and changes no contrast, so this is the frame-level statement that rule supports: bottleneck distance below `ε_0 + ½·osc(c)` up to one global level. The bound is attained (`V = 0`, `F = c`), and nothing in that rule bounds `osc(c)`. The term `½·osc(c)` is exactly what was missing from its certificate; the rule without offset is the case `c = 0`.

**Tightness of the level.** A uniform change `u` over a patch has `M_r = u` at every scale, so it is sent as soon as `u ≥ min_r ε_r`, the bound of the largest box, and not before. With the benchmark schedule `Δ_r = 80·(2r+1)^(−½)`, `r ≤ 3`, that is 15 grey levels. The rule with the per-patch offset has no such bound at any threshold.

**What changes besides the level.** A compact object of contrast κ on a flat patch gives `e = κ` on the object and 0 elsewhere, where the midrange gave `±κ/2`. At the same schedule the rule without offset therefore keeps an object containing a `(2r+1)`-box from contrast `ε_r` on, half of `Δ_r`. In the range form a patch it drops is one the rule with offset would drop too (`2‖d‖∞ < Δ` implies `range(d) < Δ`). In the multi-scale form neither rule's keeps contain the other's, because the midrange of a patch with one outlying pixel shifts every box mean.

**The sequential layer** is unchanged. Its statistic has a plane projected out per patch, so it cannot see a uniform change, by construction; the memoryless part now bounds what that can hide, by (c).

**What this does not handle.** A level change of the whole frame of `ε_r` or more resends the frame. That is the truth being sent, and it is the right behaviour for a light switch. For a camera whose gain hunts by a few grey levels it would be waste; one bounded level per frame could be divided out at no cost to a model that ignores global brightness. A prototype of that variant made no measurable difference on the six clips tried, so it is not in the code.

**Relation to prior work, stated plainly.** Lighting changes and the unseen interior of a uniform object are two of the canonical problems of background maintenance, the "light switch" and the "foreground aperture" of Toyama, Krumm, Brumitt and Meyers (*Wallflower*, ICCV 1999). Video codecs compensate brightness globally (weighted prediction in H.264). Neither is new. What is added is the observation that the *certificate* was stated modulo a constant per patch that no consumer holds, and Theorem 11, which is what the stability theorem gives once that constant is gone: a statement about the frame and about the held copy.

---

## 10. Frame rate: a threshold on speed against a threshold on displacement

**Setting.** A patch shows `x(s)` at time `s` seconds, and the camera samples it at `f` frames per second, `F_t = x(t/f)`. A *consecutive-frame* rule keeps the patch at frame `t` iff `σ(F_t − F_{t−1}) ≥ τ`, for a seminorm `σ`; the mean absolute difference is the choice of EVS-style pruning and of run-length tokenisation. A *reference* rule keeps it iff `σ'(F_t − R) ≥ ε`, with `R` the last kept copy; every rule of Sections 2–6 and 9 is of this kind.

**Proposition 12.** Suppose that on a time interval the patch changes at a bounded rate, `σ(x(s) − x(s')) ≤ L·|s − s'|`, with `L` in grey levels per second.
(a) If `f > L/τ`, the consecutive-frame rule keeps nothing on that interval. Its held copy stays what it was when the interval began, and the error `σ(F_t − held)` can reach `L` times the length of the interval. No choice of `τ` bounds it.
(b) The reference rule has `σ'(F_t − held) < ε` at every frame, for every `f`. On the interval it keeps at most `V/ε + 1` times, where `V` is the total variation of `x` in `σ'`: a number that does not depend on `f`.
*Proof.* (a) Every consecutive difference is at most `L/f < τ`. (b) The bound is the rule. Between two keeps the patch has moved by at least `ε` in `σ'`, and the sampled path is no longer than the continuous one. ∎

*Reading.* `σ(F_t − F_{t−1}) ≥ τ` asks for a rate of change of at least `τ·f` grey levels per second. A consecutive-frame threshold is a threshold on *speed*, and it rises in proportion to the frame rate. A reference threshold is one on *displacement*, and the frame rate does not enter it. At one or two frames per second the two coincide for most of what happens in front of a camera, and that is the rate at which such rules are normally run. At the camera's own rate the consecutive-frame rule is blind to everything slower than `τ·f`, which includes the far side of every object that leaves a patch gradually: the copy it holds keeps part of what is no longer there.

*Measured.* On eleven fixed cameras, read at rates from 1 fps to their native 25 or 30 fps, with the consecutive-frame rule re-tuned at every rate to the skip rate of the rule of Section 9: the share of the static part of the frame that is wrong in the held copy has median 0.12 % at 1 fps and 1.07 % at the native rate, against 0.08 % and 0.09 % for the reference rule (`RESULTS.md`, Part IX; `scripts/framerate_bench.py`).

*What is and is not new.* That consecutive-frame rules miss slow change is the blind spot this project started from, and Part I of the results measures it on synthetic fades. The statement in terms of frame rate, and its measurement on real footage with no planted object, are what this section adds.

---

## 11. What to hold: the fewest sends a certificate allows

Every rule above holds a *frame*. When the certificate breaks, the current frame is sent, and it is held until the certificate breaks again. This section asks what the fewest sends are for a given certificate, whatever is held, and how far a rule that holds frames is from that. The answer depends only on the geometry of the certificate's norm.

**Setting.** One patch. Frames `x_0, …, x_{n−1}` in a normed space `(V, ‖·‖)`, where `‖·‖` is the norm of the certificate: the sup norm for the range rule without offset, and `‖e‖ = max_r M_r(e)/ε_r` for the multi-scale rule without offset (a norm, because `r = 0` is in the schedule; tolerance 1). A *policy* chooses held copies `h_0, …, h_{n−1} ∈ V`. It *sends* at `t = 0` and at every `t` with `h_t ≠ h_{t−1}`. It is *certified* at tolerance `ε` if `‖x_t − h_t‖ < ε` for every `t`, and it has *latency* `L` if `h_t` depends only on `x_0, …, x_{t+L}`. What is held need not be a frame. *Send-on-delta* is the policy of latency 0 that holds `x_s` from a send at `s` until the first `t` with `‖x_t − x_s‖ ≥ ε`, and sends `x_t` then.

Two ways of cutting the clip greedily into maximal runs of consecutive frames, starting a new run at the first frame that breaks the condition:

* `D_η(x)`: the number of runs when a run must have diameter `< η`;
* `C_ε(x)`: the number of runs when a run must be *coverable*, meaning that some point of `V` is within `ε` of all its frames.

A coverable run has diameter `< 2ε`, and a run of diameter `< ε` is covered by any of its own frames, so `D_{2ε}(x) ≤ C_ε(x) ≤ D_ε(x)`.

**Theorem 13 (the fewest sends).** Every policy certified at tolerance `ε`, with any latency, sends at least `C_ε(x)` times, and the policy that holds a covering point of each greedy coverable run sends exactly `C_ε(x)` times.
*Proof.* Between two sends the held copy is one point within `ε` of every frame, so the frames between two sends form a coverable run; and a sub-run of a coverable run is coverable. Let the greedy runs start at `g_1 = 0 < g_2 < …` and the policy send at `s_1 = 0 < s_2 < …`. Then `s_k ≤ g_k` for all `k`. It holds for `k = 1`. If `s_k ≤ g_k` and `s_{k+1} > g_k`, the frames `g_k, …, s_{k+1} − 1` lie between two sends, so they are coverable, and the greedy run that starts at `g_k`, being maximal, contains them: `s_{k+1} ≤ g_{k+1}`. So when greedy starts its `k`-th run the policy has sent at least `k` times. The second claim is the definition of the greedy runs. ∎

**Theorem 14 (the price of holding a frame).** Send-on-delta at tolerance `ε` sends at most `D_ε(x)` times. Hence

```
D_{2ε}(x)  ≤  C_ε(x)  =  fewest sends of any certified policy  ≤  sends of send-on-delta  ≤  D_ε(x).
```

Both ends are attained, and their ratio is unbounded. By Theorem 13 at tolerance `ε/2`, no policy that guarantees `ε/2` sends less than send-on-delta does to guarantee `ε`: holding frames costs at most a factor two *in the tolerance*, and can cost any factor in the number of sends.
*Proof.* Let send-on-delta send at `s`, in the greedy run `[g, g')` of diameter `< ε`. Every frame of that run is within `ε` of `x_s`, so the next send is at `g'` or later: distinct sends lie in distinct runs. For the ends, take scalar frames. A staircase whose steps are `2ε` or more makes all five quantities equal to the number of steps. The flicker `x_t = a + (−1)^t·b` with `ε/2 ≤ b < ε` has consecutive frames `2b ≥ ε` apart, so send-on-delta sends on every frame and `D_ε = n`, while the single held copy `a` is within `b < ε` of every frame and `C_ε = D_{2ε} = 1`. ∎

**Theorem 15 (the sup norm: centres attain the bound).** In `ℓ∞^d` a set of frames is coverable at tolerance `ε` iff its diameter is `< 2ε`, and its coordinatewise midrange covers it. So `C_ε = D_{2ε}`, the bound is computed in one pass with a running maximum and minimum per pixel, and it is attained by holding the midrange of each run.
*Proof.* Coordinate by coordinate: numbers in an interval of length `< 2ε` are within `ε` of its midpoint. ∎

*Remarks.* (i) This is the binary intersection property of `ℓ∞`: balls that meet pairwise have a common point (hyperconvexity; Aronszajn and Panitchpakdi, 1956). Equivalently the Jung constant of `ℓ∞` is ½: a set of diameter `D` lies in a ball of radius `D/2`. In Euclidean space the radius can reach `D·√(d/(2(d+1)))` (Jung, 1901) and in a general norm `D·d/(d+1)` (Bohnenblust, 1938). The sup norm, in which the stability of persistence is stated, is therefore also the norm in which holding a centre gains the most over holding a frame: the full factor two of Theorem 14. (ii) The multi-scale norm is the sup norm of the linear features `e ↦ (mean_B e / ε_r)` over all boxes `B` and scales `r`, so the patch space sits isometrically inside a larger `ℓ∞`. `D_2` computed on the features is a lower bound by Theorems 13 and 14. It need not be attained, because the midrange of the features of several images need not be the features of an image.

**Theorem 16 (latency).** Let `A_L` be the policy that, on a break at frame `s`, holds a point within `ε` of all of `x_s, …, x_{s+j}` for the largest `j ≤ L` for which there is one. `A_L` has latency `L` and is certified at tolerance `ε`; `A_0` is send-on-delta; and, with `ℓ_k` the lengths of the greedy coverable runs,

```
C_ε(x)  ≤  sends of A_L  ≤  Σ_k ⌈ℓ_k / (L+1)⌉  ≤  C_ε(x) + n/(L+1).
```

In particular `A_L` is optimal on any clip whose coverable runs are no longer than `L + 1` frames.
*Proof.* The held point certifies frames `s, …, s+j` by choice, and any later frame it fails to certify is a break. Let `A_L` send at `s` inside the greedy run `[g_k, g_{k+1})`. If `j = L` the next send is at `s + L + 1` or later. If `j < L` and the clip has not ended, `x_s, …, x_{s+j+1}` is not coverable, while `x_s, …, x_{g_{k+1}−1}` is, so `s + j + 1 ≥ g_{k+1}`. Either way the next send is no earlier than `min(s + L + 1, g_{k+1})`: two sends in the same greedy run are at least `L + 1` frames apart. ∎

*Remark (no latency, no guarantee).* With latency 0 no policy is within a constant of `C_ε`. For scalar integer frames, whatever a zero-latency policy holds, the next frame can be placed `ε` or more away from it on the side that keeps at least half (less one) of the interval of points covering all frames so far; the policy must send, and this can be repeated about `log₂ ε` times while one point still covers every frame, so `C_ε = 1`. Holding the midpoint of that interval meets the bound up to a constant. With `d` pixels the adversary can do this one pixel at a time. Latency is what removes the logarithm, and Theorem 16 says how much of it is needed.

**The rule.** `python/certskip/centre.py`: `fewest_sends` computes `D_2` on the features, and `CentrePruner(lookahead=L)` is `A_L` with the pixelwise midrange as the point held. For the sup norm that is exactly Theorem 16. For the multi-scale norm the midrange is a candidate, accepted for a run only if it certifies every frame of the run, so the certificate holds whatever the candidate is; the lower bound still holds and the upper bound of Theorem 16 is not claimed. What the model is given is the midrange of a few consecutive frames, an image that was never a frame. The certificate says each of those frames is within `ε` of it, which is all it ever said about a held frame.

**What is and is not new.** Cutting a sequence greedily into the fewest pieces under a property that passes to sub-runs is textbook. Send-on-delta is the Lebesgue sampling of event-triggered control (Åström and Bernhardsson, 2002; Miśkowicz, 2006). Approximating a scalar time series by the midrange of greedy segments under a sup-norm bound, and its optimality in the number of segments, is the PMC-MR algorithm of Lazaridis and Mehrotra (2003). Hyperconvexity and Jung's constant are classical. What is added: the question asked of a skip certificate, with the patch as the unit that is sent; Theorem 14, which places every frame-holding rule of this project between the optimum at `ε` and the optimum at `ε/2`; the bounded-latency policy and Theorem 16; and the measurement, on real cameras, of how far the rule in use is from the bound and how much of the distance a few frames of latency recover (`RESULTS.md`, Part X).

---

## 12. Two invariances, and what breaks without them

Sections 9 to 11 are three answers to one question: of what, exactly, should a skip rule be a function? Stated abstractly, a clip is a path `x` in a metric space `(X, d)` of images, a group `G` acts on `X` by isometries, and the consumer is any map out of `X` that is 1-Lipschitz and unchanged by `G`. Two invariances are forced.

**In space: the consumer's group, and no larger.** For a group `H` acting by isometries let `d_H(x, y) = inf_{g ∈ H} d(x, g·y)`.

**Proposition 17.** (a) `d_G(x, y)` is the largest discrepancy `ρ(Φ(x), Φ(y))` over all `G`-invariant 1-Lipschitz maps `Φ` into (pseudo)metric spaces `(M, ρ)`: the quotient distance is the universal certificate for that class of consumers. (b) For images, with `H` and `G` linear subspaces of patchwise fields acting by addition and `d` the sup distance: a certificate `d_H(F, R) < ε` bounds `d_G(F, R)` iff `H ⊆ G`, and then by `ε`; if `H ⊄ G` there are `F, R` with `d_H(F, R) = 0` and `d_G(F, R)` as large as one likes.
*Proof.* (a) For such a `Φ` and any `g`, `ρ(Φx, Φy) = ρ(Φx, Φ(g·y)) ≤ d(x, g·y)`; and the quotient map `X → X/G` with the distance `d_G` is itself such a `Φ`. (b) If `H ⊆ G` the infimum over `G` is over more. If `c ∈ H \ G`, take `F = R + λc`: `d_H = 0` and `d_G = λ·dist(c, G)`, which is positive because `G` is closed, and grows with `λ`. ∎

The rules of Sections 2–6 took `H` = one brightness level per patch. A consumer is invariant at most under `G` = one level per frame, and `dist(c, G) = ½·osc(c)`: that is the corollary of Section 9, and the reason its fix is to shrink `H`, not to tune a threshold.

**In time: the path, not its parametrisation.** The quantities of Section 11 depend on the frames only through their order.

**Proposition 18.** (a) `D_η` and `C_ε` do not increase when frames are removed from a clip, and do not change when a frame is repeated. So for a path sampled on finer and finer grids they increase to a limit, the value for the path itself, which is at most `1 + V/η` for a path of total variation `V`. The fewest sends a certificate needs are a property of what happened, whatever the frame rate. (b) The number of keeps of a consecutive-frame rule has no such limit: for a path that changes at a bounded rate it is zero on every fine enough grid (Proposition 12).
*Proof.* (a) A cut of the full clip into runs of diameter `< η` restricts to a cut of the sub-clip into no more runs of diameter `< η`, and the greedy cut is the smallest (the argument of Theorem 13); likewise for coverable runs. A repeated frame extends the run it is in. A greedy run ends because the next frame is at distance `η` or more from one of its frames, so the path has at least `η` of variation between the start of each run and the start of the next. ∎

A threshold on `d(x_t, x_{t−1})` is a function of the parametrisation. A threshold on `d(x_t, h)` is a function of the path and of what is held. That is the whole difference between the two families of rules, and Part IX of the results is what it looks like on a camera.

*What is and is not new.* Quotient metrics and their universal property are standard, and so is the invariance of a length-type quantity under reparametrisation. Putting the two side by side as the two conditions a skip certificate has to meet, and reading the offset gap and the frame-rate law as the failure of one each, is this project's.
