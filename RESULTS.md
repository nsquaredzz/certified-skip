# Results (2026-10-08)

All numbers produced by the scripts in this repo on an Apple M4; patch 16 px, Δ = 32 grey levels unless stated. Figures and JSON are in `out/`.

The JSON, logs and plots in `out/` are tracked. The videos (`out/*.mp4`) and the large stills are not: regenerate them with the commands in the README, or see the short versions in `assets/`. Footage is fetched by `scripts/fetch_data.py`.

## 1. Real footage (the gating experiment)

Sources: Xiph.org derf test set (uncompressed CIF, 30 fps, 10 s; `hall_monitor` is a fixed CCTV-style hallway camera with two people walking, `akiyo` is a fixed-camera news anchor on a static backdrop) and the Blender Foundation open movie *Tears of Steel* (CC-BY, live action with cuts and camera motion, 720p H.264, 12 min).

| clip | camera | σ (robust) | drop @Δ=16 | @24 | @32 | @48 | Gaussian ceiling @32 |
|---|---|---|---|---|---|---|---|
| hall_monitor | fixed, analog-era CCTV | 1.7 | 0.26 | 0.61 | **0.73** | 0.87 | 1.00 |
| akiyo | fixed, clean digital | 0.0 | 0.82 | 0.87 | **0.89** | 0.92 | 1.00 |
| Tears of Steel, 2 fps | moving, cuts | 1.9* | 0.25 | 0.31 | **0.37** | 0.46 | 1.00 |
| Tears of Steel, 24 fps (C++ CLI) | moving, cuts | – | – | – | **0.58** | – | – |

\* inflated by motion; the robust estimator assumes half the patches are static.

Surveillance footage of people being tracked (added later the same day):

| clip | scene | fps analysed | drop @Δ=16 | @24 | @32 | @48 |
|---|---|---|---|---|---|---|
| VIRAT_S_000200_00 (Kitware public release) | parking lot, 1280×720 MPEG-4, people and cars | 10 | 0.97 | 0.99 | **0.99** | 0.99 |
| VIRAT_S_000200_01 | same camera | 10 | 0.95 | 0.98 | **0.98** | 0.99 |
| CAVIAR WalkByShop1cor (INRIA/Edinburgh) | shopping-mall corridor, 384×288 MPEG-1 | 25 | 0.83 | 0.90 | **0.93** | 0.95 |
| CAVIAR LeftBag | lobby, several people | 25 | 0.35 | 0.59 | **0.77** | 0.90 |
| CAVIAR Walk1 / Meet_Crowd | lobby | 25 | 0.36 / 0.37 | 0.61 / 0.61 | **0.78 / 0.77** | 0.90 / 0.89 |

The VIRAT clips are the canonical fixed-camera surveillance benchmark and come out at the synthetic level or above: the MPEG-4 encoder already zeroes static macroblocks, so the post-codec noise is tiny and almost all of the kept budget goes to the moving people and cars (see `out/virat_000200_00_overlay.mp4`). The CAVIAR lobby clips sit at 77 % because the lobby camera has visible MPEG-1 flicker on the bright floor reflections; the corridor camera of the same dataset reaches 93 %. All audits: 0 violations.

* **Clean static camera (akiyo): 89 % drop at Δ=32**, close to the synthetic 96 %.
* **Noisy static camera (hall_monitor): 73 %.** The Gaussian model predicts ~100 %, so the shortfall is not i.i.d. sensor noise. The overlay figure `out/hall_monitor_frame150.png` shows what is kept besides the walkers: vertical strips along high-contrast wall edges, i.e. sub-pixel edge jitter / analog line noise. Max−min is maximally sensitive to that; an averaging rule is not. This is the real-footage cost of the guarantee and must be reported.
* **Movie footage: 37 % at 2 fps, 58 % at 24 fps.** Per-30-second windows range from 11 % to 64 %, none above 80 % (`scripts` ad-hoc run). The guarantee is per patch for a fixed camera; on cinematic footage the rule is not useful without motion compensation. This confirms the paper's own caveat rather than extending the claim.
* **Soundness:** 500 + 500 + 300 dropped patches audited with exact bottleneck distances (dark and bright blobs); 0 violations, worst bottleneck/(s/2) = 1.000 (bound attained, never exceeded).
* **No codec keyframe spikes** were detected in these clips (the Xiph clips are uncompressed; the movie's GOP noise is below the detector's threshold at 2 fps).

Throughput of the C++ core on the full movie (17,620 frames at 1280×534, decode included): 836 frames/s, 571 Mpx/s. 1080p synthetic: ~1,150 frames/s.

## 2. Synthetic benchmark (matched drop budgets)

Three static scenes, 40 s at 2 fps, 640×360, H.264 crf 23, sensor σ = 2 before encoding. The certified rule runs at Δ = 32; every baseline is calibrated on an object-free control clip to drop **min(certified rate, 0.97)**, so baselines always have at least as much keep budget as the certified rule (certified dropped 0.99–1.00).

| rule | 5×5 px object, 1 frame (contrast 48) | fade-in over 8 s (8×8 px, to 64) | worst view error (grey levels) |
|---|---|---|---|
| certified (Δ=32) | 192 / 192 | 192 / 192 | 15.5 (proved bound) |
| consecutive-frame mean (EVS / TimeChat style) | 192 / 192 | 168 / 192 | not bounded |
| mean vs last kept | 192 / 192 | 192 / 192 | 46 (measured) |
| uniform sampling | 4 / 192 | 45 / 192 | not bounded |

Harder setting (crf 18, σ = 3, global flicker σ = 1.5): certified 192/192 and 192/192; consecutive mean 192/192 and **53/192**; mean-vs-kept 192/192 and 192/192 with measured worst error 30; uniform 2/192 and 49/192.

Sub-threshold small objects (3×3 px, contrast 40; mean contribution 40·9/256 ≈ 1.4 < τ ≈ 2.5): certified 192/192, consecutive mean 184/192, mean-vs-kept 187/192, uniform 5/192.

Soundness audit on 2,000 dropped synthetic patches: 0 violations, worst bottleneck/(s/2) = 0.875.

**Reading.** The slow-fade blind spot of consecutive-frame rules reproduces strongly (13–28 % missed, 72 % missed under flicker). The small-object blind spot of averaging rules only appears when size × contrast / patch² falls below τ; at 5×5/48 it does not, at 3×3/40 it does (4 % missed). Mean-vs-last-kept catches everything here but its view error is unbounded in principle and measured at 2–3× the certified bound. The claim to make is therefore: *same drop budget, same or better recall, and a proved error bound the others lack*, not *the others miss objects*.

## 3. What this changes in the paper's framing

1. Lead with static-camera streaming/monitoring. Report the movie number as the limit, not as a failure.
2. Report the heavy-tailed-noise cost (hall_monitor 73 % vs. 89 % on a clean camera) and the mechanism (edge jitter). A probabilistic variant that tolerates a bounded fraction of outlier pixels, with a correspondingly weakened guarantee, is the natural follow-up.
3. The baselines' failure mode is slow change, not small objects, at realistic sizes. Say so.
4. End-to-end accuracy with a real video LLM is still missing.


---

# Part II — beyond the range rule (2026-10-08, later)

Three further certificates were built from different mathematics (see `THEORY.md`): a **transport** (flat-norm) certificate, a **multi-scale box** certificate, and a **quotient** certificate that fits a sub-pixel translation per patch and certifies only the residual. `scripts/real_bench.py` plants objects into real footage and sweeps every rule's threshold, so rules with different certificates are compared on the one axis that matters operationally: drop rate at a given recall.

Planted per clip: 40 small objects (5×5 px, contrast 40, one frame), 40 tiny (3×3, contrast 48, one frame), 20 slow fades (8×8 rising to 48 over 3 s; must be kept by the time contrast reaches 32), 20 movers (6×6, contrast 40, 2 px/frame for 12 frames; "coverage" = fraction of those frames kept). Contrast sign always points away from saturation.

## Drop rate at 100 % recall of small, tiny and fade objects

| rule | hall_monitor | CAVIAR lobby | VIRAT | mover coverage (hall / lobby / VIRAT) |
|---|---|---|---|---|
| range (original) | 0.808 | 0.847 | 0.984 | 1.00 / 1.00 / 1.00 |
| flat norm ℓ=4 (≥95 % recall; never reaches 100 % on tiny) | 0.785 | 0.812 | 0.987 | 1.00 / 1.00 / 1.00 |
| multi-scale, γ=½ | 0.925 | 0.959 | 0.987 | 0.99 / 0.97 / 1.00 |
| multi-scale, γ=1 | 0.917 | 0.967 | 0.987 | 0.97 / 0.95 / 0.96 |
| quotient + range | 0.859 | 0.891 | 0.985 | 1.00 / 1.00 / 1.00 |
| **quotient + multi-scale, γ=½** | **0.945** | **0.973** | **0.988** | 0.95 / 0.91 / 0.93 |

Kept fraction, which is what the model pays for: hall_monitor 19.2 % → 5.5 %, lobby 15.3 % → 2.7 %, VIRAT 1.6 % → 1.2 %. Curves: `out/real_bench.png`; raw sweeps: `out/real_bench.json`.

## Reading

* **The multi-scale certificate is the main gain** (2.5–3.5× fewer kept patches at equal recall on noisy cameras). Noise averages away over a box, a jittering one-pixel edge dilutes by the box width, a compact object does not. The certificate it gives is weaker in exactly one way: it certifies objects that *contain a box* of the given size at the scale's contrast, not single pixels; the sweep shows 3×3 objects are still caught at 100 %.
* **The quotient certificate adds a consistent 4–5 points on top** and is the only one that attacks jitter at its cause. Its certificate is the original one, modulo a translation of at most one pixel, and it is the step that generalises to moving cameras.
* **Transport (flat norm) does not help.** A jittering step edge is single-signed mass: there is nothing for transport to cancel. The flat norm behaves like a mean rule, robust to noise but blind to 3×3 objects (recall 0.10–0.33 at its high-drop settings). It is kept as a baseline with a different, mass-based certificate.
* **Cost of the gains.** Mover coverage dips to 0.91–0.95 for the combined rule: a 6×6 object moving 2 px/frame is partly explained as sub-pixel motion and partly diluted; it is still kept on more than nine frames in ten, and the certificate says so explicitly (sub-pixel position is not certified, objects smaller than the box are not certified at the larger scales). A tracker that needs every frame should use quotient + range.
* **On a clean camera (VIRAT) every rule sits at the 98–99 % ceiling.** The alternatives matter where the original rule was weakest.

## Soundness

Theorem checks in `tests/test_rules2.py`: dual-feasible test functions never exceed the flat-norm bound; box statistics match brute force and the multi-scale certificate holds exactly (bottleneck distance of box-filtered frames ≤ ε_r) on every dropped patch of a test sequence; the quotient rule's L∞ and bottleneck certificates hold against the warped view; the C++ `QuotientPruner` matches numpy bit for bit on keep masks. 28 tests pass.

## Speed

numpy quotient rule 104 fps at 352×288; C++ port 1,040 fps at 352×288 and 106 fps at 1280×720 (single thread). Multi-scale is integral images and costs about the range rule.


---

# Part III — the time axis: sequential (quickest-detection) rule (2026-10-08, later still)

Every rule in Parts I–II is memoryless and can never see a persistent object fainter than one frame's noise. Section 6 of `THEORY.md` adds the time axis with the tools of sequential analysis: a window-limited GLR implemented as a space-time scan on the quotient residual, **self-normalised** per patch and time scale because the cameras' noise is spatially correlated and drifts in time (the parametric Gaussian version fired on 40–90 % of static patch-frames and was discarded; both versions are described in the theory note). The sequential test only adds keeps to the quotient + multi-scale rule at its Part-II operating point (Δ₀ = 80), so every dropped patch still carries that rule's worst-case certificate.

New planted events: 45 **faint persistent** objects per clip (6×6 px, contrast 8, 12 or 16, present for 3 s). Recall = kept within 1 s of appearance; delay = frames from appearance to first keep.

## Ceiling check (offline oracle)

Fraction of patch-frames with real change according to a temporal-median background oracle no online rule can use (|F − median| > 30 on ≥ 8 px, 3-frame majority): hall 0.081, lobby 0.034, VIRAT 0.014. The rules below keep 0.055–0.059, 0.027–0.029 and 0.012–0.015 respectively: at or below the oracle, because a person who pauses needs no new token. There is no drop-rate headroom left; the gains below are in what gets *seen*.

## Faint-object recall at the same token budget

All rules at the threshold giving 100 % recall of the Part-II objects (small, tiny, fade).

| rule | clip | drop | faint 8 | faint 12 | faint 16 | delay (frames) | mover coverage |
|---|---|---|---|---|---|---|---|
| range | hall | 0.808 | 0.27 | 0.33 | 0.47 | 17 / 1 / 0 | 1.00 |
| quotient + multi-scale | hall | 0.944 | 0.00 | 0.13 | 0.33 | – / 4 / 3 | 0.94 |
| **+ sequential, z′ = 8** | hall | **0.941** | **0.93** | **1.00** | **1.00** | 0 / 0 / 0 | 0.95 |
| range | lobby | 0.847 | 0.53 | 0.67 | 0.67 | 0 / 1 / 0 | 1.00 |
| quotient + multi-scale | lobby | 0.973 | 0.13 | 0.07 | 0.20 | 6 / 11 / 1 | 0.90 |
| **+ sequential, z′ = 8** | lobby | **0.971** | **1.00** | **1.00** | **1.00** | 0 / 0 / 0 | 0.90 |
| range | VIRAT | 0.984 | 0.00 | 0.07 | 0.07 | – / 2 / 1 | 1.00 |
| quotient + multi-scale | VIRAT | 0.988 | 0.00 | 0.07 | 0.07 | – / 0 / 0 | 0.90 |
| **+ sequential, z′ = 8** | VIRAT | **0.985** | **0.93** | **1.00** | **1.00** | 0 / 0 / 0 | 0.90 |

The range rule's non-zero faint recall is luck: contrast 8–16 is below its threshold of 40, and it fires when noise happens to add to the object within the 1-s horizon. The sequential rule's recall is by design, on the first frame, at a cost of 0.2–0.4 points of drop rate. Sweep (`out/real_bench_seq2.json`): z′ from 4 to 12 moves the drop rate from 0.88 to 0.943 on hall with faint-12 recall ≥ 0.93 throughout; z′ = 8 is the knee on all three cameras.

Re-run after the scale floor and global veto of THEORY.md §6 were added (needed by the breadth footage, Part VI): at z′ = 8 the drop rates are 0.943 / 0.972 / 0.987 and faint recall at contrast 8 / 12 / 16 is 0.87 / 0.93 / 1.00 (hall), 0.93 / 1.00 / 1.00 (lobby), 1.00 / 1.00 / 1.00 (VIRAT): the guards cost nothing here.

## What the null looks like on a real camera

Learned per patch on quiet frames, hallway camera, window 4 frames, 5×5 box: location ≈ 1.0 grey level, spread ≈ 0.18. At z′ = 8 a persistent object of contrast ≈ 2.5 grey levels is therefore detectable within four frames, against an effective memoryless threshold of 36 for the same box. False fires measured on static patches (oracle-static, generous dilation): 0.13 % of patch-frames at z′ = 8 (0.03 % at z′ = 10); on white Gaussian noise 0.03 %.

## Soundness and speed

Tests (`tests/test_rules2.py`): faint persistent object caught within two frames where the memoryless rule never fires; false-fire rate on white noise below 0.3 %; no re-fires once the object is in the reference; delay bound monotone; C++ `SequentialPruner` matches numpy keep-for-keep on both a synthetic clip and the planted hallway clip (0 of 118,800 patch decisions differ). numpy 59 fps and C++ 467 fps at 352×288; C++ 89 fps at 1280×720, single thread, with 2·16+1 frames of float32 prefix sums in memory.

## What this is and is not

It is the first skip rule that can certify *both* directions: worst-case, nothing of contrast Δ appeared in a dropped patch (Theorems 3–4); statistically, any persistent change above a few grey levels is kept within a provable, near-optimal delay (Propositions 5–6, with the null measured rather than assumed). It is not a higher drop rate: that is at the oracle ceiling already. And the mathematics is not new, it is sequential analysis from 1954–1998 and the a-contrario school, applied where nobody had applied it: see the honesty paragraphs in `THEORY.md` §6.

---

# Part IV — pixel-accurate object masks at video rate (2026-10-08, last)

Goal: masks of the quality of a semantic-segmentation figure (filled silhouettes, flat background, one colour per object, no class labels), produced on a video at video rate rather than frame by frame. `THEORY.md` §7 gives the mathematics: the per-frame Potts problem solved exactly through its convex TV relaxation, and the video treated as a time-varying convex program whose minimiser is tracked (prediction from object velocities, three primal–dual correction steps on the active patches, KKT residual as the certificate for reused labels).

## Fidelity of tracking versus solving

Hallway camera, same evidence, tracked mask (3 steps/frame) against the fully converged per-frame solution (150 iterations): mean IoU 0.938, median 0.957, 3 % of frames below 0.8; KKT residual on the frozen region below 0.05 on every frame of the final runs (max 0.050, by construction of the re-activation rule). C++ against numpy on the same clip: mean IoU 0.987–0.993.

## Speed (single-threaded C++, `cpp/rtseg.hpp`, Apple M4; rendering excluded)

| clip | resolution | active set | ms / frame | fps |
|---|---|---|---|---|
| hallway (Xiph) | 352×288 | 49 % | 7.1 | 140 |
| mall corridor (CAVIAR) | 384×288 | 29 % | 6.2 | 161 |
| lobby (CAVIAR) | 384×288 | 23 % | 5.6 | 178 |
| parking lot (VIRAT), full 70 s | 1280×720 | 4.6 % | 34.0 | 29 |

Per-frame full solve (numpy, 60 iterations): 15.7 fps at 352×288. numpy tracker: 38 fps at 352×288. The event-driven C++ path makes the cost scale with the active set: at 720p only one pixel in twenty is touched by the evidence, correction and certificate stages.

## Quality, honestly

Videos `out/masks_hall.mp4`, `out/masks_corridor.mp4`, `out/masks_lobby.mp4`, `out/masks_virat_720p.mp4` (three panels: frame, overlay with per-object boxes, flat mask). Silhouettes are filled and stable, shadows on flat floors are suppressed, a briefcase set down on a cabinet becomes its own object, a car crossing a parking lot is one clean blob. Identity through contact is handled by the tracker rather than by connected components: when one component overlaps two substantial predicted objects it is partitioned by nearest predicted object (a BFS from the advected labels), so two people walking shoulder to shoulder keep separate masks and colours; fragments such as a foot or a bag cannot split the body they rejoin (the 0.3 / 0.2 area rule). What classical evidence cannot do: a grey coat against a grey wall thins the silhouette at the waist (a radius-2 closing bridges it), people who enter the scene already touching share one identity until they separate, a polished pillar edge that reflects passers-by is segmented as change because it is one, far-away people a few pixels tall are at the mercy of `min_area`, and a person who stands still through the background initialisation is background. A learned segmenter would beat this on appearance; nothing in it is learned, and every mask comes with the statement that it is an η-stationary point of a convex energy whose data term is a calibrated statistical test.

## What is new here

Prediction–correction tracking of a time-varying convex program (control theory, 2016–2020) applied to make a segmentation energy follow a video; the active set driven by the certified change statistic of Part II; and a KKT certificate for the labels that are reused rather than recomputed. The ingredients (background subtraction, Potts/TV segmentation, Chambolle–Pock, morphology, overlap tracking) are all standard and are named as such in the theory note.


## Addendum — 1080p footage, crispness and the three additions (Part IV, continued)

Footage: VIRAT 1080p scenes 0400 (building forecourt, cars) and 0500 (street with pedestrians and wind in the trees), public release on Kitware, 30 fps. Additions (THEORY.md §7.1): Lagrangian evidence accumulation, combined residual-and-image boundary weights, guided-filter boundary snap; the active set now uses the studentised statistic, hole filling and the snap band are local to object boxes, and the per-patch stages are multi-threaded in a four-colour schedule.

| clip | resolution | active set | ms / frame | fps |
|---|---|---|---|---|
| building 0400 | 1920×1080 | 5.0 % | 35–42 | 24–29 |
| street 0500 (foliage) | 1920×1080 | 9.5 % | 52–55 | 18–19 |
| hallway | 352×288 | 49 % | 6.6 | 151 |

Lessons from the high-resolution footage: (1) the per-frame displacement bound inherited from CIF (8 px) silently disabled prediction and advection for cars at 1080p and produced a trailing smear; raised to 48 px. (2) The active-set test on raw grey-level differences made half the frame active under wind; the studentised version brings it to 5–10 %. (3) Closing radius 1 with a tighter guided-filter regularisation (ε = 16) is marginally crisper than the CIF defaults on 1080p; the remaining softness is in the evidence (motion blur, cast shadows on asphalt, pedestrians walking in contact), not in the optimiser. (4) Ghosts: a car stopped at the intersection during initialisation left a ghost in its old spot, and five seconds of absorption left a second one attached to the moving car; the frame-versus-model contour-contrast test removes the first and not absorbing objects removes the second (THEORY.md §7.1). (5) The C++ ghost buffer was unsized and crashed the process on first use; a one-line fix, recorded here because silent crashes in native code are the thing to look for first when a render script prints nothing. Videos: `out/masks_street_1080p.mp4`, `out/masks_building_1080p.mp4`; final speed 25 fps building, 19 fps street.

---

# Part V — predictive look scheduling with a distribution-free guarantee (2026-10-08, last)

Setup in `THEORY.md` §8; code `python/certskip/sampler.py`, benchmark `scripts/sampling_bench.py`. Native frame rate, certified per-patch rule = quotient + multi-scale at Δ₀ = 80, miss = onset of ≥ 3 new patches outside the tracked region during skipped frames, `H = 30`, `γ = 0.3/√n`. Oracle motion for staleness from a temporal-median background.

| clip | onset rate (quiet / active, per frame) | best uniform at 10 % miss | adaptive α = 0.10 (looks/s, miss, stale) | uniform at the same cost (miss, stale) |
|---|---|---|---|---|
| mall corridor | 0.17 / 0.40 | none below h = 1 | 23.2, 0.040, 0.07 | 0.044, 0.05 |
| lobby | 0.11 / 0.24 | none below h = 1 | 21.4, 0.046, 0.14 | 0.065, 0.11 |
| VIRAT parking lot | 0.0037 / 0.0076 | h ≈ 10 (3 looks/s) | 3.6, 0.027, 6.64 | 0.040, 3.82 |

Guarantee check: miss rate ≤ α + slack in 12 of 12 runs (slack 0.16–0.84 for these run lengths; it decays as 1/√N). Adversarial unit test: an opponent that plants an onset in every skipped window cannot push the miss rate above the bound, and the scheduler converges to looking every frame. Calibration test: for memoryless onsets at rate λ the learned horizon converges to α/λ.

**Verdict.** Sound, self-calibrating, and not better than a tuned uniform rate on real surveillance footage, for the structural reason in Proposition 10: onsets are nearly unpredictable from recent activity (quiet-to-active rate ratio 2–3), so adaptivity has little to exploit, and uniform spacing minimises staleness. The first version of this benchmark used "≥ 3 refreshed patches" as the miss event and collapsed to looking at every frame because continuous activity and flicker noise made every window a "miss"; the onset definition is the right one and is what the table reports. The idea stays in the repository as the measured answer to "should the model look adaptively?": on fixed cameras, no; look every frame with the certified statistic, which is what Parts I–III do.


---

# Part VI — breadth: 25 fixed-camera clips (2026-10-08, last)

Every fixed-camera clip in `data/` (`scripts/breadth_bench.py`): VIRAT public release (Kitware) scenes 0002, 0100, 0101, 0102, 0400, 0401, 0500, 0502, 0503 at 720p and 1080p (analysed at 960×544), the Edinburgh CAVIAR lobby and corridor cameras (MPEG-1), and the Xiph analog captures hall_monitor, akiyo and the two bridge cameras. 600 frames per clip at native rate. Operating points fixed in advance from Parts II–III: range Δ = 32, quotient + multi-scale Δ₀ = 80, sequential z′ = 8 with the scale floor and veto of THEORY.md §6. Planted objects per clip: 20 small, 20 tiny, 10 fades, 10 faint persistent (6×6, contrast 12).

| clip | source | resolution analysed | fps | σ (grey) | oracle change | drop: range Δ=32 | drop: quotient+MS | drop: +sequential | faint-12 recall MS / seq |
|---|---|---|---|---|---|---|---|---|---|
| VIRAT_S_000200_00.mp4 | VIRAT (Kitware) | 960x544 | 30.0 | 0.00 | 0.015 | 0.989 | 0.992 | 0.991 | 0.00 / 1.00 |
| VIRAT_S_000200_01.mp4 | VIRAT (Kitware) | 960x544 | 30.0 | 0.00 | 0.013 | 0.986 | 0.991 | 0.991 | 0.00 / 0.90 |
| VIRAT_S_040000_00b_1080p.mp4 | VIRAT (Kitware) | 960x544 | 30.0 | 0.00 | 0.014 | 0.990 | 0.998 | 0.998 | 0.00 / 1.00 |
| VIRAT_S_050000_05_1080p.mp4 | VIRAT (Kitware) | 960x544 | 30.0 | 0.00 | 0.065 | 0.950 | 0.966 | 0.965 | 0.10 / 1.00 |
| akiyo_cif.y4m | Xiph derf | 352x288 | 30.0 | 0.00 | 0.182 | 0.927 | 0.972 | 0.971 | 0.20 / 1.00 |
| caviar_EnterExitCrossingPaths1cor.mpg | CAVIAR | 384x288 | 25.0 | 0.30 | 0.104 | 0.935 | 0.961 | 0.959 | 0.10 / 0.90 |
| caviar_Fight_Chase.mpg | CAVIAR | 384x288 | 25.0 | 0.37 | 0.055 | 0.820 | 0.964 | 0.964 | 0.20 / 1.00 |
| caviar_LeftBag.mpg | CAVIAR | 384x288 | 25.0 | 0.46 | 0.038 | 0.816 | 0.971 | 0.970 | 0.20 / 0.90 |
| caviar_Meet_Crowd.mpg | CAVIAR | 384x288 | 25.0 | 0.40 | 0.029 | 0.821 | 0.973 | 0.972 | 0.00 / 0.90 |
| caviar_OneLeaveShopReenter1cor.mpg | CAVIAR | 384x288 | 25.0 | 0.37 | 0.059 | 0.956 | 0.974 | 0.971 | 0.10 / 0.90 |
| caviar_Walk1.mpg | CAVIAR | 384x288 | 25.0 | 0.45 | 0.039 | 0.828 | 0.974 | 0.974 | 0.40 / 1.00 |
| caviar_WalkByShop1cor.mpg | CAVIAR | 384x288 | 25.0 | 0.32 | 0.062 | 0.963 | 0.981 | 0.980 | 0.30 / 1.00 |
| hall_monitor_cif.y4m | Xiph derf | 352x288 | 30.0 | 1.69 | 0.081 | 0.784 | 0.949 | 0.948 | 0.10 / 1.00 |
| virat_010000_02.mp4 | VIRAT (Kitware) | 960x544 | 24.0 | 0.00 | 0.024 | 0.980 | 0.995 | 0.995 | 0.10 / 0.90 |
| virat_010106_01.mp4 | VIRAT (Kitware) | 960x544 | 24.0 | 0.00 | 0.018 | 0.990 | 0.997 | 0.996 | 0.00 / 0.90 |
| virat_010109_00.mp4 | VIRAT (Kitware) | 960x544 | 24.0 | 0.00 | 0.061 | 0.971 | 0.996 | 0.996 | 0.10 / 1.00 |
| virat_010112_00.mp4 | VIRAT (Kitware) | 960x544 | 24.0 | 0.00 | 0.067 | 0.983 | 0.996 | 0.996 | 0.10 / 1.00 |
| virat_010200_02.mp4 | VIRAT (Kitware) | 960x544 | 24.0 | 0.00 | 0.006 | 0.996 | 0.999 | 0.999 | 0.00 / 1.00 |
| virat_010204_01.mp4 | VIRAT (Kitware) | 960x544 | 24.0 | 0.00 | 0.017 | 0.988 | 0.993 | 0.993 | 0.10 / 1.00 |
| virat_040000_02.mp4 | VIRAT (Kitware) | 960x544 | 30.0 | 0.00 | 0.010 | 0.992 | 0.999 | 0.999 | 0.10 / 1.00 |
| virat_040103_08.mp4 | VIRAT (Kitware) | 960x544 | 30.0 | 0.00 | 0.013 | 0.985 | 0.995 | 0.994 | 0.00 / 1.00 |
| virat_050201_03.mp4 | VIRAT (Kitware) | 960x544 | 30.0 | 0.00 | 0.232 | 0.953 | 0.990 | 0.989 | 0.30 / 1.00 |
| virat_050300_00.mp4 | VIRAT (Kitware) | 960x544 | 30.0 | 0.00 | 0.286 | 0.887 | 0.974 | 0.973 | 0.30 / 0.90 |
| xiph_bridge_close_cif.y4m | Xiph derf | 352x288 | 30.0 | 1.07 | 0.113 | 0.756 | 0.908 | 0.908 | 0.40 / 1.00 |
| xiph_bridge_far_cif.y4m | Xiph derf | 352x288 | 30.0 | 1.25 | 0.000 | 0.886 | 0.997 | 0.997 | 0.00 / 0.90 |

25 clips. Drop rate median range / quotient+MS / +sequential: 0.956 / 0.990 / 0.989; minimum 0.756 / 0.908 / 0.908; sequential cost over quotient+MS: max 0.003, median 0.001. Recall of small / tiny / fading objects (sequential): 1.00 / 1.00 / 1.00; faint-12 recall quotient+MS 0.13 versus sequential 0.96.

**Reading.** On codec-clean cameras (σ = 0: static blocks repeat exactly) the certified rule drops 99 to 99.9 %; the lowest numbers are honest ones, a river camera whose water moves (oracle change 11 %, drop 90.8 %), two busy scenes with 23–29 % of patch-frames in real motion, and the analog hallway. The sequential rule's cost is now at most 0.3 points on any clip (it was 11 points on one scene before the codec-breathing guards) and it raises faint-object recall from 0.13 to 0.96 on average with no loss on the other object types. Not covered: night and bad weather, because the ChangeDetection 2014 server was unreachable during this session; those are the next clips to add, and the heavy-tailed analog cameras here (hall, bridges) are the closest proxy for what they will do to the range rule.

---

# Part VII — end to end with a real video LLM (2026-10-08, last)

Model: Qwen2-VL-2B-Instruct, unmodified, on Apple's MPS backend (`.venv`, Python 3.12; 2.3 s per query). Protocol (`scripts/e2e_vlm.py`): the model is shown the **view** a pruning rule leaves it, kept patches at truth, dropped patches at their last kept content. This isolates the information content of the pruning decisions from token-reuse mechanics and needs no change to the model; the token savings are the drop rates of Parts I–VI. Three rules at the *same* drop rate per clip: the consecutive-frame mean rule of EVS and run-length tokenisation, calibrated to the certified rule's drop rate; the certified quotient + multi-scale rule; and the certified rule with the sequential layer.

## E2: does certified pruning change the model's answers on real footage?

Fifteen frames per clip, "How many people are visible?" and "Is there a person visible?", agreement with the model's own answer on the full frame.

| clip | drop rate (all rules) | EVS-style view: count agreement / MAD | certified view | + sequential |
|---|---|---|---|---|
| hallway | 0.949 | 1.00 / 0.00 | 1.00 / 0.00 | 1.00 / 0.00 |
| lobby | 0.979 | **0.47 / 0.53** | **0.93 / 0.07** | 0.87 / 0.13 |
| VIRAT parking lot | 0.996 | 0.80 / 0.27 | 0.80 / 0.27 | 0.80 / 0.27 |

Person-presence agreement was 0.93–1.00 for every rule. On the lobby the heuristic view changes the model's people count on half the frames at a 97.9 % drop rate; the certified view, whose every dropped patch is within the contrast certificate of the truth, changes it on one frame in fifteen. On VIRAT the three views agree with each other on every frame and the 0.80 is the model's own instability on far-away people, identical across views.

## E1: the fade-in blind spot, end to end

A square of contrast 70 fades in over 6–8 s (0.4–0.9 grey levels per frame, far below any consecutive-frame threshold) on the hallway, lobby and parking-lot cameras; twelve trials per clip in the first run.

*Mechanism, pixel level.* Fraction of trials in which the object is present in the view at the end of the fade: EVS-style 0.25 / 0.00 / 0.00; certified 1.00 / 0.83 / 1.00; sequential 1.00 / 0.83 / 1.00 (hallway / lobby / parking lot). The heuristic never refreshes a patch that changes by less than its threshold per frame, however large the change becomes; the certified rule refreshes when the accumulated change reaches Δ.

*Model level.* The small model perceives a synthetic square unreliably even on the truth frame (2 / 6 / 2 of 12 trials perceived with the first protocol, which used bright squares on bright surfaces half the time). Conditional on the model perceiving the object in the truth frame and not in the control: EVS-style view 0 / 10 trials answered yes, certified and sequential views 5 / 10. A second run with the square contrasting against its local surroundings and 48 px (CIF) / 96 px (HD) is reported below.

*Second run, corrected protocol* (square contrasting against its local surroundings, 48 px at CIF and 96 px on the parking lot, 16 trials per clip, 48 in all; `out/e2e_e1b.json`):

| | hallway | lobby | parking lot | all 48 trials |
|---|---|---|---|---|
| object present in view at end of fade, EVS-style / certified / sequential | 0.12 / 1.00 / 1.00 | 0.25 / 1.00 / 1.00 | 0.00 / 1.00 / 1.00 | 6 / 48 / 48 |
| trials in which the model perceives the object in the truth frame | 8 | 12 | 6 | 26 |
| of those, model answers yes on the EVS-style view | 0.12 | 0.08 | 0.00 | **2 / 26** |
| of those, model answers yes on the certified view | 0.75 | 0.92 | 1.00 | **23 / 26** |
| of those, model answers yes on the sequential view | 1.00 | 0.92 | 0.83 | **24 / 26** |
| false "yes" on the no-object control | 0.00 | 0.00 | 0.00 | 0 / 48 |

**Verdict of the end-to-end experiment.** At identical token budgets, an object that fades in slowly reaches the model's answer through the certified rule in 23 of 26 perceivable cases and through the EVS-style rule in 2 of 26; and on ordinary footage the certified view leaves the model's answers as stable as the full frame (hallway, parking lot) or far more stable than the heuristic view (lobby, 0.93 versus 0.47 count agreement). Limits to state: a 2-billion-parameter model on a laptop, three cameras, synthetic squares rather than real objects (the model perceives them in only 26 of 48 truth frames, which is why the metric is conditioned), and views instead of actual token reuse inside the model. The mechanism those limits cannot touch, the pixel-level presence of the object in the view, is 48 / 48 against 6 / 48.

**Video of E1.** `scripts/render_e2e.py` writes `out/e2e_blindspot.mp4` (46 s): three of the 48 trials
(hall 0, lobby 0, VIRAT 4) replayed with the exact geometry of the finished run, as truth / EVS-style view /
certified view / certified + sequential view at the matched drop rate, kept patches in green, the planted
square outlined in yellow, and the recorded model answers stamped on a hold card at the end of each fade.
Three-panel versions of the lobby and parking-lot trials are tracked as `assets/blindspot_lobby.mp4` and
`assets/blindspot_parking.mp4` (`scripts/make_readme_media.py`).

---

# Part IX — frame rate: a threshold on speed against one on displacement (2026-10-10)

Run on a Linux laptop (8 threads, GTX 1650 Ti), not on the M4 of Parts I–VII. Theory in `THEORY.md` §10; script `scripts/framerate_bench.py`; outputs `out/framerate.{json,txt,png}` and `out/framerate_summary.png`.

**Question.** A consecutive-frame rule compares each frame with the one before it. Proposition 12 says its threshold is a threshold on speed, which rises with the frame rate, so the same rule should see less of the same scene when the scene is read faster. Does it, on real footage with nothing planted?

**Protocol.** The eleven fixed-camera clips of CIF size (hall_monitor, akiyo, the two Xiph bridge cameras, seven CAVIAR clips), each read at its native rate and subsampled by 2, 3, 6, 10, 15 and 30. Only real frames are used. At every rate two rules run at the same skip rate: the certified rule of §9 (quotient + multi-scale, no offset, Δ₀ = 80) and the consecutive-frame mean rule of Part I, re-tuned at that rate to the certified rule's skip rate ("re-tuned") or tuned once at about 2 fps and left alone ("fixed"). After every frame the copy the model holds is compared with the truth: *object pixels wrong* is the share of moving-object pixels (more than 30 grey levels from the clip's temporal median, an oracle) that are off by more than 30 grey levels; *ghost pixels* is the same share over the rest of the frame, which is what a rule leaves behind where nothing is any more.

## Median over the clips

| step from native | about fps | clips | ghost %, certified | heuristic, re-tuned | heuristic, fixed | object pixels wrong %, certified | heuristic, re-tuned | heuristic, fixed | skip rate %, certified / re-tuned / fixed |
|---|---|---|---|---|---|---|---|---|---|
| 1/30 | 1 | 9 | 0.08 | 0.12 | 0.15 | 1.0 | 1.5 | 1.6 | 91.0 / 91.0 / 91.4 |
| 1/15 | 2 | 11 | 0.07 | 0.17 | 0.17 | 1.0 | 2.3 | 2.3 | 91.5 / 91.5 / 91.5 |
| 1/10 | 3 | 11 | 0.08 | 0.20 | 0.24 | 1.1 | 2.0 | 2.8 | 92.5 / 92.4 / 92.4 |
| 1/6 | 5 | 11 | 0.08 | 0.29 | 0.28 | 1.2 | 2.9 | 4.2 | 93.5 / 93.5 / 93.9 |
| 1/3 | 10 | 11 | 0.08 | 0.44 | 0.45 | 1.7 | 5.2 | 6.6 | 94.4 / 94.4 / 95.0 |
| 1/2 | 15 | 11 | 0.09 | 0.54 | 0.64 | 2.0 | 7.5 | 8.1 | 94.8 / 94.8 / 96.1 |
| 1/1 | 25 to 30 | 11 | 0.09 | 1.07 | 1.17 | 2.5 | 9.9 | 11.5 | 96.6 / 96.6 / 96.9 |

## Per clip, at about 2 fps and at the native rate (heuristic re-tuned)

| clip | native fps | ghost %, about 2 fps: certified / heuristic | ghost %, native: certified / heuristic | object pixels wrong %, native: certified / heuristic |
|---|---|---|---|---|
| akiyo_cif | 30 | 0.75 / 1.21 | 0.70 / 3.55 | 10.0 / 23.7 |
| caviar_EnterExitCrossingPaths1cor | 25 | 0.01 / 0.07 | 0.07 / 0.80 | 2.5 / 5.8 |
| caviar_Fight_Chase | 25 | 0.10 / 0.17 | 0.09 / 0.83 | 2.0 / 9.9 |
| caviar_LeftBag | 25 | 0.10 / 0.50 | 0.12 / 2.18 | 3.0 / 13.8 |
| caviar_Meet_Crowd | 25 | 0.09 / 0.21 | 0.10 / 1.07 | 2.7 / 7.4 |
| caviar_OneLeaveShopReenter1cor | 25 | 0.01 / 0.05 | 0.03 / 0.54 | 2.5 / 9.2 |
| caviar_Walk1 | 25 | 0.08 / 0.24 | 0.09 / 0.86 | 2.3 / 12.4 |
| caviar_WalkByShop1cor | 25 | 0.03 / 0.15 | 0.08 / 2.21 | 2.1 / 8.0 |
| hall_monitor_cif | 30 | 0.07 / 0.07 | 0.11 / 1.98 | 1.6 / 3.7 |
| xiph_bridge_close_cif | 30 | 0.07 / 0.44 | 0.09 / 1.19 | 2.8 / 16.5 |
| xiph_bridge_far_cif | 30 | 0.07 / 0.14 | 0.07 / 0.24 | 9.3 / 48.0 |

## Reading

* **The heuristic leaves more behind the faster it looks, the certified rule does not.** Median ghost pixels of the consecutive-frame rule go from 0.12 % at 1 fps to 1.07 % at the native rate, a factor nine, at equal skip rates throughout. The certified rule stays between 0.07 and 0.09 %. The heuristic's ghosts are higher at the native rate than at 2 fps on all eleven clips, and higher than the certified rule's at the native rate on all eleven.
* **At the rates where video models are usually fed, the two are close.** At 1 fps the heuristic leaves 1.5 times the certified rule's ghosts, at 2 fps 2.3 times, at the native rate 12 times. A consecutive-frame rule looks adequate at 1–2 fps because at 1–2 fps it nearly is.
* **Moving objects.** The share of moving-object pixels that are wrong goes from 2.3 % to 9.9 % for the heuristic and from 1.0 % to 2.5 % for the certified rule. The certified number rises too: at a high frame rate more frames fall between two sends, and each of them is off by something below the certificate's bound.
* **Re-tuning the threshold at every rate does not help.** It keeps the skip rate equal and leaves the picture about as wrong as a fixed threshold does. The failure is in what is compared, not in the threshold.

## What this is and is not

It is the blind spot of Part I, measured without planting anything and stated as a law in the frame rate, with the two-line proof of Proposition 12 behind it. It is a measurement on pixels: the model was not asked. The oracle is a temporal median, not hand-labelled ground truth, and the heuristic is the consecutive-frame criterion re-implemented here, not any published system's full pipeline. The clips are the eleven CIF-size ones; nothing faster than 30 fps was available, so the law is tested downwards from the native rate only.

---

# Part X — what to hold: the fewest sends a certificate allows (2026-10-10)

Linux laptop, CPU only. Theory in `THEORY.md` §11; code `python/certskip/centre.py`; tests `tests/test_centre.py`; script `scripts/centre_bench.py`; outputs `out/centre.{json,txt,png}`.

**Question.** Every rule so far holds a frame: when the certificate breaks it sends the current frame. Theorem 13 gives the fewest sends any policy can make for a certificate, whatever it holds and with any latency; Theorem 14 places a frame-holding rule between that and the same bound at half the tolerance. How far from the bound is the rule in use on real cameras, and how much of the distance does a little latency recover?

**Protocol.** The eleven CIF-size fixed-camera clips, 600 frames each (fewer where the clip is shorter), patch 16. Certificate: the multi-scale norm of the benchmarks without offset (Δ₀ = 80, γ = ½) and without the sub-pixel shift, the same at every latency. *Bound*: the greedy cut into runs of diameter below 2 (`fewest_sends`). *L = 0*: send-on-delta, the current frame is held. *L > 0*: on a break, the centre (pixelwise midrange) of the longest run among the next L frames that the centre certifies is held; the model's picture lags L frames. *In use*: the C++ rule of §9 at the same schedule, which also fits a sub-pixel shift per patch and so certifies less (position to within a pixel is not certified). Numbers are the share of patch-frames sent after the first frame.

## Sends, %, multi-scale certificate

| clip | bound | L = 0 | L = 1 | L = 2 | L = 4 | L = 8 | L = 15 | in use (with sub-pixel shift) |
|---|---|---|---|---|---|---|---|---|
| akiyo_cif | 2.88 | 6.02 | 4.66 | 4.14 | 3.68 | 3.37 | 3.16 | 3.14 |
| caviar_EnterExitCrossingPaths1cor | 3.31 | 5.58 | 4.46 | 4.06 | 3.78 | 3.61 | 3.53 | 4.31 |
| caviar_Fight_Chase | 2.82 | 6.16 | 4.19 | 3.77 | 3.46 | 3.23 | 3.07 | 4.27 |
| caviar_LeftBag | 2.31 | 5.59 | 3.64 | 3.23 | 2.93 | 2.71 | 2.55 | 3.50 |
| caviar_Meet_Crowd | 2.15 | 5.30 | 3.54 | 3.11 | 2.81 | 2.60 | 2.44 | 3.38 |
| caviar_OneLeaveShopReenter1cor | 2.18 | 3.61 | 2.91 | 2.70 | 2.54 | 2.41 | 2.36 | 2.88 |
| caviar_Walk1 | 2.01 | 4.83 | 3.24 | 2.87 | 2.61 | 2.38 | 2.23 | 3.13 |
| caviar_WalkByShop1cor | 1.53 | 2.66 | 2.17 | 1.96 | 1.79 | 1.68 | 1.63 | 2.10 |
| hall_monitor_cif | 4.56 | 8.02 | 5.78 | 5.36 | 5.09 | 4.92 | 4.81 | 5.64 |
| xiph_bridge_close_cif | 4.13 | 12.54 | 8.58 | 6.95 | 5.60 | 4.83 | 4.50 | 10.95 |
| xiph_bridge_far_cif | 0.03 | 0.85 | 0.67 | 0.58 | 0.44 | 0.32 | 0.21 | 0.57 |

| over the 11 clips | L = 1 | L = 2 | L = 4 | L = 8 | L = 15 |
|---|---|---|---|---|---|
| sends against L = 0, median (range) | −28 % (−18 to −35) | −33 % (−25 to −45) | −44 % (−30 to −55) | −48 % (−33 to −62) | −50 % (−35 to −76) |
| above the bound, median | +57 % | +40 % | +27 % | +17 % | +9 % |
| against the rule in use, median (range) | +3 % (−22 to +48) | −7 % (−37 to +32) | −16 % (−49 to +17) | −22 % (−56 to +7) | −27 % (−64 to +1) |
| clips on which it sends less than the rule in use | 2 of 11 | 9 of 11 | 10 of 11 | 10 of 11 | 10 of 11 |

Holding a frame (L = 0) sends a median of 2.2 times the bound, from 1.65 times to 32 times (the far bridge camera, where almost nothing happens and nearly every send is one that a centred copy would have avoided). The sup-norm certificate with ε = 40 gives the same picture to within a point or two; both are in `out/centre.txt`. The largest certificate norm met on any frame of any run was 0.9996, below 1 as it must be.

## Reading

* **A rule that holds frames sends about twice the bound, and the bound is real.** Theorem 13 is a statement about every policy with this certificate on this patch grid, learned or not. Part III's "there is no drop-rate headroom left" compared the rules with an oracle of what moved; against what a certificate actually requires there is a factor of two, and Theorem 14 says where it comes from: a held frame sits wherever the scene happened to be when the certificate broke, and tolerates half of what a centred copy would.
* **A little latency recovers most of it.** One frame of look-ahead removes a quarter of the sends, four frames 44 %, eight frames 48 %, and at fifteen frames the rule is within 9 % of the bound in the median. On the river camera, where the water never stops, sends fall from 12.5 % to 4.8 % at eight frames.
* **Against the rule in use.** The sub-pixel shift of §3 was introduced to absorb edge jitter, and it removes a median 31 % of the sends of L = 0 by certifying less. Holding a centre absorbs any oscillation smaller than the tolerance, jitter included, and certifies more: from four frames of latency on it sends less than the rule in use on 10 of the 11 clips, a median of 16 % less at four frames and 22 % less at eight. The exception is akiyo, a head-and-shoulders clip whose motion really is a sub-pixel sway. The two are not exclusive; a rule that holds centres and fits a shift is not built.
* **What it costs.** The model's picture lags by the latency, and what it is given is the midrange of a few consecutive frames, which was never a frame. The certificate is unchanged; whether the model's answers are is the experiment below.

## What did not work

The first version had no latency: on a break it held the centre of the run *so far*. On five cameras it sent 7 to 20 % less than holding a frame on two (hallway, lobby) and 10 to 30 % *more* on three (corridor, river, news anchor). When something is moving through a patch, the centre of the past is behind the present, and it breaks sooner than the present frame would. The remark after Theorem 16 is the general fact: without latency no policy is within a constant of the bound. Adding the run so far to the look-ahead window did not help from one frame of latency on, so the rule looks ahead only.

## What is and is not new

The scalar version, a time series approximated by the midrange of greedy segments under a sup-norm bound, is PMC-MR (Lazaridis and Mehrotra, 2003), and send-on-delta is the Lebesgue sampling of event-triggered control. New here: the lower bound asked of a skip certificate and measured on cameras, Theorem 14, the bounded-latency policy and Theorem 16, and the observation that the norm in which the stability theorem is stated is the one norm in which centres give the full factor two. Only the numpy reference exists; there is no C++ port, the look-ahead costs L frames of memory, and nothing here has been combined with the sequential test.
