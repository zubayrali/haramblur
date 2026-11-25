## BLUF

DP6 is a defensible *stopgap* and the wrong *strategy*. It patches the fitness-content miss by stacking three independently-unreliable detectors (modesty-label → face → gender → geometry) into a multiplicative chain, which fixes the demo images that have a clean frontal female face and silently re-breaks on the ones that don't — which is most real fitness content. Worse, the thing you're using as a precision gate (a female face is present) has almost zero discriminative power against your most important negative class (ordinary photos of clothed women). And the entire safety story rests on a **17-image dev split**, which cannot estimate precision to better than roughly ±20%. Ship a trimmed version if you must, but the real moves are elsewhere.

I did not read the codebase for this — the brief is self-contained and the critique is about the plan's structure, not its implementation. Say the word if you want me to ground any specific claim against `policy`/anchoring/seg code.

---

## 1. Critique of DP6 — where it breaks

**The anchoring chain is a recall cliff, not a precision gate.**
Composite recall = recall(modesty label) × recall(face) × P(gender=female) × P(geometry matches). Every factor is < 1 and they multiply. The motivating case — a woman in a sports bra shot torso-up, from the back, or with the face cropped/occluded (standard YouTube thumbnail framing) — frequently has **no usable frontal face at all**. So the exact content DP6 exists to fix will still fail whenever the face isn't detected or isn't classified female. You're gating your highest-value case on the one signal that case most often lacks.

**The gender classifier becomes a silent single point of failure.**
Face gender/age classifiers are weakest on: tiny/low-res crops (thumbnail faces are small), partial faces, hats/glasses, children, androgynous presentation, and they carry documented accuracy bias across skin tones. A female misclassified as male now triggers the *existing male-context suppression* → full silent discard. The failure is asymmetric and invisible. You're feeding this classifier the worst possible inputs (small thumbnail face crops) and making it load-bearing for the whole pipeline.

**The anchor doesn't discriminate the negative you actually fear.**
You suppressed `FEMALE_BREAST_COVERED` because negatives fire it up to 0.81 — i.e., the label is *miscalibrated at every threshold on your data*. Anchoring doesn't recalibrate it; it masks it with a correlated gate. But think about the Bayes: `P(immodest | label, female-face-present)` is barely higher than `P(immodest | label)`, because a female face is present in **nearly every photo of a woman, clothed or not**. The anchor only tells you "a woman is here," which is true for portraits, selfies, and every fully-dressed-woman negative. The discriminator you actually want is `P(immodest | label, exposed-skin geometry)` — that's *coverage*, not *face presence*. So anchoring specifically fails to suppress FPs on your most common negative class: clothed women with visible skin (necklines, arms, tight clothing), now *un-suppressed* because the face satisfies the anchor. Plan item 5 ("fully-dressed stay face-only") is an admission this risk is real and unmanaged.

**Geometry is a forever-patch.** "Below a face in torso geometry" assumes upright, frontal, single-scale, one-person-per-face. It breaks on: reclining/yoga/beach poses, bending, inverted gymnastics, seated close-ups, Dutch-angle thumbnails, and every **collage** (face inset in a corner, body elsewhere; face A above torso B). YouTube thumbnails are collages. Multi-person turns anchoring into an assignment problem (which of N faces owns which of M modesty boxes?) that "nearest/below" approximates badly — and here the new female-anchoring and the old male-suppression issue *conflicting verdicts* on the same torso, with resolution order deciding the outcome. That's a 3am heuristic that never stops needing exceptions.

**Whole-person silhouette blur raises FP cost exactly when you've raised FP rate.** A false positive used to be a stray box; now it's an entire censored human — far more jarring. And binary seg masks are temporally unstable: at 7.5fps the boundary shimmers frame-to-frame ("boiling" silhouette). Combined with the 500ms person-pass rate cap → pop-in/pop-out of whole-body blur. This is a UX regression on video even when detection is *correct*.

**Where DP6 is genuinely right (keep this):** anatomical context does help gender-ambiguous parts; reusing the existing seg model is cheap (~3-4ms); and inverse-anchoring is a natural symmetric twin of the male-suppression you already trust. As an *incremental* fix it's cheap and reasonable. The error is treating it as the architecture.

**One free simplification:** `BELLY_EXPOSED` fires at **0.80-0.83** on your target images — that label is *working well*. It was suppressed by a modesty gate, which is simply a policy bug. You don't need the anchoring machine for it; just raise its trust. Anchoring is an over-engineered answer to the `BREAST_COVERED` miscalibration problem being misapplied to a label that doesn't have that problem.

---

## 2. A fundamentally better architecture (off-the-shelf only)

Stop asking "which naughty label fired?" and start measuring **coverage**, which is what modesty (awrah — the area that must be covered) actually *is*.

**Per-person skin/clothing-coverage segmentation.** Use an off-the-shelf human-parsing or clothing-segmentation model (SCHP / LIP-trained parsers, DensePose, or a U²-Net cloth-seg; Segformer on ModaNet/DeepFashion for clothing) to compute `exposed_skin_area / body_area` per person instance. This is:
- **gender-agnostic** (no gender classifier on the critical path),
- **pose-agnostic** (no "below the face" geometry),
- **continuous** (a sports-bra torso has a measurable exposed ratio; degrades gracefully instead of cliff-failing),
- and it models the product's actual definition instead of proxying it through photo-trained part labels.

You already run person-instance seg — this is an incremental head/model swap, not a new pipeline. Coverage ratio becomes the modesty signal; the NudeNet labels become a secondary vote, not the gate.

**CLIP/SigLIP as the arbiter in the uncertain band.** A small image encoder scoring the *element* against a few prompts ("person in revealing/athletic wear" vs "fully clothed person") gives a holistic, semantic second opinion that is robust to exactly the sports-bra/leggings case because it isn't part-based. Run it only on elements in the uncertain confidence band. Runs fine in ORT WASM/WebGPU. This is a smarter version of upstream's NSFWJS ensemble vote.

**The honest transformative move: a fine-tune.** Both competitors win because of *trained weights*. Porda-AI's custom two-class YOLO at 320² beats general NudeNet because it learned *your exact distribution*. "No training infrastructure" is over-stated: a LoRA/head fine-tune of YOLOv8 on a few thousand labeled fitness/modesty/clothed-negative images is **one rented GPU for a weekend**, not an MLOps program. The coverage-seg approach is the no-train bridge; a fine-tuned detector is the only thing that structurally beats Porda-AI. Don't take it off the table — it's arguably the whole ballgame for the detection-quality axis.

---

## 3. How to surpass both competitors (not reach parity)

Both competitors do stateless frame-level blur. The wins are in *time* and *guarantee*, not in catching one more label.

1. **Temporal tracking (biggest perceived-quality win).** Add a lightweight IoU/centroid tracker so a blurred region persists through frames where detection momentarily drops (occlusion, motion blur, a skipped pass) and stops flickering. "It never flashes the frame it missed" is immediately, viscerally better than both competitors, who leak single frames.

2. **Zero-leak-by-default / blur-first-on-uncertainty.** For this audience a *single leaked frame is total product failure*. Invert the default: a new or changing element is **blurred optimistically** for the first N ms until classified clear, rather than "clear until proven." Combined with tracking, you can make a defensible **"no content leaks during detection latency"** claim. Neither competitor can say this. This is the category-defining feature.

3. **Coverage score → content types both miss.** Continuous coverage handles partial/athletic wear, beachwear-vs-lingerie context, and (with an illustration-capable path, since NudeNet is photo-trained) **drawings/anime** — a large uncovered category for a browser extension.

4. **Intentional-reveal UX.** Tap-and-hold (sustained intent, not a click) to reveal per-element, with an optional **reveal counter / self-accountability log** — for this user base the friction *is* the feature, and some users explicitly want reveals logged. Calm escalation, never a flashing censor box.

5. **Per-element temporal hysteresis you can tune** so the blur doesn't oscillate on borderline content — measurable as a flicker metric (below).

---

## 4. What you're not measuring (and must)

- **Leak-frame rate on video** — frames where risky content was visible before blur painted. This is the entire product value and it's unmeasured. Everything else is secondary.
- **False-blur rate on a real negative corpus** — hundreds of ordinary pages and clothed-people photos, not 17. Your "precision 100%" is `0 FPs on a handful of negatives` — statistically indistinguishable from 70%.
- **Recall on the actual target distribution** (fitness thumbnails). You admit it's not in the dev split — which is *why the gap shipped undiscovered*. If it's the bread-and-butter case it must be the bread-and-butter eval slice.
- **Per-stage failure attribution.** When a fitness image fails, was it detector / face / gender / geometry / policy? Without per-stage logging you're debugging a 5-stage chain blind.
- **Gender-classifier accuracy in isolation**, including bias across skin tones and small/partial faces — before you make it load-bearing.
- **Temporal flicker rate** — blur on/off transitions per element per second.
- **Latency-to-blur (ms from element-appears to blur-painted)** — this *is* the leak window.
- **Per-class precision/recall curves on a real-sized labeled set** — which would have shown `BREAST_COVERED` is miscalibrated and that anchoring can't fix calibration.

---

## The 3 highest-leverage moves

1. **Build a real, stratified eval set (hundreds–low-thousands: fitness, clothed-women negatives, collages, drawings, video clips) and measure leak-frame rate + false-blur rate.** The 17-image split is *why the fitness gap survived* and *why DP6's precision claims are unfalsifiable*. This is cheap, and nothing else you decide is trustworthy until it exists. Do this first.

2. **Replace label-gating with per-person coverage segmentation as the modesty signal.** Gender-agnostic, pose-robust, continuous, degrades gracefully, models awrah directly, and reuses your seg infra. Fixes fitness content *without* the anchoring house of cards. Keep NudeNet labels and (optionally) CLIP as secondary votes, not gates.

3. **Add temporal tracking + blur-first-on-uncertainty for video.** This is the move that *surpasses* rather than matches both competitors — the "never leaks a frame" guarantee — and it simultaneously smooths the shimmer that whole-person masks would introduce.

Honorable mention that may actually be #1 for the *capability ceiling*: **a weekend LoRA fine-tune of the detector on your target distribution.** It's one rented GPU, not training infrastructure, and it's the only thing that structurally beats Porda-AI's trained weights. Don't rule it out on a framing error.

**On DP6 itself:** if you ship it, ship the trimmed version — un-suppress `BELLY_EXPOSED` directly (it's already firing at 0.80+, it never needed anchoring), and route the genuinely miscalibrated `BREAST_COVERED` through coverage-seg or CLIP arbitration *instead of* face-anchoring. Anchoring as the primary gate is the mistake; coverage as the gate is the fix.