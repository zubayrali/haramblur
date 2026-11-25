Read the policy, region, person-association and host gating code to ground this. Here's the critique.

---

# 1. What DP6 gets right, and the one thing that kills it

**Right instinct:** your negatives fire modesty labels to 0.81 and your positives sit at 0.67–0.76. Those distributions *overlap*. No threshold separates them — so adding an orthogonal feature instead of moving a threshold is the correct move. DP4's curve work (`src/shared/policy.ts:97-124`) has reached the end of what score-only policy can do, and the plan correctly notices.

**The thing that kills it:** female-face anchoring is nearly uncorrelated with your actual false-positive population.

Your dangerous FPs are skin textures *on images that contain women*. A fully-dressed woman in a tan coat, a close-up portrait with a bare neck, a woman in a headscarf — these fire `FEMALE_BREAST_COVERED` at 0.7 **and have a confident `FACE_FEMALE` directly above the firing box in textbook torso geometry**. The anchor passes. You've re-enabled the labels and added 200 lines of geometry that filters out only the FPs that had no person in them at all — the easy ones you weren't being hurt by.

**Measure this before writing any DP6 code:** of the modesty-class firings at ≥0.6 on your dev-split negatives, what fraction are female-face-anchored in torso geometry? You have `scripts/probe-dets.ts` and the fixtures. If that number is above ~50%, DP6 step 1 is dead and you should know it in an afternoon, not after a PR.

My prior: it's above 70%, because `woman-headscarf.jpg`, `portrait-woman.jpg`, `woman-elderly-gambian.jpg`, `painting-woman-ingres.jpg` all have exactly that geometry.

---

# 2. Where else DP6 breaks

### The anchor is absent in precisely your target case
DP6 gates fitness content on a confident `FACE_FEMALE`. Fitness thumbnails are the content type *most* likely to lack one:

- **Headless torso shots.** A large share of fitness/gym thumbnails are cropped at the neck. No face → no anchor → no blur. A bare midriff with the head cropped out is arguably the most immodest framing available, and DP6 blurs it the *least*.
- Back-to-camera (leggings shots — your stated failing case), face occluded by a barbell/dumbbell/arm, sunglasses, extreme yaw, motion blur, thumbnail text overlay sitting on the face, face below the detector size floor in a grid thumbnail.

You've made your flagship case depend on the signal least available in it.

### Gender-classifier error is a cliff, and correlated
`FACE_FEMALE` confidence ≥ threshold becomes a binary gate. Gender accuracy on small/angled/occluded faces is realistically 80–90%. Worse: those error conditions are *the same conditions* as the anchor-absence cases above — the failures aren't independent, they compound multiplicatively. And the failure is **silent**: male-context suppression failing produces a visible false blur the user complains about; female anchoring failing produces nothing, and nobody files a bug against a mechanism they can't see.

### The geometry constants will overfit to six thumbnails
Look at what you already shipped (`src/shared/policy.ts:56-65`):

```
inX   = cx >= mx − mw*0.35 && cx <= mx + mw*1.35
torsoY = dy >= my + mh*0.6 && dy <= my + mh*2.6
```

Four magic constants assuming upright, frontal, ~1:1 face aspect, standard head-to-torso ratio. The inverse rule needs the same windows and breaks in the same places: seated, reclining (yoga mat, beach — torso is *lateral*, not below), bent-over (gym), overhead camera, face near the bottom crop edge, heavily zoomed faces where `mh` is large enough that the torso window spans past the frame. You will tune these against `tests/fixtures/yt-*.jpg` — six images — and ship a rule that generalizes to six images.

### Multi-person is worse under anchoring, not better
- Side-by-side man + woman: a belly box between them anchors to whichever face wins a test with ±35% slop. Coin flip.
- **2×2 thumbnail grids and before/after collages:** person A's face in the top cell sits directly above person B's torso in the bottom cell, in perfect "torso geometry." You anchor A's gender to B's body. Fine when both are women; catastrophic when you then blur "the whole person silhouette" for the wrong person.

### Two existing escalation rules nullify DP6 steps 2–3 on your target content
This is the concrete one. From `src/content/region.ts:200-204`:

```
if (femaleFaceCount(...) >= SMART_CENSOR_MAX_FEMALE_FACES) return whole;   // ≥2 → whole-element
if (regionsCoveredArea(regions) > settings.escalateAreaPct) return whole;    // >30% → whole-element
```

- **Group fitness thumbnails have 2+ women** → whole-element blur, silhouettes never painted. A large slice of your target content skips DP6 steps 2–3 entirely.
- **Single standing person in a 9:16 thumbnail:** instance box covers ~70% of the frame, `mask.coverage` ~0.5 → `regionsCoveredArea` ≈ 0.35 > 0.30 → whole-element. So the silhouette work you're adding gets thrown away in the single-subject case too.

DP6 step 3 is largely unreachable without reworking escalation first, and the plan doesn't mention it.

### Silhouette masks degrade badly under temporal lag
Person pass at 500ms on GPU (`src/engine/host.ts:44-45`), capture at ~7.5fps → a fresh mask every ~4 frames, carry-forward in between. Box regions degrade *gracefully* under lag: they're convex and padded, so a moving body stays inside. Silhouette masks degrade *terribly* — the sliver that slides out from under the mask is exactly a limb or torso edge. You'll get strobing partial reveal on any motion. Boxes are the safer failure mode and DP6 treats masks as the upgrade.

### Mask paint cost is main-thread and uncached on video
`maskDataUrl` (`src/content/region.ts:711-729`) does canvas + `toDataURL` per mask, cached on the raw bits string. On video, masks are never bit-identical frame to frame — **100% cache miss, every frame, synchronous, in the content script**, plus cache thrash at 100 entries. Eight people in a group photo = eight rasterize + eight `toDataURL` + eight masked compositing layers. Your 3–4ms inference number is measuring the cheap half. Nobody has measured the expensive half.

### DP6 step 5 is a UX regression
"Whole-person at max strictness" makes the strictness slider change the *shape* of the output, not just the threshold. You already have `coverageMode` as a separate axis for exactly this. Conflating them means a user nudging strictness sees a discontinuous behavior change and reads it as a bug.

---

# 3. The better architecture: invert the pipeline

**Stop deriving blur regions from immodesty labels. Derive them from person instances. Use immodesty labels only to escalate, never to enable.**

Per person instance from YOLOv8n-seg:
1. Associate faces → gender (you already have `associateFacesToPersons`, `src/engine/person.ts:374`, containment-primary — far more robust than hand-tuned torso windows).
2. Score modesty on the instance.
3. Blur the instance.

NudeNet becomes explicit-content detection and escalation. It stops being the region source.

Why this is strictly better:
- **Person-class segmentation is one of the best-trained object classes in existence.** Its precision on "is there a person" vastly exceeds NudeNet's on "is this a covered breast." You're currently routing your most important decision through your weakest model.
- **Solves the headless torso** — an instance exists with no face at all.
- **Instance masks *are* the anchoring.** You built the right primitive and DP6 is about to bypass it with geometry constants.
- **Gives you a per-person state object** — the prerequisite for tracking, per-person reveal, and "unblur just her."

### The modesty signal: self-calibrated skin fraction inside the mask

You have a binary silhouette and the frame bitmap. Compute the fraction of pixels inside the mask whose chroma is within ΔE of **the median chroma of that person's own associated face patch**.

A woman in a sports bra: ~35–50% skin inside her silhouette. The same woman in a winter coat: ~8%. One scalar, a few hundred microseconds on a downscaled crop, no model, no training, monotone and explainable, and it drops straight onto your existing strictness curve as a continuous feature.

The face-patch calibration is the part that matters: it makes the signal **self-calibrating per skin tone**, which a fixed HSV skin range is not. Given that `woman-elderly-gambian` is your single hardest dev-split miss, a tone-invariant modesty signal isn't a nice-to-have — a filter that works worse on darker-skinned women is an ethical failure, not just a metric.

Honest FP sources: beige/tan clothing, sheer stockings, grayscale images (detect and skip), blown-out lighting, extensive tattoos. All of which argue for blending it with detector evidence rather than using it alone — which is what you want anyway.

### Off-the-shelf models worth your attention

**Clothes/human parsing (the most on-target model nobody in this space uses).** SegFormer fine-tunes on ATR/LIP give per-pixel labels for upper-clothes, pants, skirt, dress, arms, legs, torso-skin — `mattmdjaga/segformer_b2_clothes` is MIT. That is *literally the modesty question*: which body regions are skin-labelled versus clothing-labelled. One model replaces NudeNet's 18-class box soup for the entire modesty tier, with pixel-accurate regions you can blur directly instead of silhouettes. ~15–40ms at 512² on GPU — affordable for images, probably not every video frame, perfect as the "accuracy pack" you already have infrastructure for (DP3 §3). **Verify the license on the specific checkpoint**; base SegFormer weights carry NVIDIA non-commercial terms, and AGPL distribution makes that non-negotiable.

**CLIP/SigLIP on instance crops with a text-prompt bank.** MobileCLIP (~35MB) or ViT-B/32 ONNX, single-digit ms for a 224² crop on GPU. Zero-shot score against prompts like "a woman in a sports bra," "a woman in a hijab," "a fully clothed person." Weak at fine-grained attributes and carries known biases, so blend it — don't make it an oracle. But the product consequence is large: **modesty defined in editable text, not frozen in weights.** Different awrah definitions become a data file. No competitor can match that, because theirs are baked into proprietary weights.

**Where a fine-tune is transformative** — and it isn't a detector. It's a **binary modest/immodest classifier on person crops**: frozen CLIP embeddings + logistic-regression head, 2–3k hand-labelled crops, ~50 lines, trains on a laptop CPU in under an hour, no GPU, no detection annotation. You don't need better localization than YOLOv8-seg already gives you; you need a better *judge* on crops you already localize. That's the realistic path past Porda's proprietary weights, and your own reveal telemetry (below) generates the training set for free.

---

# 4. How to surpass both competitors

### Temporal coherence — both competitors are stateless, and this is free
Track person instances across frames (mask-IoU + greedy assignment, ~30 lines, no model). Accumulate evidence per *track*, decide on the track, not the frame.

- One confident frame protects the next 30 → you can run the detector *less* and be *more* reliable.
- Zero flicker, ever.
- Motion prediction lets you translate and dilate masks between passes — fixes the trailing-reveal problem above.
- Single-frame FPs get voted out by neighbours → you can run **lower** thresholds at the same precision.

This moves your precision/recall tradeoff from per-frame to per-track, which is a strictly better operating point, reached with pure CPU logic. It's also what makes you *feel* faster without being faster: a tracked blur is pinned to the subject, where a stateless 60fps detector still jitters.

### Zero-leak mode — the thing the whole category gets backwards
Every extension here blurs *after* the first frame paints. For this audience a 300ms exposure of haram content is a **product failure**, not a latency metric. Your hysteresis (`docs/DETECTION_PARITY.md:130`) explicitly *delays* blur — correct for flicker, wrong for the core promise.

Ship an opt-in strict mode: in-viewport unverified images held hidden until verdict; newly-playing video starts **blurred and clears** once N frames verify. Blur-first-then-clear instead of clear-then-blur. That's a guarantee no competitor offers, and it's the one guarantee your user actually wants.

### UX nobody has
- **Per-person hold-to-reveal.** Unblur one silhouette, auto-re-blur on release. Needs per-instance identity — i.e. the tracker.
- **Calibration by example instead of a strictness slider.** Show 15 images spanning the modesty spectrum; the user marks where their line is; you fit the thresholds. Nobody can reason about "strictness: 73." With a CLIP embedding space this is a 1-D projection fit — trivial.
- **Scholarly presets as reviewable data.** Named awrah-definition policy packs, AGPL, community-maintainable. Structurally impossible for a proprietary-weights competitor.
- **"Why was this blurred"** in advanced mode. Trust is the scarcest resource in a product that makes mistakes on religious grounds.
- **Site-shape awareness.** You already have `src/content/youtube.ts`. Pre-blur the whole thumbnail grid on scroll-in and clear progressively — far better than per-image pop-in, and it composes with zero-leak mode.

### Content types both competitors miss
Anime/illustrated (NudeNet is weak here; CLIP is not), text-overlay thumbnails, CSS background images, canvas/WebGL video, `<picture>`/srcset variants, PDF-embedded images, and animated WebP/GIF. Each is a "they can't do this" bullet.

---

# 5. What you are not measuring

**1. Recall on the content type that matters.** Your dev split is 17 images of fine-art nudes, portraits, and group photos. It contains approximately zero instances of the real target distribution: fitness/gym, swimwear, influencer selfie, music-video frames, anime, text-overlay thumbnails, 9:16 crops. 94.1% accuracy on n=17 has a 95% CI of roughly ±11 points — that's a vibe, not a measurement. The fitness gap didn't "survive an oversight"; **your eval has no statistical power to detect it.** You need ~200 stratified dev images with per-type recall reported separately. DP6 step 4 is right but badly under-scoped — a "fitness slice" of six images reproduces the exact error at smaller scale.

**2. Anchor-rate on negatives.** The single number from §1. Measure it first.

**3. Time-to-first-blur (p50/p95) and total leak-duration per session.** Milliseconds of unblurred haram pixels visible per page. *This is the product's actual KPI* and nobody measures it. "Porda feels fast" is this number, and you're optimizing inference ms instead.

**4. Video stability.** Blur-state transitions per minute on a steady scene (flicker rate), and mask-vs-ground-truth IoU as a function of time-since-pass. The second one tells you whether carry-forward is safe; right now you're guessing.

**5. Main-thread cost in the content script, separate from inference.** Mask rasterization, `toDataURL`, layout/paint, long tasks >50ms, INP on the host page. An extension that makes YouTube scroll badly gets uninstalled regardless of detection quality — and §2 says this is your likely real bottleneck.

**6. False-blur rate in the wild.** Instrument a local-only counter of user-initiated reveals per 100 blurs. A reveal is a free labelled false positive — offline, consented, no network. That's your production precision metric *and* your fine-tune training set. This is the flywheel neither competitor can build, because neither is offline-first and open.

**7. Power.** Sustained GPU/CPU% and battery drain over a 10-minute YouTube session versus competitors. 7.5fps continuous inference on every visible video is a real claim you haven't priced.

**8. Gender-classifier error stratified by face size (px), yaw, occlusion, and skin tone.** If DP6 hangs on `FACE_FEMALE`, its error profile is load-bearing and currently unknown. The skin-tone axis is mandatory, not optional.

**9. A growing adversarial regression set.** You have dev/test splits with proper hygiene — good. You have no "images that failed in the wild" suite. These six YouTube thumbnails should be entry #1 in a set that only grows, every user-reported miss appended, run on every PR.

---

# The 3 highest-leverage moves

**1. Flip to person-instance-first, with self-calibrated skin fraction as the modesty signal.** Regions come from person instances (built), gated by associated-face gender (built) plus skin-fraction-inside-mask calibrated against that person's own face chroma (new, ~40 lines, no model, no training). NudeNet demotes to escalation-only. This fixes fitness content, headless torsos, leggings-from-behind, *and* the 0.81-negative problem in one move — without touching a single threshold.

**2. Per-person tracking with cross-frame evidence accumulation.** Pure CPU logic, no model, no training. Buys zero-flicker video, safely lower thresholds via track-level voting, motion-carried masks that don't trail, and per-person reveal UX. Both competitors are stateless — this is where you pass them rather than match them.

**3. Rebuild the eval around content types: ~200 stratified dev images, per-type recall, plus leak-duration and main-thread cost.** Boring, cheap, and the direct cause of the problem you're currently fixing. Without it you cannot tell whether moves 1 and 2 worked, and you will ship this exact class of gap again.

**Do not build DP6 as written.** Steps 1–3 cost you weeks of geometry tuning, land a filter uncorrelated with your real FPs, and are nullified in the common case by escalation rules already in `region.ts`. Keep step 4 (expand it 30×) and drop step 5.

**Before anything:** run the anchor-rate measurement on dev-split negatives. One script, one afternoon, and it either kills DP6 outright or tells you I'm wrong — in which case the geometry is worth building after all.