# Gaze-protection parity plan

Status: action plan. **Supersedes** the Part 4 pivot of `docs/COMPETITIVE_PARITY.md:508-733`
(whose factual Parts 0–3 stand), the DP6 proposal killed by
`docs/reviews/dp6-eval-opus-5.md` and `docs/reviews/dp6-eval-opus-4-8.md`, and the
policy sections of `docs/PLAN.md:66-77` / `docs/DETECTION_PARITY.md:173-203`.

**Date:** 2026-11-25.

---

## 0. The reframe, stated once

The product is **gaze protection for practising Muslims browsing the ordinary web**
— YouTube, Instagram, X, news, shopping. The user is never on a porn site. The harm
is *involuntary exposure*: a woman appears in a thumbnail, feed card or ad, and the
obligation to lower the gaze is already violated by the time the user can look away.

Four consequences, and every section below is downstream of them:

1. **The detection target is women as persons.** Not body parts, not clothing state,
   not nudity.
2. **Nudity/NSFW is a secondary escalation layer**, at most. It changes *how* we
   censor, never *whether*.
3. **The per-instance question is binary: is this a woman?** How much of her is
   covered — face, head, figure, whole element — is a **user preference**, a
   separate axis, never an output of the detector.
4. **Covered-vs-exposed is a question the product does not ask.** It is also the
   question our entire detection stack is built to answer. That mismatch is the
   whole problem.

### 0.1 What this deletes outright

| Dead | Why |
| --- | --- |
| DP6 female-face anchoring | Gates the flagship case on the signal that case most often lacks (`dp6-eval-opus-4-8.md:11-12`). Under the reframe it is not merely weak — it is answering the wrong question with the wrong evidence. |
| `strictModesty` / `MODESTY_LABELS` as a gate (`src/shared/policy.ts:174-180`) | "Is her belly bare" is not the question. Neither competitor asks it; both beat us. |
| `FEMALE_BREAST_COVERED` calibration work | Load-bearing only in a coverage-based policy. Stops existing. |
| Self-calibrated skin fraction (`dp6-eval-opus-5.md:87-96`) | Elegant, and a coverage metric. Coverage is a preference now. Dead. |
| Clothes-parsing segmentation (SegFormer/ATR) | Per-pixel answer to a deleted question, at 15–40 ms and a licence minefield. |
| Per-class strictness curves over 18 NudeNet classes (`src/shared/policy.ts:97-124`) | Collapses to one threshold on one question. Keep `curveThreshold()`; delete the table. |

That is roughly all of DP4 and all of DP6. Retiring a week of threshold tuning is
the cheapest part of this plan; the expensive part is admitting it early.

---

## 1. Architecture under the reframe

### 1.1 The question, decomposed by element size

"Is there a woman here" has two different cheapest answers depending on how big the
element is, and conflating them is a mistake:

- **Small/medium media (thumbnails, feed cards, avatars, ≲400 px):** the subject
  *is* the image. A whole-image classifier answers it in one pass, with no
  localisation, and whole-element censor is the correct — arguably preferable —
  output anyway. Upstream already forces whole-blur under 70 px
  (`COMPETITIVE_PARITY.md:398`); we extend the same logic much further up.
- **Large media and video frames:** the subject is a fraction of the frame. A
  whole-image classifier sees her at 20 px after the 224² resize and fails.
  Localisation first, classify crops second.

### 1.2 Target pipeline

```
element
  │
  ├─ Tier 0  gate  ── image-level "woman present?"  (1 × 224² forward pass)
  │     small element  → verdict is final → whole-element censor / clear
  │     large element  → gate is a cheap reject only; continue
  │
  ├─ Tier 1  localise ── YOLOv8n-seg person instances, 640²  (already shipped)
  │     ≤4 instances, masks, COCO person class
  │
  ├─ Tier 2  decide  ── per instance: is this a woman?
  │     a) crop classifier on the instance crop        ← the new model
  │     b) face gender (YuNet→FastFace) when a face lands in the instance
  │     fuse: face-gender wins when ≥0.85 confident, crop classifier otherwise
  │     no instance found but Tier 0 says woman → whole-element censor
  │
  └─ Tier 3  escalate ── nsfwjs (and optionally NudeNet) only upgrades the censor:
        region → whole element, blur → solid. Never enables a blur that
        Tier 2 declined.
```

Average cost **falls**. Today every image pays 85 ms of NudeNet
(`docs/ENGINE_NOTES.md:3`). Under this design the overwhelming majority of web
images (logos, products, landscapes, men, text) exit at Tier 0 for one small
forward pass. The full ladder runs on the minority that contains people.

### 1.3 The three candidate answers to Tier 2, scored case by case

Legend: ✅ works · ⚠️ degraded · ❌ structurally fails.

| Failure case | **A. Face-gender** (shipped: YuNet 640² → FastFace 112²) | **B. CLIP/SigLIP zero-shot on person crop** | **C. Small fine-tuned crop classifier** |
| --- | :--: | :--: | :--: |
| Frontal face, ≥48 px | ✅ strongest signal there is | ✅ | ✅ |
| Faceless torso (neck-cropped fitness thumbnail) | ❌ no face → no verdict | ⚠️ clothing/body cues; **unmeasured** | ✅ if the training set contains them |
| Back to camera (leggings/gym) | ❌ | ⚠️ hair, hips, garment; **unmeasured** | ✅ |
| Face occluded (barbell, arm, sunglasses, text overlay) | ❌ / ⚠️ | ✅ rest of the figure carries it | ✅ |
| Extreme yaw / profile | ⚠️ FastFace abstains → `FACE_PERSON` → no blur | ✅ | ✅ |
| Small figure in a grid thumbnail | ⚠️ face below YuNet's useful floor | ⚠️ crop upscaled from ~60 px | ⚠️ same limit |
| Anime / illustrated woman | ❌ face models are photo-trained | ✅ **CLIP's native strength** | ⚠️ needs anime in the training set |
| Statue / painting / 3D render | ⚠️ | ✅ | ⚠️ |
| Motion-blurred video frame | ❌ | ⚠️ | ⚠️ |
| Child vs adult | ✅ **only path to `age`** | ❌ | ❌ |
| Non-Western dress (thobe, kurta, abaya) | ✅ face is face | ⚠️ **documented CLIP bias risk** — must be an eval slice with a bar | ✅ if represented |
| Bundle cost | shipped (0.2 MB + 1.6 MB) | 16–90 MB, licence-dependent | ~5 MB |
| Per-crop cost | ~3–8 ms | ≈5 ms WebGPU / 20–40 ms WASM *(estimate — E2 measures)* | ~2–5 ms |

**Read the table as a union, not a contest.** A beats B and C where a face exists
and is the only route to `skipChildren`. B/C are the only route to the four rows A
fails outright — which are exactly the target content. **Ship both. Fuse them.**
A stays in the pipeline for its precision and for age; B-or-C becomes the default
answer and the sole answer when no face exists.

The headline consequence: today, an instance with no face produces **no verdict at
all**. Every case in rows 2–5 is currently a silent miss, and
`src/engine/worker.ts:656-664` makes it structural — the person pass only runs when
`coverageMode === "body"` **and a `FACE_FEMALE` already exists**. Our best-trained
model is gated behind our weakest signal. That gate is inverted and is the first
thing P2 deletes.

### 1.4 Why person-instance crops and not whole-figure detection

Both competitors trained a detector whose **classes encode gender**
(`COMPETITIVE_PARITY.md:61-68, 234-245`). That requires box annotations: expensive,
and it rebuilds something we already have. Decomposing it into
*COCO-person detection* (free, excellent, shipped) × *crop classification* (cheap to
train or to do zero-shot) reaches the same answer and routes around their only moat.
Upstream agrees with us: their bundle contains a dormant
`person-detect → crop classifier` path (`COMPETITIVE_PARITY.md:368-376`), built and
not shipped. We can ship it first.

### 1.5 Model-size reality, which can kill option B

A browser extension cannot bundle a ViT-B/32 image tower (≈88 M params ⇒ ≈88 MB at
int8). Candidates that could fit, **all unverified until E2**:

| Candidate | Image tower | Licence | Risk |
| --- | --- | --- | --- |
| TinyCLIP-ViT-8M-16 (Microsoft) | ~8 M ⇒ ≈16 MB fp16 | MIT | Accuracy at 8 M is the open question |
| MobileCLIP-S0 (Apple) | ~11 M | **Apple ASCL, not OSI** | Likely incompatible with AGPL redistribution — assume blocked until legal-checked |
| SigLIP-base (Google) | ~86 M | Apache-2.0 | Too big to bundle; **viable as an opt-in pack** |
| OpenAI/OpenCLIP ViT-B/32 | ~88 M | MIT | Same — pack only, or offline teacher |

We already ship a **103 MB opt-in pack** (`src/shared/model-packs.ts:56-68`) with
pinned SHA-256 and OPFS install, plus a typed slot system
(`src/shared/model-slots.ts:21-52`). So the shipping shape is settled regardless of
which way E2 lands:

- **bundled default:** the smallest model that clears the accuracy bar;
- **opt-in accuracy pack:** the big Apache-2.0 encoder for users who want it;
- **offline-only:** the big encoder may also be used purely as a *teacher* to
  pseudo-label crops and distil into a 5 MB student that ships — which neatly
  dissolves the size constraint entirely.

---

## 2. Is a fine-tune now unavoidable?

**Short answer: a fine-tune is probably required, but not the fine-tune the
competitors did, and it is roughly two orders of magnitude cheaper.**

Long answer, split by what is actually being asked:

| Sub-question | Needs training? |
| --- | --- |
| Where are the people? | **No.** COCO person is one of the best-trained classes in computer vision, already shipped (`src/engine/person.ts`), already masked and associated. |
| Is this person a woman? | **Maybe.** Zero-shot CLIP is the only no-train candidate. Its accuracy on *frontal photo* gender is well above 90 % in the literature; its accuracy on **faceless torsos and back views at thumbnail resolution** is, as far as we can tell, unpublished. Unknown ≠ good. Measure it (E1). |
| Is this a child? | **No** — FastFace already emits `age`. |
| Is this explicit? | **No** — nsfwjs/NudeNet already shipped, and it is escalation-only now. |

### 2.1 If zero-shot is not enough, the escalation ladder — cheapest first

1. **Prompt-bank engineering + calibration.** ~20 prompts per class, averaged
   embeddings, threshold fit on dev. Hours. Typically worth several points.
2. **Linear probe on frozen CLIP embeddings.** Logistic regression, ~50 lines,
   CPU-minutes, no GPU, **no box annotation** — only crop-level `woman / man / neither`
   labels. Typically +3–8 points over zero-shot. *This is the realistic end state.*
3. **Distil to a 5 MB student.** Big teacher offline → MobileNetV3-small on crops.
   Solves bundle size and inference cost at once. One afternoon, CPU-trainable.
4. **Full fine-tune / LoRA of the backbone.** One rented GPU for a weekend
   (`dp6-eval-opus-4-8.md:44`). Only if step 2 shows the frozen embeddings do not
   separate the classes — i.e. if the linear probe ceiling is itself below bar.

Steps 1–3 need no GPU and no detection annotation. **Step 4 is the only one that
resembles what Porda did, and we should expect never to reach it.**

### 2.2 Data: how much, how obtained, what it costs

| Purpose | Volume | Method | Cost |
| --- | --- | --- | --- |
| **Validate** zero-shot (E1) | 300–500 crops | Mined from the new eval images via our own person detector; hand-labelled | ~30 min labelling |
| **Calibrate / linear probe** | 2 000–3 000 crops | Same source; CLIP pseudo-labels pre-sort them so the human only *confirms or flips* | ≈2–3 h at ~3 s/crop |
| **Distillation** (if used) | 20 000+ crops | **No human labels at all** — teacher CLIP labels them; only a held-out slice is hand-checked | ~0 |

Annotation cost is not the bottleneck. **Sourcing legally redistributable images in
the target distribution is.** Respect the existing fixture policy
(`tests/fixtures/SOURCES.md`): public-domain/CC only in-repo. Everything else is a
local-only, git-ignored corpus with a manifest of URLs — same hygiene as
`scripts/fetch-models.ts`.

One licensing trap, stated now: pedestrian-attribute datasets (PA-100K, RAP, PETA)
and models trained on them carry research-only terms. They look like a shortcut to
whole-body gender; under AGPL redistribution most of them are not usable. Do not
plan on them.

---

## 3. What happens to the existing stack

| Component | Fate | Rationale |
| --- | --- | --- |
| **YOLOv8n-seg person** (`src/engine/person.ts`, 640², masks, association) | **Promoted to backbone.** Un-gate it (`src/engine/worker.ts:656-664`). Raise `PERSON_MAX_INSTANCES` (4) once measured — group fitness shots exceed it. | Our best model, currently cosmetic. |
| **YuNet + FastFace** (`yunet.ts`, `fastface.ts`) | **Kept, demoted to a vote.** Highest-precision gender signal when a face exists; **sole source of `age`**. Keep the 0.85 abstention gate — abstention is now cheap because the crop classifier still answers. | Precision + `skipChildren`, a feature neither competitor has. |
| **nsfwjs** (`nsfw.ts`, 224²) | **Kept, escalation only.** Upgrades censor scope/style; never enables a blur. | Cheap porn/suggestive safety net. |
| **NudeNet 320n** (`nudenet.ts`, 12 MB, 85 ms) | **Demoted to escalation; candidate for deletion.** Keep only the `CORE_EXPOSED` subset for the "explicit close-up with no detectable person" case. | Under the reframe its 18 classes answer a question we stopped asking. If P3's eval shows nsfwjs + person + crop classifier catches everything it catches, **delete it**: −12 MB, −85 ms/image. Measure before removing. |
| **`FEMALE_BREAST_COVERED`, all `MODESTY_LABELS`, `COVERED_LABELS`** | **Deleted from policy.** | Miscalibrated answers to a deleted question. |
| **`strictModesty` setting** | **Removed** (migration drops it). | Nothing left for it to gate. |
| **Per-class threshold curves** (`policy.ts:97-124`) | **Deleted.** Keep `curveThreshold()` for the two surviving thresholds. | One question, one dial. |
| **Male-context suppression** (`policy.ts:52-66`) | **Deleted.** | A hand-tuned geometry patch for a part-label pipeline that no longer exists. Instance-level classification subsumes it. |
| **New: gender crop classifier** | New model slot (`ModelSlot` union + `MODEL_SLOTS`) + optional pack. | The one genuinely new component. |

### 3.1 How much of DP0–DP5 survives

Honest accounting — most of it, because the reframe invalidates the **taxonomy and
policy**, not the **plumbing**:

| Work | Verdict |
| --- | --- |
| DP0/DP1 WebGPU EP, per-model backend selection | **Fully retained, more valuable.** Per-instance crop classification is only affordable on GPU. |
| DP2 continuous video loop, sampler, backoff, scene change | **Fully retained.** Untouched. |
| DP3 resolution headroom, model slots, model packs, OPFS overrides, hash-pinned fetch | **Fully retained and now load-bearing** — this is exactly the mechanism for shipping a new classifier slot, and later an upgraded head, without a store release. |
| DP4 threshold architecture | **~80 % retired.** The mechanism (`curveThreshold`) survives; the 18-class table and its tuning do not. |
| DP5 masks, transport, `carryMasksForward`, validation | **Retained and promoted** — masks become the censor geometry, not a flourish. |
| Region portal, blur rendering, reveal UX, queue, discovery, `youtube.ts`, video manager, settings/profiles/i18n, E2E | **Entirely untouched.** This is most of the codebase. |
| Eval harness (`scripts/eval.ts`, `eval-compare.ts`, `probe-dets.ts`, dev/test split hygiene) | **Retained; labels change.** Manifest v2's `expect: [NudeNet classes]` must become `women: [boxes]` + `womanPresent`. Schema work, not tooling rework. |
| Zero-leak (`blurryStartMode`) | **Already shipped and already ON by default** (`src/shared/settings.ts:120`), unlike upstream. `COMPETITIVE_PARITY.md:484` marks us 🔴 here — **that row is wrong**; the primitive is implemented in `blur.ts:96-101`, `video/manager.ts:93`, `index.ts:156`. What is missing is *measurement*, not the feature. |

---

## 4. Strictness and UX semantics

Detection is binary; coverage is preference. Three independent user-facing axes,
and they must never be collapsed into one slider (`dp6-eval-opus-5.md:65-66`).

### Axis 1 — Who (binary gates)

`blurFemale` (default on), `blurMale` (default off). **Male detection is the same
pipeline with a different output head read** — one forward pass yields both scores,
so `blurMale` costs nothing extra. This is a straight generalisation of what
upstream does (`COMPETITIVE_PARITY.md:234-245`).

Plus one new mode worth having precisely because it is *structurally* perfect:

> **"Blur every person."** Skips Tier 2 entirely — YOLOv8n-seg instances, no gender
> classification, therefore **zero gender-classifier error**. The strictest setting
> is also the most reliable one, and it is free. No competitor offers it.

### Axis 2 — Sensitivity (the surviving meaning of `strictness`)

One number: the confidence bar on *is this a woman*.
`womanThreshold = curveThreshold({max, min, pow}, strictness)`, with `max/min`
**fitted to the measured ROC from P1, not chosen by taste.**

The slider's label should state its measured consequence, not an abstraction —
"blurs ~1 in N clear images by mistake, misses ~1 in M women" read straight off the
eval curve. Nobody can reason about "strictness 73" (`dp6-eval-opus-5.md:126`); we
are the only project in this space with an eval set capable of printing the real
number, so print it.

There is no separate "unsure" band. Three states is one knob too many; "unsure"
simply sits on one side of the single bar, and under gaze protection the bar
defaults to the safe side.

### Axis 3 — Coverage (pure preference, never derived from detection)

`coverageMode`: `face` → `head` → `body` (instance mask) → `whole` element, plus
`wholeImages` for "always censor the whole thumbnail". **Default moves `head` →
`body`** (`src/shared/settings.ts:135`).

Consequential detail: `escalateAreaPct 0.30` (`settings.ts:137`) plus
`SMART_CENSOR_MAX_FEMALE_FACES` (`region.ts:200-204`) will escalate almost every
body-coverage verdict to whole-element — `dp6-eval-opus-5.md:46-57` flagged this as
hypothetical; under this plan it is *immediate*. Either raise the threshold or
accept whole-element as the honest default for feed media (which is what Porda
effectively does). Decide with data in P2, not now.

`skipChildren` stays on by default and becomes a **requirement**, not a nicety: a
product that blurs children because it cannot tell them from women is a worse
product, and we are the only one with the signal to avoid it. Known gap, accepted:
a child whose face is not detectable may be classified as a woman and blurred. That
error is in the safe direction.

---

## 5. Be brutal: what will not work

1. **Face-anchored gender as the primary gate is dead.** Not weak — dead. Rows 2–5
   of §1.3 are the target distribution, and face-gender scores ❌ on all of them.
2. **The entire coverage/modesty line of thinking is dead for parity.** Skin
   fraction, clothes parsing, covered-vs-exposed calibration: a beautifully
   engineered answer to a question this product does not ask. Not "deferred" —
   removed from the critical path.
3. **Zero-shot CLIP will probably not hit 95 % on back-views and faceless torsos
   at thumbnail resolution.** Expect zero-shot to land somewhere in the 80s on
   those strata. Plan for the linear probe (§2.1 step 2); be pleased if it is
   unnecessary.
4. **Do not plan around SigLIP/ViT-B as the bundled default.** They do not fit.
   They are teachers or opt-in packs.
5. **Do not imitate the competitors' fine-tune.** A gender-classed whole-figure
   detector needs box annotations — 10–50× the labelling cost — to reach a place
   COCO-person + a crop head already reaches.
6. **There is no path with zero labelling.** Even the "no fine-tune" outcome needs
   ~400 labelled crops to *prove* it is the right outcome. Budget it; it is hours,
   not weeks.
7. **Watch for CLIP's cultural bias in exactly our user base's direction.** A man in
   a thobe/kurta read as "woman" is a visible false blur on content our users see
   daily. This is a mandatory eval slice with a pass bar, not a footnote — as is
   accuracy stratified by skin tone (`dp6-eval-opus-5.md:93`).
8. **Per-frame CLIP on video is not affordable on WASM.** At ≤4 instances × 20–40 ms
   it blows the 10 fps budget. Tracking (P5) is therefore not a "surpass" nicety —
   it is the cost amortisation that makes the architecture viable on video at all.

---

## 6. Sequenced plan

### P0 — Measure before writing any detection code *(2–3 days, no product code)*

This is the cheapest decisive experiment and nothing below is funded until it exists.
It runs **offline in Node with onnxruntime-node** — no extension build, no Playwright.

**E0 — person-detector coverage.** On every eval image containing a woman: does
YOLOv8n-seg emit an instance at `PERSON_SCORE_FLOOR 0.25` covering her, and does
`PERSON_MAX_INSTANCES 4` truncate group shots?
*Gate:* ≥95 % instance recall. Below that, localisation is the bottleneck and the
whole Tier-1 design needs revisiting before Tier 2 matters.

**E1 — the decisive one: gender accuracy on person crops.** Build 300–500 hand-
labelled person crops from the P1 images, stratified by: face-visible · faceless
torso · back-to-camera · small/distant · occluded face · anime/illustrated ·
conservatively-dressed · non-Western dress · men (negatives) · children. Score three
ways on the same crops:

   - **(a)** current YuNet→FastFace — and critically, its *coverage*: on what
     fraction of crops does a face exist at all? That number alone prices the
     status quo.
   - **(b)** CLIP/SigLIP zero-shot with a prompt bank.
   - **(c)** 5-fold-CV **linear probe** on the same frozen embeddings — 20 lines,
     and it reveals the fine-tune ceiling *without collecting another image*.

*Decision rule, written before the numbers arrive:*

| Outcome | Action |
| --- | --- |
| (b) ≥93 % balanced acc. overall **and** ≥85 % on faceless+back strata | **Ship zero-shot.** No training. Go straight to P3. |
| (b) below bar, (c) − (b) ≥ 5 pts, (c) ≥93 % | **Linear probe.** Collect 2–3 k crops, train the head (CPU, <1 h). |
| (c) < 90 % | Frozen embeddings don't carry the signal → scope a weekend LoRA, or reconsider whether whole-figure gender is reachable at all. **Say so out loud at that point.** |
| (a) coverage < 60 % | Confirms face-gender cannot be primary — expected, and worth having on paper. |

**E2 — shipping feasibility.** For each candidate encoder: ONNX export exists
(transformers.js/`Xenova` ONNX repos are the fast path), licence permits AGPL
redistribution, size ≤ ~30 MB bundled, latency ≤10 ms/crop WebGPU and ≤40 ms WASM
measured through our own `scripts/bench-engine.ts`. A candidate failing licence or
size is not a candidate; it is a teacher.

**Acceptance:** three tables in a PR description. Zero changes to `src/`.

---

### P1 — Eval set v3 *(M: labour, not difficulty — collect once, label twice)*

~300 dev images, stratified and reported **per slice**: fitness/gym thumbnails ·
headless torsos · back-to-camera · swimwear · influencer selfie · group fitness ·
2×2 collages · text-overlay thumbnails · 9:16 crops · anime/illustrated · **a large
clothed-women block** · **a men block** (the false-blur class `blurMale:false`
makes expensive) · non-Western dress · children · skin-tone strata · plus a
never-shrinking adversarial set seeded from `tests/fixtures/yt-*.jpg`.

Manifest v3 schema: per-image `womanPresent`, per-instance `{box, gender, faceVisible,
strata[]}`. The crops for E1 fall out of this for free — so **P1's image collection
runs first and P0's measurement runs on it**; they are one sprint, not two.

Add the two product metrics nobody measures: **time-to-first-blur (p50/p95)** and
**leak-duration per page session**.

**Acceptance:** `scripts/eval.ts` reports per-slice precision/recall and the two
latency metrics. Test split stays held out.

---

### P2 — Instance-first regions *(S, ships alone, no new model)*

Pure plumbing, immediately valuable, independent of how P0 lands:

1. Delete the inverted gate at `src/engine/worker.ts:656-664` — run the person pass
   on its own merits, not behind `FACE_FEMALE`.
2. Region source becomes the person instance (mask, or `coverageMode:"body"` box);
   default `coverageMode` `head` → `body`.
3. Blur radius scales with region size, as both competitors do
   (`COMPETITIVE_PARITY.md:173-174`), replacing fixed `blurAmount: 20`.
4. Merge overlapping regions by bounding-box union before painting (upstream does;
   we don't).
5. Re-tune `escalateAreaPct` against real measurements (§4, Axis 3).

**Acceptance:** on the fitness slice, the censored region covers ≥70 % of the
subject's visible torso box on ≥90 % of positives, with dev-split false-blur rate no
worse than the P1 baseline.

---

### P3 — The gender decision *(M — the core of the plan)*

New model slot + optional pack. Tier 0 image gate, Tier 2 per-instance scoring,
fusion with face-gender (face wins at ≥0.85, classifier otherwise), element-size
routing per §1.1. NudeNet/nsfwjs demoted to Tier 3.

**Acceptance, per-slice on P1:**

- women recall ≥0.95 overall; **≥0.85 on the faceless-torso and back-to-camera
  slices** (today: ≈0);
- false-blur rate ≤5 % on the clothed-women slice and ≤3 % on the men slice at
  default settings;
- **no slice (skin tone, dress style) more than 7 points below the overall rate** —
  a hard gate, not a report line;
- anime slice recall ≥0.80 (today: near zero);
- p95 added latency ≤40 ms/image on WASM, ≤15 ms on WebGPU.

---

### P4 — Policy and settings rewrite *(S, mechanical once P3 lands)*

Delete modesty gating, the curve table and male-context suppression; collapse to
`womanThreshold` + escalation threshold; remove `strictModesty` with a migration;
add "blur every person"; relabel the strictness slider with measured consequences.
`src/shared/policy.ts` should get **shorter**.

**Acceptance:** policy unit tests rewritten against the new semantics; no eval
regression vs P3; settings migration covered.

---

### P5 — Tracking: classify once per track *(M — now load-bearing, not optional)*

Mask-IoU + centroid greedy assignment across frames (~30 lines, no model). Classify
a track **once**, reuse for its lifetime; re-verify on a slow cadence. Carry masks
forward with motion. Replaces the symmetric 2/2 hysteresis
(`src/content/video/sampler.ts:32`) with track-level voting. Steal upstream's L2
pixel-diff frame-skip and Porda's hold-down timer alongside.

**Acceptance:** blur-state transitions per minute on a steady scene → 0; per-frame
classifier cost on video drops ≥5× versus naive per-frame; detector threshold can
drop ≥0.05 with no increase in track-level FP rate.

---

### P6 — Zero-leak: measure, then claim *(S)*

The feature exists and is on by default. What is missing is the number. Instrument
leak-duration, prove 0 ms on the YouTube grid fixture, and **fix the 🔴 in
`COMPETITIVE_PARITY.md:484`**. Then it is a claim we can make in the store listing,
which upstream (default-off) cannot.

---

### P7 — Surpass *(L, only on P1 evidence)*

Prompt-editable "who counts" as a data file · calibration-by-example instead of a
slider · per-person hold-to-reveal (unlocked by P5's identity) · local-only reveal
telemetry as a free labelled FP stream and future training set · benchmarked model
selection (upstream's 60/100 ms tiers) · dominant-background-colour cover (Porda) ·
hourly self-test inference (upstream).

### Sequence

```
P1 collect images ─┬─▶ P0 measure (E0/E1/E2)  ── gates everything
                   └─▶ P1 label + manifest v3
P2 instance-first regions (S) ── parallel, ships immediately, no new model
      └─▶ P3 gender decision (M) ─▶ P4 policy rewrite (S)
                                      ├─▶ P5 tracking (M)
                                      └─▶ P6 zero-leak measurement (S)
                                            └─▶ P7 surpass (L)
```

**First thing built:** nothing. **First thing done:** collect the P1 images and run
E1 on crops from them. One afternoon of labelling plus one script answers whether
this plan is a configuration project or a training project, and the answer changes
everything after it.

---

## 7. Where this surpasses the competitors — re-derived, not inherited

| Claim | Survives the reframe? | Why |
| --- | --- | --- |
| **Anime / illustrated women** | ✅ **Strengthened** | Both competitors' weights are photo-trained whole-figure detectors; CLIP is trained on web alt-text and is natively strong on illustration. Under the reframe anime women are squarely in scope (thumbnails, art feeds), not an edge case. |
| **Temporal coherence** | ✅ **Promoted to necessary** | Still uncontested — neither competitor tracks (`COMPETITIVE_PARITY.md:518`). Now also the mechanism that makes per-instance classification affordable on video. |
| **Zero-leak** | ✅ **Already won, unmeasured** | Shipped and default-on; upstream's equivalent is default-off. Needs a number, not a feature. |
| **Tone-invariance** | ⚠️ **Changed form** | The skin-fraction mechanism is dead. The *obligation* is stronger: gender-classifier accuracy stratified by skin tone and dress is now the load-bearing fairness metric, with a hard gate in P3. |
| **User-defined coverage** | ✅ **Cleaner than before** | Coverage ⟂ detection is now a principle, not a workaround. Four coverage levels, none of them an artefact of what the detector happened to localise. |
| **Child skip** | ✅ **New prominence** | Neither competitor has `age`. Under "blur women wholesale" semantics, not blurring children is a visible quality difference. |
| **"Blur every person" mode** | ✅ **New** | Strictest setting, zero classifier error, free. Nobody offers it. |
| **Offline / no network** | ✅ **Underrated** | Upstream fetches its NSFW model from the network at runtime and runs Firebase auth (`COMPETITIVE_PARITY.md:304, 486`). For this audience that is not a footnote — it is the whole trust story. |
| **Measured FP rate in the UI** | ✅ **New** | Only possible because of P1. "Strictness 73" becomes "misses ~1 in 20". |
| **Covered-vs-exposed nuance** | ❌ **Abandoned** | It was our only true technical lead and it answers a question the product does not ask. Let it go. |

---

## 8. Risks

| Risk | Mitigation |
| --- | --- |
| No encoder clears licence × size × accuracy simultaneously | E2 runs in P0, before commitment. Fallback is distillation (§1.5) — big teacher offline, 5 MB student shipped. |
| Zero-shot and linear probe both under bar | Then whole-figure gender is genuinely hard off-the-shelf and the honest answer is a scoped fine-tune. P0 surfaces this in days, not months. |
| Blurring men / non-Western dress becomes the new visible failure | Hard per-slice gates in P3's acceptance; men and dress slices are first-class, not negatives-in-general. |
| Body-coverage default escalates everything to whole-element | Measured in P2; either raise `escalateAreaPct` or accept whole-element for feed media and say so. |
| WASM-only devices can't afford Tier 2 | Element-size routing (§1.1) means most media exits at Tier 0; degrade to image-level verdict + whole-element censor on WASM, mirroring upstream's CPU-tier downgrade. |
| We delete NudeNet and lose a real case | Deletion is gated on the P3 eval showing no loss. Until then it stays on the escalation path. |
