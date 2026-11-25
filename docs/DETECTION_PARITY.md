# Detection parity plan

Status: proposed. Source of truth for the detection-speed/recall workstream.
Complements `docs/PLAN.md` (v1 scope); where the two conflict on numbers
(input sizes, frame rates, caps), this document wins for the workstreams it
owns, and PLAN.md should be updated when each phase lands.

## Why this exists

User testing against two shipping extensions shows our detection feels slow
and unreliable by comparison. Reverse engineering and source reading give a
clear picture of why:

|                     | HaramBlur upstream 0.6.11                               | Porda-AI 1.1.x                | this repo (today)                            |
| ------------------- | ------------------------------------------------------- | ----------------------------- | -------------------------------------------- |
| ML runtime          | TF.js **WebGL (GPU)**                                   | TF.js **WebGL (GPU)**         | onnxruntime **WASM (CPU)**                   |
| detector input      | 640×640 (416 turbo)                                     | 320×320                       | 320×320                                      |
| video frame capture | scaled to 640px                                         | scaled to ≤320px              | ≤480px, then 320 at inference                |
| video cadence       | 15–25fps capture loop                                   | continuous one-in-flight loop | fixed ~2fps                                  |
| detector            | proprietary 3-class + Human face/gender + nsfwjs enrich | proprietary 2-class YOLO      | NudeNet 18-class + YuNet + FastFace + nsfwjs |
| strictness model    | per-class thresholds scaled down to ~0.1 floor          | —                             | single threshold knob                        |

The decisive shared factor is the **GPU backend**. Both competitors feel fast
because inference costs single-digit milliseconds, which makes high input
resolution and high frame rates free. Every cost-driven compromise in our v1
design (320px input, 480px frame cap, 2fps sampling, aggressive person-pass
rate caps) is a WASM-budget decision — correct engineering for the wrong
hardware layer.

Secondary factor: the **continuous one-in-flight frame loop** (Porda) — no
fixed fps; capture the next frame the moment the previous verdict arrives,
back off when static/hidden/idle. This extracts maximum throughput from
whatever backend exists and is worth adopting independent of the backend
outcome.

What we cannot and will not copy: both competitors' detection weights are
proprietary and license-restricted. Parity means matching their _performance
envelope_ with open models, not their artifacts.

## Goals (measurable)

G1. NudeNet 320n@320 inference **< 15 ms** on the GPU path (WASM baseline
today: ~47 ms), measured in the offscreen-worker context on reference
hardware (M-series MacBook, mid-range Windows laptop if available).
G2. Effective video re-detection rate **≥ 6 fps** during active motion on GPU
(today: 2 fps fixed), with no fixed upper cap other than the power ceiling.
G3. Recall on the eval dev split **not worse** than today at equal precision
after the threshold rework; ideally measurably better at higher input
resolution.
G4. WASM-only users (Firefox, no-adapter machines) regress **nowhere**: same
or better effective throughput via the continuous loop, same input sizes
as today unless measurements say otherwise.
G5. Detection quality validated by user on the same X.com/video fixtures that
motivated this workstream.

## Non-goals

- No TF.js migration. onnxruntime-web stays; the WebGPU EP is the GPU path.
- No copying of proprietary models, no runtime model downloads (zero-network
  guarantee stands — everything bundled or explicitly user-downloaded packs).
- No accounts/premium/companion/site-blocking scope (upstream platform
  features remain out of scope per PLAN.md).
- Person-segmentation mask UX work is deferred until DP3 lands — polishing
  masks on a starved pipeline is wasted effort.

## Work packages

### DP0 — WebGPU reachability spike (small, decisive, do first)

Verify what is actually reachable, before any build-out:

1. In a built MV3 offscreen document, inside a dedicated Worker:
   `navigator.gpu` presence, adapter request result, adapter limits
   (maxBufferSize, maxComputeWorkgroups), fp16 shader support.
2. If the worker context fails: same probe in the offscreen document main
   thread. The engine host architecture (worker inside offscreen) can be
   adjusted — the worker exists for ONNX WASM threading isolation, not as an
   end in itself.
3. Load `ort.webgpu` build in that context, run NudeNet 320n zero-tensor
   inference, time cold/warm. Record shader-compile latency on first run.
4. Firefox: probe `navigator.gpu` in the background page/worker. Record only;
   Firefox staying on WASM is an acceptable outcome.

Deliverable: a measurement table in the PR description and a go/no-go on DP1.
If no-go (no adapter in any extension context): DP2 and DP4 still proceed
(they are backend-agnostic), DP1/DP3 are re-scoped around WASM-threaded
optimizations instead.

### DP1 — WebGPU execution provider in the engine

1. Engine backend selection: try `webgpu` EP first, fall back to `wasm`
   (current behavior), per session, cached per model. Some models may fail on
   WebGPU while others succeed — selection is per-model, not global.
2. Bundling: ship the `ort.webgpu` artifacts; verify WXT/Vite copies the
   wasm+jsep binaries and the worker can resolve them in the built extension
   (not just dev).
3. dtype handling: YOLOv8n-seg currently ships fp16 for size. WebGPU fp16
   shader support is adapter-dependent — keep the fp32 variant as the
   automatic fallback when fp16 is unsupported. No user-facing setting.
4. Shader-compile warm-up: run one zero-tensor inference per session at load
   (we already warm up for WASM); measure cold-start regression budget
   (< 2 s total model load remains the v1 acceptance bound).
5. Memory: enumerate adapter limits; downscale input or fall back when a
   frame bitmap exceeds texture limits rather than erroring (upstream's GPU
   memory error handling is a cautionary tale — they hard-fail detection).
6. Settings surface: `auto` only. No user-facing backend picker in v1 —
   measure, don't ask. Debug logging of the chosen backend per model in the
   advanced tab diagnostics is acceptable.

Acceptance: G1 numbers in PR; 320n, YuNet, FastFace, nsfwjs all running on
WebGPU in Chromium; forced-WASM fallback verified by test flag; unit tests
for the selection logic; `pnpm check`/vitest/build green.

### DP2 — Continuous video sampling loop

Replace the fixed-rate sampler timing with the one-in-flight pattern,
preserving everything already fixed in PR #33:

1. Loop shape: capture frame → classify → on verdict arrival, immediately
   schedule the next capture. No timer in the hot path; inference latency is
   the natural rate limiter.
2. Ceilings and floors: hard ceiling ~10 fps even when inference is faster
   (power/thermal); floor unchanged behavior when the tab is hidden, video
   paused/ended/off-screen — existing pause rules stay.
3. Adaptive backoff (from Porda, generalized): when N consecutive verdicts
   are clear and the scene-hash is unchanged, exponentially back off toward
   ~1 fps; any blur verdict, scene change, or seek resets to full rate. The
   existing scene-change detection becomes an input to this, not just a rate
   trigger.
4. Keep: rVFC where available, hysteresis so blur doesn't flicker, the
   re-parenting/re-observation fixes, one-in-flight guarantee (never queue
   frames).
5. The person-pass grant logic (movement-triggered, rate-capped) survives
   unchanged in WASM mode; on GPU the caps can be re-measured and loosened in
   DP3 since cost drops ~10×.

Acceptance: effective re-detection rate ≥ 6 fps during motion on GPU (G2),
no regression in the X.com re-parenting fixture, CPU usage on WASM-only
machines no worse than today (backoff works), unit tests for the loop state
machine with fake clock.

### DP3 — Resolution headroom

With GPU throughput, resolution becomes a quality lever instead of a cost:

1. NudeNet input size: same dynamic-input treatment FastFace got in PR #31.
   The ONNX has dynamic H/W — input size becomes a per-model metadata value,
   not a hardcoded 320.
2. Measure on the eval dev split: 320n weights at 320/416/480 input; time on
   GPU. Pick the default per backend (e.g. 320 on WASM, higher on GPU) from
   measurements, not intuition.
3. 640m pack: the hash-verified 103 MB model from the earlier investigation
   becomes a curated optional download pack — accuracy-over-speed for users
   who opt in, exactly the pack-system use case it was designed for. Only
   after measurements show it beats scaled-up 320n on the dev split.
4. Video frame capture cap: raise from 480px to match the active detector
   input size; capture and inference resolutions should not compound
   downscales.
5. Person segmentation on GPU: re-measure. If a pass costs tens of ms, the
   movement-gated rate caps can loosen substantially — revisit the 1.5 s
   interval and displacement thresholds with real numbers.

Acceptance: eval table (dev split only, never the held-out test split) in the
PR: precision/recall per class at each input size; chosen defaults justified
by the table; G3 holds.

DP3 implementation note (2026-11-25): bundled 320n uses 320² on WASM and
480² on WebGPU; video capture follows the active detector edge. Dev-split
measurements are recorded in the DP3 PR. The optional hash-verified 640m
pack is retained as an opt-in accuracy/region-quality tradeoff, not a speed
default.

### DP4 — Sensitivity & threshold architecture

Upstream feels more sensitive because it is: strictness scales per-class
thresholds down to a ~0.1 floor for porn/sexy classes. Ours is a single knob.

1. Per-class threshold table in `policy.ts` (pure, unit-tested): each
   blur-capable label gets base threshold + strictness scaling curve. High
   strictness biases aggressively toward blurring; low strictness toward
   precision. Curves derived from eval dev-split measurements, documented in
   the PR.
2. Gender-consistency and child-skip logic re-checked against the new
   thresholds — interactions between low thresholds and FastFace confidence
   gating need eval coverage.
3. nsfwjs whole-image pass: align its thresholds with the same curve model;
   it currently has its own logic.
4. Migration: existing strictness values (0–100) map onto the new curves —
   no settings version bump needed if the mapping is continuous; document the
   mapping.

Acceptance: precision/recall table per strictness level on the dev split; no
regression at default strictness (G3); user validates sensitivity feel on
live sites.

DP4 implementation note (2026-11-25): per-class `{max, min, pow}` curves in
`policy.ts` replaced the flat genitalia floor and noisy-exposed margin. Curves
were derived from dev-split detection-score distributions mined from the DP3
eval runs. At default strictness the table is regression-neutral (94.1%
accuracy, 100% precision, 88.9% recall — identical at strictness 0 and 0.5);
at strictness 1 recall reaches 100% (the `woman-elderly-gambian` miss is
recalled) with the documented false-blur tradeoff on negatives.

### DP5 — Validation, then masks

1. Full acceptance pass: G1–G5, plus the v1 acceptance items this touches
   (model load < 2 s, no-network audit).
2. Only now revisit person-segmentation mask UX: association strictness,
   carry-forward tuning, silhouette vs box tradeoffs — with a pipeline that
   can actually afford the passes.
3. Optional stretch: user-facing per-site or per-mode performance/quality
   toggle if measurements show real divergence in what hardware can afford.

## Sequencing

```
DP0 spike ──▶ DP1 WebGPU ─┬─▶ DP3 resolution ─▶ DP4 thresholds ─▶ DP5 validation+masks
              └─▶ DP2 continuous loop (parallel-safe, backend-agnostic)
```

DP0 first — it is small and decides DP1's shape. DP2 can run concurrently
with DP1 (different files, backend-agnostic value). DP3 strictly after DP1.
DP4 can start after DP0 (pure logic + eval) but its final tuning wants DP3's
resolution numbers. DP5 last.

One PR per work package, single-purpose, per repo convention.

## Risks

- **WebGPU absent in offscreen worker.** Mitigation: probe first (DP0);
  fallback shapes include running the engine on the offscreen main thread or
  in a background-page worker. Worst case: WASM-only improvements from DP2/DP4
  still ship — parity gap narrows but doesn't close.
- **Shader-compile jank / first-inference latency** on WebGPU. Mitigate with
  warm-up at load; measure cold-start.
- **fp16/fp32 fragmentation.** Handled per-model with automatic fallback;
  never expose to users.
- **Power/thermal** on laptops at sustained high fps. Mitigated by the
  10 fps ceiling + backoff; measure CPU/GPU% during validation.
- **Firefox** stays WASM for now — G4 exists precisely to keep that path
  honest.
- **Eval contamination**: all tuning on the dev split only. The held-out test
  split is touched once, at DP5, for final validation.
