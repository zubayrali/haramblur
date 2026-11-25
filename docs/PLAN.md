# HaramBlur v1 rewrite plan

This repository is a fork of [alganzory/HaramBlur](https://github.com/alganzory/HaramBlur) (AGPL-3.0). Its full history is kept. The rewrite replaces the code; it stays AGPL-3.0 and credits the original authors.

## Goals

1. **No media slips through.** New images and videos are blurred until they have been classified (fail closed, user-configurable). Media seen before the models finish loading is queued, never dropped.
2. **Better detection.** Replace nsfwjs (5 whole-image classes) and Human (CDN-loaded face/gender) with bundled models: NudeNet 320n (YOLOv8n, ONNX) for region detection (18 classes including `FACE_FEMALE`, `FACE_MALE`, and covered/exposed body parts), OpenCV YuNet for faces with landmarks, FastFace for landmark-aligned gender/age, and a bundled nsfwjs MobileNetV2 for the whole-image suggestive/anime signal the original relied on.
3. **Blur what the user sees.** Classify `currentSrc` (srcset and `<picture>`), CSS background images, `poster`, and media inside iframes.
4. **Private and offline.** Every model ships inside the extension. No CDN, no telemetry, no remote inference.
5. **Cross-browser from one codebase.** Chrome/Edge (MV3) and Firefox from the same source with WXT.
6. **Measured, tested, automated.** Unit tests, extension E2E tests, an accuracy eval harness, and CI on every PR.

## Decisions

| Area              | Decision                                                                                                                                                                                                                                                                                                                                                                    | Notes                                                                                                       |
| ----------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------- |
| Build / framework | WXT `0.21.x`, Vite, pnpm, Node 22 LTS                                                                                                                                                                                                                                                                                                                                       | `wxt build -b chrome`, `wxt build -b firefox`, `wxt zip`                                                    |
| Language          | TypeScript, `strict: true`                                                                                                                                                                                                                                                                                                                                                  | No `any` without a comment saying why                                                                       |
| Lint / format     | Biome                                                                                                                                                                                                                                                                                                                                                                       | One tool; replaces Husky + Prettier                                                                         |
| UI                | Svelte 5 + Tailwind CSS 4                                                                                                                                                                                                                                                                                                                                                   | Popup and options page; RTL-aware                                                                           |
| Messaging         | `@webext-core/messaging` (typed protocol in `src/shared/protocol.ts`)                                                                                                                                                                                                                                                                                                       | All cross-context calls go through it                                                                       |
| Settings          | WXT `storage` (`storage.defineItem`) in `src/shared/settings.ts`, `sync:` area, versioned with migrations                                                                                                                                                                                                                                                                   | Migrates HaramBlur 0.2.x `hb-settings`                                                                      |
| State machines    | XState 5 (stable) for the ML engine lifecycle and the video sampler                                                                                                                                                                                                                                                                                                         | Not per image: element state stays a `data-hb-state` attribute so feeds with thousands of images stay cheap |
| ML runtime        | `onnxruntime-web` `1.31.x`; WebGPU EP when `navigator.gpu` gives an adapter, else WASM (SIMD, threads when cross-origin isolated)                                                                                                                                                                                                                                           | Runs in a dedicated Worker inside the engine host page                                                      |
| Engine host       | Chrome: offscreen document (`reasons: ["WORKERS"]`). Firefox: the background page                                                                                                                                                                                                                                                                                           | Same engine code; a small host adapter                                                                      |
| Detector          | NudeNet `320n.onnx`, from Hugging Face `deepghs/nudenet_onnx`, SHA-256 `c15d8273adad2d0a92f014cc69ab2d6c311a06777a55545f2c4eb46f51911f0f` (12,150,158 bytes) pinned, fetched by `pnpm models:fetch` into `public/models/` (git-ignored)                                                                                                                                     | AGPL-3.0, compatible with this repo                                                                         |
| Face detector     | OpenCV YuNet `face_detection_yunet_2023mar.onnx` (Apache-2.0) — one letterboxed pass with 5 landmarks; replaces MediaPipe BlazeFace which missed small/group faces                                                                                                                                                                                                          | Landmark-aligned crops feed the gender model                                                                |
| Gender / age      | FastFace `fastface-small-112.onnx` (Apache-2.0) on eye-aligned crops; below the confidence gate the face collapses to neutral `FACE_PERSON` instead of trusting NudeNet's gendered label; `age` rides the `Detection` for `skipChildren`                                                                                                                                    | Replaces NudeNet's weak `FACE_*` gender guesses                                                             |
| Whole-image NSFW  | nsfwjs MobileNetV2 uint8 (MIT weights, ONNX port) at 224×224 → `NSFW_EXPLICIT` / `NSFW_SUGGESTIVE` full-frame detections; whole-pass only, skipped when the pass already blurs unless region blur is on (an NSFW verdict still escalates to whole-element)                                                                                                                  | Covers bikinis/suggestive/anime that region labels miss                                                     |
| Image pixels      | The engine fetches `currentSrc` itself (extension host permissions avoid CORS). `data:` and `blob:` URLs: the content script draws to a canvas and sends a downscaled JPEG data URL                                                                                                                                                                                         | Revisit later: transferable `ImageBitmap` through an extension iframe                                       |
| Video frames      | `requestVideoFrameCallback` (rAF fallback); a downscaled frame capped at the active detector input edge is sent as WebP q0.92 (PNG fallback); continuous one-in-flight sampling (10 fps ceiling; exponential backoff toward 1 fps for static clear scenes); pause while hidden, paused or off-screen; hysteresis so the blur doesn't flicker                                |                                                                                                             |
| Blur              | CSS `filter: blur(var(--hb-blur)) grayscale(...)` on the element; reveal on hover (delay) or click, per user setting. Region blur (`regionBlur`, default on): one `#hb-region-portal` div on `<html>` with an open shadow root tracks clipped viewport rects and paints `backdrop-filter` boxes — safe for React-managed parents (X.com), background images, and transforms | Whole-element blur remains the fallback for full-frame detections                                           |
| Testing           | Vitest (unit), Playwright with the built unpacked extension in Chromium (E2E), eval script that scores the engine on a labelled fixture set                                                                                                                                                                                                                                 | CI: GitHub Actions                                                                                          |

### Not in v1

Safari, Chrome built-in AI, WebNN, server-side anything, Effect 4 (too new for the team's tooling; revisit when the codebase is stable).

## Architecture

```
page (all frames)                     extension
┌──────────────────────────┐  typed   ┌──────────────────────┐        ┌─────────────────────────────┐
│ content script           │ messages │ background (SW / page)│ ensure │ engine host                 │
│  discovery (Mutation +   │ ───────▶ │  routes classify →    │ ─────▶ │  (offscreen doc / FF bg)    │
│   Intersection observers)│          │  engine; settings;    │        │   Worker: onnxruntime-web   │
│  queue (visible first)   │ ◀─────── │  context menu; badge  │ ◀───── │   NudeNet + YuNet +        │
│  blur / reveal CSS       │ verdict  └──────────────────────┘        │   FastFace + nsfwjs        │
│  region portal (shadow)  │                                          │   policy → verdict, cache   │
│  video sampler (XState)  │                                          └─────────────────────────────┘
└──────────────────────────┘
popup / options (Svelte) ── settings storage ──▶ all contexts listen for changes
```

### Message protocol (`src/shared/protocol.ts`)

-   `classifyImage({ url, width, height, frameUrl }) → Verdict`
-   `classifyFrame({ dataUrl, videoId, ts }) → Verdict`
-   `engineStatus() → { state: "booting" | "ready" | "degraded" | "failed", backend: "webgpu" | "wasm" | null, error?: string }`
-   `Verdict = { decision: "blur" | "clear", reasons: Detection[], ms: number, cached: boolean }`
-   `Detection = { label: DetectionLabel, score: number, box: [x, y, w, h], age?: number }` (box normalized 0..1; `age` only on face detections)

Requests made while the engine is booting wait (bounded, default 30 s) instead of failing. On timeout or error the content script applies the user's fail mode (default: keep blurred, show a small "couldn't check" badge on hover).

### Policy (`src/shared/policy.ts`, pure function, unit-tested)

Input: detections + settings. Output: `blur` or `clear`. Settings map to per-class thresholds:

-   `blurFemale` → `FACE_FEMALE` (YuNet+FastFace), `FEMALE_BREAST_*`, `FEMALE_GENITALIA_*` …
-   `blurMale` → `FACE_MALE`, `MALE_BREAST_EXPOSED`, `MALE_GENITALIA_EXPOSED`
-   `strictModesty` (default off) → `MODESTY_LABELS`: belly/armpits/feet covered classes — high false-positive rate on clothed photos, so opt-in.
-   `skipChildren` → `FACE_*` detections with `age < 15` don't blur on their own (never suppresses `CORE_EXPOSED_LABELS`).
-   `NSFW_EXPLICIT`/`NSFW_SUGGESTIVE` always blur regardless of gender toggles; threshold lerps with `strictness`.
-   Per-class threshold curves (DP4): each blur-capable label has `{max, min, pow}` with `threshold = max − (max−min)·strictness^pow`. Genitalia/noisy-exposed use pow 2–3 so mid-strictness precision is preserved (dev-split negatives cluster at 0.36–0.52) while strictness 1 drops toward ~0.1–0.15 (upstream parity); the mapping is continuous, so existing strictness values need no migration.
-   Exposed nudity of any person is always blurred unless the extension is off for the site.

## Repository layout

```
src/
  entrypoints/        background.ts, content/, offscreen/, popup/, options/
  engine/             worker.ts, nudenet.ts, yunet.ts, fastface.ts, nsfw.ts (pre/post-processing, NMS), backend.ts, cache.ts, lifecycle.ts
  content/            discovery.ts, queue.ts, blur.ts, region.ts (shadow-DOM portal), site.ts, video/ (manager, sampler, frames)
  shared/             protocol.ts, settings.ts, policy.ts, labels.ts, site.ts (allowlist), i18n
  ui/                 Svelte components
public/models/        fetched by scripts/fetch-models.ts (git-ignored)
scripts/              fetch-models.ts, eval.ts
tests/                unit/, e2e/, fixtures/
docs/                 PLAN.md, ARCHITECTURE.md, DECISIONS.md
```

## Work packages

Each work package is one PR into `main`. Packages in the same wave run in parallel and own separate directories.

| Wave | Package                          | Owns                                                                                                    | Done when                                                                                                                                                                          |
| ---- | -------------------------------- | ------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1    | **WP0 Scaffold**                 | everything at the top level, `src/shared/*`, entrypoint stubs, CI                                       | `pnpm i && pnpm models:fetch && pnpm check && pnpm test && pnpm build` pass in CI; the unpacked extension loads in Chromium; old JS sources removed                                |
| 2    | **WP1 ML engine**                | `src/engine/`, `src/entrypoints/offscreen/`, engine routing in background                               | NudeNet runs in the offscreen worker with WebGPU or WASM; boot handshake; requests queued during boot; LRU verdict cache; benchmark script prints load and per-image ms            |
| 2    | **WP2 Content pipeline**         | `src/content/` (not `video/`), `src/entrypoints/content/`                                               | fail-closed blur, `currentSrc`/srcset/picture/background/poster/iframes, visible-first queue, reveal on hover/click, per-site off switch, settings apply live                      |
| 2    | **WP3 Video**                    | `src/content/video/`                                                                                    | rVFC sampler XState machine, adaptive rate, hysteresis, pause rules, unit tests with fake clock                                                                                    |
| 2    | **WP4 UI**                       | `src/entrypoints/popup/`, `src/entrypoints/options/`, `src/ui/`, `src/shared/i18n/`, `public/_locales/` | popup (on/off, this site, quick toggles), options page (all settings, allowlist, about/licenses), en/ar/tr/ur translations carried over, RTL, context menu "reveal this"           |
| 3    | **WP5 Integration + E2E + eval** | `tests/e2e/`, `scripts/eval.ts`, fixes across packages                                                  | Playwright E2E covers: cold start, srcset/picture, lazy-loaded, background image, iframe, video, reveal, site allowlist; eval prints precision/recall per class on the fixture set |
| 3    | **WP6 Firefox + release**        | WXT browser config, `docs/`, README                                                                     | `wxt build -b firefox` works with the engine in the background page; `pnpm zip` produces store zips for Chrome and Firefox; README rewritten                                       |

### Phase 2 — accuracy & pipeline hardening (WP-A…WP-F)

Follow-up packages after v1 acceptance, each one PR:

-   **WP-A policy + engine host**: label groups (`CORE_EXPOSED` always-on, `MODESTY` opt-in via `strictModesty`), 0.55 floor on genitalia classes, verdict cache stores detections so settings re-decide cached URLs, `skipChildren` plumbing.
-   **WP-B benchmark**: `manifest.v2.json` dev/test split + `scripts/eval-compare.ts` (per-split metrics, `--baseline` head-to-head).
-   **WP-C face pipeline**: YuNet whole-image pass → landmark-aligned FastFace gender/age → synthetic `FACE_*` detections with `age`; NudeNet `FACE_*` suppressed on covered boxes.
-   **WP-D whole-image NSFW**: bundled nsfwjs classifier → `NSFW_EXPLICIT`/`NSFW_SUGGESTIVE`, whole-pass only, skipped when already blurred unless region blur can still escalate.
-   **WP-E region blur + pipeline fixes**: `#hb-region-portal` shadow-DOM overlays (React-safe, bg-image/video/clip-aware), X.com paired-media skip, `blurryStartMode`/`failClosed` decoupled (settings v4), bounded queue LRU, unified video states, detector-sized WebP frames.
-   **WP-F UI + release**: popup uses shared `isAllowlisted` + `Active · <backend>` badge; options exposes `regionBlur`/`strictModesty`/`skipChildren`; corrected license/feedback links; Chrome MV3 + Firefox MV2 packages verified.

## Acceptance for v1

-   E2E suite green in CI on Chromium.
-   No regressions against the old extension's verified behaviours: detection after warm-up, late-inserted images, hover reveal.
-   Fixed: cold-start images stuck in processing; srcset/picture classified by the wrong file; media visible while checking.
-   Engine: model load < 2 s and < 150 ms per image on WASM on a 4-core laptop CPU.
-   Zero network requests except fetching the page's own media.
