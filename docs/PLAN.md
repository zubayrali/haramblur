# HaramBlur v1 rewrite plan

This repository is a fork of [alganzory/HaramBlur](https://github.com/alganzory/HaramBlur) (AGPL-3.0). Its full history is kept. The rewrite replaces the code; it stays AGPL-3.0 and credits the original authors.

## Goals

1. **No media slips through.** New images and videos are blurred until they have been classified (fail closed, user-configurable). Media seen before the models finish loading is queued, never dropped.
2. **Better detection.** Replace nsfwjs (5 whole-image classes) and Human (CDN-loaded face/gender) with one bundled region detector: NudeNet 320n (YOLOv8n, ONNX). It returns boxes for 18 classes, including `FACE_FEMALE`, `FACE_MALE`, and covered and exposed body parts.
3. **Blur what the user sees.** Classify `currentSrc` (srcset and `<picture>`), CSS background images, `poster`, and media inside iframes.
4. **Private and offline.** Every model ships inside the extension. No CDN, no telemetry, no remote inference.
5. **Cross-browser from one codebase.** Chrome/Edge (MV3) and Firefox from the same source with WXT.
6. **Measured, tested, automated.** Unit tests, extension E2E tests, an accuracy eval harness, and CI on every PR.

## Decisions

| Area              | Decision                                                                                                                                                                                                                                                       | Notes                                                                                                       |
| ----------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------- |
| Build / framework | WXT `0.21.x`, Vite, pnpm, Node 22 LTS                                                                                                                                                                                                                          | `wxt build -b chrome`, `wxt build -b firefox`, `wxt zip`                                                    |
| Language          | TypeScript, `strict: true`                                                                                                                                                                                                                                     | No `any` without a comment saying why                                                                       |
| Lint / format     | Biome                                                                                                                                                                                                                                                          | One tool; replaces Husky + Prettier                                                                         |
| UI                | Svelte 5 + Tailwind CSS 4                                                                                                                                                                                                                                      | Popup and options page; RTL-aware                                                                           |
| Messaging         | `@webext-core/messaging` (typed protocol in `src/shared/protocol.ts`)                                                                                                                                                                                          | All cross-context calls go through it                                                                       |
| Settings          | WXT `storage` (`storage.defineItem`) in `src/shared/settings.ts`, `sync:` area, versioned with migrations                                                                                                                                                      | Migrates HaramBlur 0.2.x `hb-settings`                                                                      |
| State machines    | XState 5 (stable) for the ML engine lifecycle and the video sampler                                                                                                                                                                                            | Not per image: element state stays a `data-hb-state` attribute so feeds with thousands of images stay cheap |
| ML runtime        | `onnxruntime-web` `1.31.x`; WebGPU EP when `navigator.gpu` gives an adapter, else WASM (SIMD, threads when cross-origin isolated)                                                                                                                              | Runs in a dedicated Worker inside the engine host page                                                      |
| Engine host       | Chrome: offscreen document (`reasons: ["WORKERS"]`). Firefox: the background page                                                                                                                                                                              | Same engine code; a small host adapter                                                                      |
| Detector          | NudeNet `320n.onnx`, from Hugging Face `deepghs/nudenet_onnx`, SHA-256 `c15d8273adad2d0a92f014cc69ab2d6c311a06777a55545f2c4eb46f51911f0f` (12,150,158 bytes) pinned, fetched by `pnpm models:fetch` into `public/models/` (git-ignored)                        | AGPL-3.0, compatible with this repo                                                                         |
| Face helper       | MediaPipe BlazeFace short-range (optional, behind a setting)                                                                                                                                                                                                   | Exact face boxes for region blur later                                                                      |
| Image pixels      | The engine fetches `currentSrc` itself (extension host permissions avoid CORS). `data:` and `blob:` URLs: the content script draws to a canvas and sends a downscaled JPEG data URL                                                                            | Revisit later: transferable `ImageBitmap` through an extension iframe                                       |
| Video frames      | `requestVideoFrameCallback` (rAF fallback); a downscaled frame of at most 320 px is sent as a JPEG data URL; adaptive rate (default 2 fps, up to 6 fps after a scene change); pause while hidden, paused or off-screen; hysteresis so the blur doesn't flicker |                                                                                                             |
| Blur              | CSS `filter: blur(var(--hb-blur)) grayscale(...)` on the element; reveal on hover (delay) or click, per user setting                                                                                                                                           | Region blur is v1.1                                                                                         |
| Testing           | Vitest (unit), Playwright with the built unpacked extension in Chromium (E2E), eval script that scores the engine on a labelled fixture set                                                                                                                    | CI: GitHub Actions                                                                                          |

### Not in v1

Region-only blur, Safari, Chrome built-in AI, WebNN, server-side anything, Effect 4 (too new for the team's tooling; revisit when the codebase is stable).

## Architecture

```
page (all frames)                     extension
┌──────────────────────────┐  typed   ┌──────────────────────┐        ┌─────────────────────────────┐
│ content script           │ messages │ background (SW / page)│ ensure │ engine host                 │
│  discovery (Mutation +   │ ───────▶ │  routes classify →    │ ─────▶ │  (offscreen doc / FF bg)    │
│   Intersection observers)│          │  engine; settings;    │        │   Worker: onnxruntime-web   │
│  queue (visible first)   │ ◀─────── │  context menu; badge  │ ◀───── │   NudeNet + BlazeFace       │
│  blur / reveal CSS       │ verdict  └──────────────────────┘        │   policy → verdict, cache   │
│  video sampler (XState)  │                                          └─────────────────────────────┘
└──────────────────────────┘
popup / options (Svelte) ── settings storage ──▶ all contexts listen for changes
```

### Message protocol (`src/shared/protocol.ts`)

-   `classifyImage({ url, width, height, frameUrl }) → Verdict`
-   `classifyFrame({ dataUrl, videoId, ts }) → Verdict`
-   `engineStatus() → { state: "booting" | "ready" | "degraded" | "failed", backend: "webgpu" | "wasm" | null, error?: string }`
-   `Verdict = { decision: "blur" | "clear", reasons: Detection[], ms: number, cached: boolean }`
-   `Detection = { label: NudeNetLabel, score: number, box: [x, y, w, h] }` (box normalized 0..1)

Requests made while the engine is booting wait (bounded, default 30 s) instead of failing. On timeout or error the content script applies the user's fail mode (default: keep blurred, show a small "couldn't check" badge on hover).

### Policy (`src/shared/policy.ts`, pure function, unit-tested)

Input: detections + settings. Output: `blur` or `clear`. Settings map to per-class thresholds:

-   `blurFemale` → `FACE_FEMALE`, `FEMALE_BREAST_*`, `FEMALE_GENITALIA_*` …
-   `blurMale` → `FACE_MALE`, `MALE_BREAST_EXPOSED`, `MALE_GENITALIA_EXPOSED`
-   Exposed nudity of any person is always blurred unless the extension is off for the site.
-   `strictness` 0..1 → threshold = lerp(0.6, 0.2, strictness) for faces; exposed nudity uses a lower threshold.

## Repository layout

```
src/
  entrypoints/        background.ts, content/, offscreen/, popup/, options/
  engine/             worker.ts, nudenet.ts (pre/post-processing, NMS), backend.ts, cache.ts, lifecycle.ts
  content/            discovery.ts, queue.ts, blur.ts, video/, frames.ts
  shared/             protocol.ts, settings.ts, policy.ts, labels.ts, i18n
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

## Acceptance for v1

-   E2E suite green in CI on Chromium.
-   No regressions against the old extension's verified behaviours: detection after warm-up, late-inserted images, hover reveal.
-   Fixed: cold-start images stuck in processing; srcset/picture classified by the wrong file; media visible while checking.
-   Engine: model load < 2 s and < 150 ms per image on WASM on a 4-core laptop CPU.
-   Zero network requests except fetching the page's own media.
