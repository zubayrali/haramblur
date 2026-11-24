Repository: github.com/zubayrali/haramblur. Read `docs/PLAN.md` and `docs/ENGINE_NOTES.md` on `main` first — they are the source of truth. Also read the scaffold's `src/shared/protocol.ts`, `src/shared/labels.ts`, `src/shared/settings.ts`, `src/shared/policy.ts` on `main` and build on them exactly.

Your task: **WP1 ML engine** from `docs/PLAN.md`. Open one PR into `main`. You own `src/engine/` and `src/entrypoints/offscreen/` plus the engine routing in `src/entrypoints/background.ts`. Do not touch `src/content/` or the popup/options entrypoints.

Deliver:

1. `src/engine/`: a dedicated `Worker` inside the engine host that loads onnxruntime-web 1.31.x, picks WebGPU when `navigator.gpu.requestAdapter()` succeeds, else WASM (SIMD; threads only when `crossOriginIsolated`). Set `ort.env.wasm.wasmPaths` to an absolute extension URL. Load `public/models/320n.onnx` (run `pnpm models:fetch` first locally).
2. `nudenet.ts`: letterbox preprocessing and YOLOv8 output parsing exactly per `docs/ENGINE_NOTES.md` (top-left letterbox, fill 114, class order given there), per-class NMS at IoU 0.45, boxes normalized 0..1.
3. `lifecycle.ts`: an XState 5 machine for the engine: booting → ready | degraded (WASM fallback) | failed. The host answers `engineStatus` from it. Requests arriving while booting wait; add a bounded wait (30 s) then answer `{error}` so the caller can apply fail-closed. A dead or failed worker restarts once, then stays failed.
4. `cache.ts`: LRU verdict cache keyed by image URL (cap 500).
5. Background: `classifyImage({url, width, height, frameUrl})` — fetch the URL in the engine host (`fetch` + `createImageBitmap`; extension host permissions mean no CORS block), or decode `frameUrl` data URL, then run NudeNet + `policy.decide` → `Verdict`. `classifyFrame({dataUrl, videoId, ts})` same path without the cache fetch.
6. The engine host runs as an offscreen document on Chrome; on Firefox the same code runs in the background page behind a host adapter. `browser.offscreen` may not exist on Firefox — the adapter picks the host.
7. `scripts/bench-engine.ts` or an E2E-side benchmark: load the model on WASM on `tests/fixtures/` images, print load ms and per-image ms, verify < 2 s load and < 150 ms/image on the CI machine.
8. Vitest unit tests for preprocessing, output parsing, NMS and the policy wiring.

Acceptance: CI green; you verified the engine in Chromium end to end (load the built extension, call `classifyImage` on fixture images, report real timings in the PR). Do not merge.
