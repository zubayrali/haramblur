# Architecture

Data flows page → content script → background → engine host → worker → verdict
→ back. Everything ships inside the extension: the model, the WASM runtime,
and every line of code. Nothing leaves the device.

```
page (all frames)                        extension
┌────────────────────────┐   typed     ┌───────────────────┐      ┌─────────────────────────┐
│ content script         │  messages   │ background         │      │ engine host             │
│  discovery             │ ──────────▶ │  (MV3 service      │      │  Chrome: offscreen doc  │
│  (MutationObserver +   │  classify   │   worker /         │      │  Firefox: background    │
│   IntersectionObserver)│  requests   │   MV2 page)        │      │  page itself            │
│  priority queue        │             │  routes classify → │      │   │                     │
│  blur / reveal CSS     │ ◀────────── │  engine; settings; │      │   ▼ Worker              │
│  video sampler         │   verdict   │  context menu;     │      │    onnxruntime-web      │
│                        │             │  tab stats; badge  │ ◀─── │    NudeNet 320n →       │
└────────────────────────┘             └───────────────────┘      │    policy → verdict     │
                                                                   └─────────────────────────┘
popup / options (Svelte) ── storage.sync settings ──▶ every context listens for changes
```

## Content script (`src/entrypoints/content`, `src/content/`)

Runs at `document_start` in every frame (`http(s)://*`). A MutationObserver
discovers `<img>`, `<video>`, CSS background images, `srcset` and `<picture>`
sources, and `<iframe>` contents; an IntersectionObserver promotes visible
media to the front of the classification queue. Element state is carried in
`data-hb-*` attributes (`kind`, `state`, `blur`, `reasons`, `reveal`) so
thousand-image feeds stay cheap. The video sampler (`requestVideoFrameCallback`,
rAF fallback) draws downscaled ≤480 px WebP frames and
pauses for hidden, paused, or off-screen videos. Sampling is a continuous
one-in-flight loop with a 10 fps ceiling and exponential backoff toward 1 fps
for static clear scenes.

## Background (`src/entrypoints/background.ts`)

Routes `classifyImage`/`classifyFrame` calls from content scripts to the
engine host via the typed protocol (`src/shared/protocol.ts`,
`@webext-core/messaging`), owns settings broadcast, the context menu, and
per-tab stats for the popup. A small adapter picks the host implementation:
`browser.offscreen == null` → in-process Firefox host; otherwise the Chrome
offscreen document host.

## Engine host (`src/engine/host.ts`, Chrome's `src/entrypoints/offscreen/`)

Creates and supervises the inference worker: bounded boot (30 s), bounded
inference calls (30 s), a 500-entry verdict cache, and media decoding —
same-origin URLs fetch `ImageBitmap`s directly (host permissions avoid CORS);
`data:`/`blob:` frames are decoded from the data URL the content script
painted.

## Worker (`src/engine/worker.ts`, `src/engine/`)

A module Worker running onnxruntime-web: WebGPU EP when `navigator.gpu` yields
an adapter, else WASM SIMD (+ threads under cross-origin isolation). The
bundled `ort.min.mjs` build resolves its WASM through `wasmPaths` → `/ort/` —
both the runtime and the glue modules ship in the package; the NudeNet
`320n.onnx` model ships in `/models/` (SHA-256 pinned at fetch time). Output
boxes pass through NMS into `policy.ts` — a pure function mapping detections +
settings to `blur`/`clear`.

## Fail closed

New media is **blurred before a verdict exists** — discovery tags it
`pending` + blurred at discovery time, never waiting on the engine. Requests
during engine boot queue and wait (bounded). On timeout or error the content
script applies `failClosed` (default: stay blurred). The only `clear` path is
a real verdict — engine failure never unblurs.
