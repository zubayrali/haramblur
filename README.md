# HaramBlur

A browser extension that blurs images and videos containing unwanted or
impermissible content — **entirely on-device**. Pages are scanned as they load
and as they scroll; media matching the detection policy is blurred before you
see it, and can be revealed on click or hover if you choose.

This is a WXT + TypeScript rewrite of
[alganzory/HaramBlur](https://github.com/alganzory/HaramBlur), crediting the
original authors. Same license: AGPL-3.0.

## Features

-   **Fail closed.** New media is blurred until classified — nothing slips
    through while the engine is booting.
-   **Region detection, not whole-image tags.** One bundled model (NudeNet
    `320n.onnx`, YOLOv8n, ONNX) returns boxes for 18 classes: faces by gender,
    covered and exposed body parts. `FACE_FEMALE` blurs under default settings;
    exposed nudity of any person always blurs.
-   **What you actually see.** Classification uses `currentSrc`, so `srcset`
    and `<picture>` are resolved correctly; CSS background images, video
    `poster` frames, and media inside iframes are covered too.
-   **Videos.** A sampler sends downscaled frames (~320 px) at an adaptive
    rate — faster after scene changes, paused when the video is hidden,
    off-screen, or paused — with hysteresis so the blur doesn't flicker.
-   **Reveal controls.** Click (permanent) or hover (timed) reveal, per media
    type, both optional.
-   **Allowlist.** Disable detection per hostname.
-   **RTL + localization.** Arabic, English, Turkish, Urdu.

## Privacy

Everything runs inside the extension on your machine.

-   The model ships bundled — fetched once at build time, SHA-256 pinned.
-   No CDN, no remote inference, no telemetry, no analytics.
-   Image pixels are fetched by the extension itself (host permissions avoid
    CORS); nothing is uploaded anywhere.

## Install

### Chrome / Edge / Brave (MV3)

1. `pnpm install && pnpm models:fetch && pnpm build`
2. Open `chrome://extensions`, enable Developer mode, click **Load unpacked**,
   select `.output/chrome-mv3`.

### Firefox (MV2)

1. `pnpm install && pnpm models:fetch && pnpm build:firefox`
2. Open `about:debugging#/runtime/this-firefox`, click **Load Temporary
   Add-on…**, and pick `.output/firefox-mv2/manifest.json`.

   Temporary add-ons unload when Firefox restarts. For a persistent install,
   the extension must be signed — `pnpm zip:firefox` produces the upload zip
   for AMO signing.

Requires Firefox ≥ 114 (module workers).

## Settings

Popup: master toggle, per-tab stats, engine status. Options page:

-   **Media types** — blur images / blur videos independently.
-   **Classes** — blur female-presenting media (default on), blur
    male-presenting media (default off); exposed nudity always blurs.
-   **Strictness** — 0..1 confidence-threshold slider.
-   **Reveal** — click or hover unblur, per media type.
-   **Blur amount** — CSS blur strength, plus optional grayscale.
-   **Allowlist** — hostnames where detection is off.
-   **Fail mode** — keep blurred on classification timeout (default) or show.

## Development

```sh
pnpm install
pnpm models:fetch      # downloads + verifies the bundled model into public/models/
pnpm dev               # Chrome dev server
pnpm dev:firefox       # Firefox dev server
pnpm build             # Chrome MV3 → .output/chrome-mv3
pnpm build:firefox     # Firefox MV2 → .output/firefox-mv2
pnpm zip               # Chrome store zip
pnpm zip:firefox       # Firefox store zip
pnpm check             # biome + tsc --noEmit + svelte-check
pnpm test              # Vitest unit tests
pnpm test:e2e          # Playwright E2E (needs `pnpm build` first; xvfb-run headless)
```

Stack: WXT 0.21.x + Vite, TypeScript strict, Svelte 5 + Tailwind CSS 4,
onnxruntime-web 1.31.x, Biome, Vitest, Playwright. Architecture:
[`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md); the rewrite plan and engine
notes live in [`docs/PLAN.md`](docs/PLAN.md) and
[`docs/ENGINE_NOTES.md`](docs/ENGINE_NOTES.md).

## License

[AGPL-3.0](LICENSE), same as the original project. Original work by
[alganzory/HaramBlur](https://github.com/alganzory/HaramBlur) and its
contributors — thank you.
