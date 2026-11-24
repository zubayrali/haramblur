# HaramBlur

On-device image and video blurring browser extension. Automatically blurs
images and videos containing unwanted or impermissible content — all
classification runs locally inside the extension. No CDN, no telemetry, no
remote inference.

This is an AGPL-3.0 rewrite of [alganzory/HaramBlur](https://github.com/alganzory/HaramBlur),
crediting the original authors. The rewrite plan and decisions live in
[`docs/PLAN.md`](docs/PLAN.md); engine research in
[`docs/ENGINE_NOTES.md`](docs/ENGINE_NOTES.md).

## Stack

-   WXT 0.21.x + Vite, pnpm, Node 22
-   TypeScript (strict)
-   Svelte 5 + Tailwind CSS 4 (popup / options)
-   onnxruntime-web 1.31.x + NudeNet `320n.onnx` (bundled, fetched at build time)
-   Biome (lint + format), Vitest (unit), Playwright (E2E), GitHub Actions CI

## Development

```sh
pnpm install
pnpm models:fetch      # downloads + verifies the bundled model into public/models/
pnpm dev               # Chrome dev server
pnpm dev:firefox       # Firefox dev server
pnpm build             # Chrome MV3 → .output/chrome-mv3
pnpm build:firefox     # Firefox → .output/firefox-mv3
pnpm zip               # store zips
pnpm check             # biome + tsc --noEmit + svelte-check
pnpm test              # Vitest unit tests
pnpm test:e2e          # Playwright E2E (needs `pnpm build` first; xvfb-run headless)
```

The unpacked Chrome build loads via `chrome://extensions` → "Load unpacked" →
`.output/chrome-mv3`.

## Layout

See `docs/PLAN.md` "Repository layout" and "Work packages" — directories are
owned by their work package; top-level, `src/shared/`, entrypoint stubs and CI
are WP0.

## License

[AGPL-3.0](LICENSE), same as the original project.
