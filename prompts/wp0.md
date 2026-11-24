Repository: github.com/zubayrali/haramblur (a fork of alganzory/HaramBlur, AGPL-3.0). Read `docs/PLAN.md` and `docs/ENGINE_NOTES.md` on `main` first; they are the source of truth for decisions. Do not change decisions in them; if something in them is impossible, explain why in the PR description and pick the closest option.

Your task: **WP0 Scaffold** from `docs/PLAN.md`. Open one PR into `main`.

Deliver:

1. Replace the old plain-JS extension with a WXT 0.21.x + TypeScript (strict) + Svelte 5 + Tailwind 4 project managed by pnpm (Node 22). Delete the old `src/`, `tfjs/`, `dist/`, `manifest.json`, `vite.config.js`, Husky and Prettier config. Keep `LICENSE`. Keep the existing translations by moving their strings into `public/_locales/{en,ar,tr,ur}/messages.json` (WXT i18n).
2. Biome for lint and format. Scripts: `dev`, `dev:firefox`, `build`, `build:firefox`, `zip`, `zip:firefox`, `check` (biome + `tsc --noEmit`/`svelte-check`), `test` (Vitest), `test:e2e` (Playwright), `models:fetch`.
3. `scripts/fetch-models.ts`: downloads NudeNet `320n.onnx` from `https://huggingface.co/deepghs/nudenet_onnx/resolve/main/320n.onnx` into `public/models/`, verifies the SHA-256 in `docs/PLAN.md`, skips when already present and valid. `public/models/` is git-ignored. Copy the onnxruntime-web WASM/mjs files it needs into the build output (e.g. a WXT hook or `public/ort/`, git-ignored).
4. `src/shared/`: `protocol.ts` (the typed `@webext-core/messaging` protocol exactly as in PLAN.md "Message protocol"), `labels.ts` (the 18 NudeNet labels in model order, plus groups: female, male, exposed, covered), `settings.ts` (WXT `storage.defineItem` with versioned defaults and a migration from HaramBlur 0.2.x: it stored a `hb-settings` object in `chrome.storage.sync` with keys like `status`, `blurryStartMode`, `blurAmount`, `gray`, `strictness`, `blurImages`, `blurVideos`, `blurMale`, `blurFemale`, `unblurImages`, `unblurVideos`, `whitelist`; read the old repo history for the exact shape), and `policy.ts` (pure `decide(detections, settings) → "blur" | "clear"` per PLAN.md "Policy") with Vitest unit tests.
5. Entrypoint stubs that build and load: `background.ts`, `content/index.ts` (all frames, `document_start`), `offscreen/` (Chrome; creates a Worker), `popup/`, `options/`. Stubs may be minimal, but the wiring (offscreen creation with `reasons: ["WORKERS"]`, a `ping` message round trip) must work. Firefox build must also succeed.
6. Empty owned directories with a short README each: `src/engine/`, `src/content/`, `src/content/video/`, `src/ui/`, `tests/e2e/`, `tests/fixtures/`.
7. Playwright config that loads the built unpacked Chrome extension (persistent context, `--disable-extensions-except`, `--load-extension`) and one smoke E2E test: the extension loads and the background answers `ping`.
8. GitHub Actions CI on PRs and `main`: install, `models:fetch` (cache by SHA), `check`, `test`, `build`, `build:firefox`, `test:e2e` under xvfb.
9. README: what this is, credit to alganzory/HaramBlur, dev setup, license.

Acceptance: CI green on the PR, and you have loaded the built extension in Chromium yourself and seen the ping round trip. Do not merge the PR. When done, report the PR URL and anything you could not do.
