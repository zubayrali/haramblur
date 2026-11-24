Repository: github.com/zubayrali/haramblur. Read `docs/PLAN.md`, `docs/ENGINE_NOTES.md`, `AGENTS.md` on `main` first. Your job is **WP7: detection calibration**. Open one PR into `main`.

Context: the v1 rewrite is merged and released. The eval harness (`pnpm eval`, tests/fixtures/) reports 4/5 decision accuracy: `woman-camera.jpg` is expected `blur` but scores `FACE_FEMALE@0.18`, below the face threshold (lerp 0.6→0.2 across strictness; ~0.4 at default 0.5).

Goal: raise recall on real female-face photos WITHOUT regressing the clear cases or spiking false positives.

Step 1 — expand the fixture set first (don't tune on 5 images):
- Add ~10-15 more fixtures: several photos of women at varied resolutions and face sizes, a few clearly-explicit images, and hard negatives (men, animals, objects, cartoons/drawings, small thumbnails <320px, tall/wide aspect ratios).
- Generate or source them locally (public-domain/CC0 images, or generated); keep them small (<500KB), update tests/fixtures/manifest.json with expected labels + decisions.
- Report per-fixture detections in the PR.

Step 2 — diagnose the miss: is it (a) preprocessing (letterbox to 320 crushes detail on large/wide photos), (b) threshold too high, or (c) model limit? Instrument scripts/eval.ts to print raw detections + input scale if needed.

Step 3 — pick the fix (prefer in this order):
- Preprocessing improvements: e.g. upscaling small inputs, center-crop strategies, multi-scale pass only when the 320 pass finds nothing near threshold (keep it cheap — images should stay ~85-120ms on WASM).
- Threshold/strictness remapping if the score distribution supports it (justify with the expanded fixture data — face threshold must still reject hard negatives).
- A larger NudeNet variant ONLY if the 320n is proven inadequate by the fixtures — a bigger model must stay bundled and you must justify the size cost; do NOT silently ship a 50MB model.
- Consider also checking covered-class detections (FEMALE_BREAST_COVERED etc.) which may carry signal the current policy ignores.

Constraints: no runtime CDN, no new heavyweight deps; unit tests for any preprocessing/policy change; `pnpm check`/`test`/`build`/`test:e2e` green; run `pnpm eval` before and after and put both tables in the PR description. If the woman-camera miss turns out to be a true model limitation at 320n, say so honestly rather than gaming the eval. Do not merge.
