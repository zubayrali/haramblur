Repository: github.com/zubayrali/haramblur. Read `docs/PLAN.md`, `AGENTS.md` and `src/shared/*` on `main` first — source of truth.

Your task: **WP2 content pipeline**. Open one PR into `main`. You own `src/content/` EXCEPT `src/content/video/` (another session owns it; share the queue through `src/content/queue.ts` exports only). Do not touch `src/engine/` or `src/entrypoints/offscreen/`.

Deliver:

1. `discovery.ts`: MutationObserver (childList + attributes `src`, `srcset`, `poster`, `style`, `href`, `data-src`), an IntersectionObserver marking visibility, and initial DOM scan. Find: `img` (use `currentSrc`), `picture`/`source`, elements whose computed `background-image` is a URL, `video` (element + `poster`), `iframe` coverage comes free because the content script runs in all frames. Ignore data: URLs under ~1 KB, tracking pixels under ~24 px, SVG icons.
2. `blur.ts`: a `<style>` that blurs elements with `[data-hb-blur]` using `filter: blur(var(--hb-blur))` + optional grayscale, plus a `.hb-loading` state (fail closed: media is blurred until a verdict arrives — keep it blurred on verdict error/timeout unless `settings.failOpen`). Reveal on hover after `settings.unblurDelay` or on click per settings; never reveal while the extension is on if the verdict is blur and the user disabled reveal.
3. `queue.ts`: classify visible media first, then rest; per-tab concurrency cap 4; per-URL dedup (shared with engine cache); retry once on transient errors.
4. Wiring: `classifyImage`/`classifyFrame` via `src/shared/protocol.ts`; wait-and-retry while `engineStatus` is booting. Every verdict sets `data-hb-state` + `data-hb-reasons` (JSON).
5. Per-site off: hostname/domain allowlist from `settings`, checked at init and on settings change; `chrome.storage` change events apply live without refresh (update `data-hb-blur` vars, re-run policy for stored detections where possible).
6. `src/content/index.ts` glue. The content script must not run `import` of engine code; keep it small.

Unit tests: discovery filters, queue priority/dedup, policy application. Acceptance: CI green; verified in Chromium on a local test page containing the cases listed in PLAN.md acceptance. Do not merge.
