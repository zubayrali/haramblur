# `src/content/` — WP2 content pipeline

Owned by **WP2 Content pipeline** (docs/PLAN.md). `video/` belongs to WP3.

- `discovery.ts` — MutationObserver (childList + media-URL attributes),
  IntersectionObserver visibility tracking, initial DOM scan; emits
  `MediaCandidate`s for `img`/`video`/CSS background images.
- `queue.ts` — `TaskQueue`: visible-first, concurrency-capped (4), per-key
  dedup, single retry. WP3 submits video frames through these exports.
- `blur.ts` — injected `[data-hb-blur]` stylesheet, `data-hb-*` element state,
  reveal rules.
- `engine.ts` — `classifyImage`/`engineStatus` client over
  `src/shared/protocol.ts`; waits (bounded 30 s) while the engine boots.
- `frames.ts` — canvas capture → JPEG `frameUrl` for `data:`/`blob:` media and
  posterless videos.
- `site.ts` — allowlist matching, `data-hb-reasons` JSON, policy re-decide.
- `index.ts` — glue: settings watch, discovery → queue → verdict wiring.
