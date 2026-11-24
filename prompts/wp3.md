Repository: github.com/zubayrali/haramblur. Read `docs/PLAN.md`, `docs/ENGINE_NOTES.md`, `AGENTS.md` and `src/shared/*` on `main` first.

Your task: **WP3 video detection**. Open one PR into `main`. You own `src/content/video/` only. Integrate via `src/content/queue.ts`/`protocol.ts` exports already on `main`; do not modify files outside `src/content/video/` unless they are shared types (explain any change in the PR).

Deliver:

1. `sampler.ts`: an XState 5 machine per playing video: `idle → playing → sampling → (paused | hidden | offscreen | ended)`. Use `requestVideoFrameCallback` (fall back to `requestAnimationFrame`). No work while `video.paused/ended`, `document.hidden`, or the element is out of view (IntersectionObserver).
2. `frames.ts`: draw the current frame onto an OffscreenCanvas scaled to at most 320 px on the long edge, export a JPEG data URL (quality ~0.7), call `classifyFrame({dataUrl, videoId, ts})`. Cap frame size and skip duplicate frames by a cheap pixel-hash or frame counter.
3. Adaptive rate: 2 fps baseline; up to 6 fps for ~5 s after a scene change (large pixel diff); back off when verdicts are stable. Max one in-flight request per video, max 4 videos sampled at once.
4. Hysteresis in `policy` usage: require 2 consecutive agreeing verdicts (or 1 high-confidence) before flipping blur on/off on a video, so it does not flicker. Store last verdicts per video.
5. Per-video reveal respects the same settings as images.

Unit tests with a fake clock and fake video element (Vitest + `@vitest-environment` or hand-rolled mocks) covering the state transitions, pause rules and hysteresis. Acceptance: CI green; on a real video page the sampler runs in Chromium and returns verdicts. Do not merge.
