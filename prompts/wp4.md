Repository: github.com/zubayrali/haramblur. Read `docs/PLAN.md`, `AGENTS.md`, `src/shared/settings.ts` and `src/shared/protocol.ts` on `main` first.

Your task: **WP4 popup + options UI**. Open one PR into `main`. You own `src/entrypoints/popup/`, `src/entrypoints/options/`, `src/ui/`, `src/shared/i18n/` and `public/_locales/` plus the context-menu wiring in `src/entrypoints/background.ts`. Do not touch `src/engine/` or `src/content/` internals.

Deliver:

1. Popup (Svelte 5 + Tailwind): master on/off toggle, "disable on this site" toggle, status (engine state, media counts on this tab via a `tabStats` message you add to the protocol and implement in background by counting `data-hb-state` via `chrome.scripting.executeScript` or a content-side `getTabStats` handler), quick toggles for images/videos/male/female blur, link to options page.
2. Options page: every setting in `src/shared/settings.ts`, the allowlist editor, blur amount slider with live preview on a bundled sample image, strictness slider, reveal settings, fail-open toggle, an About section listing the model/library licenses (AGPL-3.0, MIT, Apache-2.0).
3. i18n: `browser.i18n`/`$i18n` wired through `_locales`; carry over en/ar/tr/ur strings from the old repo (see git history `src/translations.js`), translate the new keys into ar/tr/ur. RTL layout for ar/ur.
4. Context menu on images/videos: "Reveal this media" (removes blur on that element via a message to the content script) and "Re-check this media". Background owns the menu.
5. Settings changes apply live to open tabs without refresh (settings change events already exist in shared).

Unit tests where meaningful. Acceptance: CI green; popup and options render correctly in Chromium including an RTL language; verified yourself in the browser. Do not merge.
