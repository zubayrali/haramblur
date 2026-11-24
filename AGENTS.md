# Working rules

-   This repo is an AGPL-3.0 fork of alganzory/HaramBlur being rewritten as a WXT + TypeScript extension. `docs/PLAN.md` and `docs/ENGINE_NOTES.md` are the source of truth. Follow them; if they are wrong, say so in your PR description and pick the closest option — do not silently change the design.
-   Package manager: pnpm. Runtime: Node 22. TypeScript strict; no `any` without a comment.
-   Lint and format with Biome only (`pnpm check`). Vitest for unit tests, Playwright for E2E against the built extension.
-   Models are fetched by `pnpm models:fetch` into git-ignored `public/models/`; never commit model binaries. If your work needs the model, run `pnpm models:fetch` first.
-   Never load models or media from a CDN at runtime; everything ships in the extension.
-   Work on a branch `devin/<timestamp>-<slug>` or `feat/<slug>`, open a PR into `main`, do not merge it.
-   Do not touch directories owned by other work packages (see PLAN.md) except through `src/shared/` types.
