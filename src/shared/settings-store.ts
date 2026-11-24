import { browser } from "wxt/browser";
import { storage } from "wxt/utils/storage";

import {
	DEFAULT_SETTINGS,
	type LegacySettings,
	migrateLegacySettings,
	SETTINGS_KEY,
	SETTINGS_VERSION,
	type Settings,
} from "./settings";

/**
 * The settings item. `version: 2` makes WXT run `migrations[2]` on any value
 * stored by 0.2.x (which has no version metadata and is treated as v1).
 * `init` writes defaults for brand-new installs only.
 *
 * Import this only from contexts with `chrome.storage` (background, content
 * scripts, extension pages) — never from the engine host or worker, which run
 * in an offscreen document on Chrome where `chrome.storage` is undefined.
 */
export const settings = storage.defineItem<Settings>(SETTINGS_KEY, {
	fallback: DEFAULT_SETTINGS,
	init: () => DEFAULT_SETTINGS,
	version: SETTINGS_VERSION,
	migrations: {
		// The stored value is a raw 0.2.x hb-settings object.
		2: (oldValue: LegacySettings) => migrateLegacySettings(oldValue),
		// v2 → v3: fill in settings added after the rewrite shipped.
		3: (oldValue: Settings) => ({ ...DEFAULT_SETTINGS, ...oldValue }),
		// v3 → v4: fill in regionBlur (and any other post-v3 additions).
		4: (oldValue: Settings) => ({ ...DEFAULT_SETTINGS, ...oldValue }),
	},
});

export async function getSettings(): Promise<Settings> {
	return settings.getValue();
}

/** Subscribe to setting changes; fires with (newValue, oldValue). */
export function watchSettings(cb: (newValue: Settings, oldValue: Settings) => void): () => void {
	return settings.watch(cb);
}

export type { Settings };
export { browser };
