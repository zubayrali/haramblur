/**
 * HaramBlur v1 settings — types, defaults and the 0.2.x migration, with no
 * storage access. Chrome offscreen documents have `chrome.runtime` but no
 * `chrome.storage`, so engine code (host/worker) must import only this module;
 * the storage-backed item lives in `settings-store.ts`, used by the
 * background, content script and UI entrypoints.
 *
 * Versioned with WXT storage migrations. `hb-settings-v2` is stored under the
 * same `hb-settings` key the 0.2.x extension used in `chrome.storage.sync`, so
 * the v1 → v2 migration runs on the user's real stored object on update.
 */
export const SETTINGS_KEY = "sync:hb-settings";
export const SETTINGS_VERSION = 6;

/** How blurred media is censored visually. */
export type CensorStyle = "blur" | "solid" | "static" | "pattern";

/** How far a blurred face region expands to cover hair/neck/body. */
export type CoverageMode = "face" | "head" | "body";

export interface Settings {
	/** Master on/off switch (old `status`). */
	status: boolean;
	/** Blur media before detection starts (fail closed). */
	blurryStartMode: boolean;
	/** CSS blur radius/percent carried over from 0.2.x. */
	blurAmount: number;
	/** Apply grayscale on top of blur. */
	gray: boolean;
	/** 0..1; higher = stricter (lower confidence thresholds). */
	strictness: number;
	blurImages: boolean;
	blurVideos: boolean;
	blurMale: boolean;
	blurFemale: boolean;
	/**
	 * Also blur modesty/secondary exposure classes (bare belly, armpits, feet,
	 * covered chest, male breast). Off by default — NudeNet's precision on
	 * these classes is poor on ordinary photos.
	 */
	strictModesty: boolean;
	/**
	 * Skip blurring faces the gender model estimates as children
	 * (age < CHILD_AGE_MAX). Wired up once detections carry `age`.
	 */
	skipChildren: boolean;
	unblurImages: boolean;
	unblurVideos: boolean;
	/** Hostnames where detection is disabled (old `whitelist`). */
	allowlist: string[];
	/**
	 * What to do when classification times out or errors. `true` = keep blurred
	 * (the default per PLAN.md, "fail closed").
	 */
	failClosed: boolean;
	/**
	 * Blur only the detected regions of an image instead of the whole element
	 * (Shadow-DOM portal overlays). Whole-element blur remains the fallback
	 * for full-frame detections (e.g. NSFW_SUGGESTIVE/EXPLICIT) and whenever
	 * region layout can't be mirrored.
	 */
	regionBlur: boolean;
	/**
	 * How far a blurred female/male face box expands past the detected face:
	 * `face` = detector box only, `head` = +hair/neck, `body` = estimated
	 * head-and-torso silhouette.
	 */
	coverageMode: CoverageMode;
	/**
	 * Escalate region blur to a whole-element censor when the content is
	 * "too much to localize": explicit (`CORE_EXPOSED`) detections, ≥2
	 * female faces, or blur regions covering more than `escalateAreaPct`
	 * of the frame.
	 */
	smartCensor: boolean;
	/** Fraction (0..1) of the frame the union of blur regions may cover
	 * before `smartCensor` escalates to whole-element blur. */
	escalateAreaPct: number;
	/**
	 * Always whole-element censor images (feed thumbnails, background
	 * images) instead of region-blurring — calmer browsing; videos keep
	 * region coverage.
	 */
	wholeImages: boolean;
	/**
	 * Visual treatment of censored areas: `blur` (frosted), `solid`
	 * (opaque matte — no silhouette leakage), `static` (noise fill),
	 * `pattern` (calm geometric tile).
	 */
	censorStyle: CensorStyle;
}

export const DEFAULT_SETTINGS: Settings = {
	status: true,
	blurryStartMode: true,
	blurAmount: 20,
	gray: true,
	strictness: 0.5,
	blurImages: true,
	blurVideos: true,
	blurMale: false,
	blurFemale: true,
	strictModesty: false,
	skipChildren: true,
	unblurImages: false,
	unblurVideos: false,
	allowlist: [],
	failClosed: true,
	regionBlur: true,
	coverageMode: "head",
	smartCensor: true,
	escalateAreaPct: 0.3,
	wholeImages: false,
	censorStyle: "blur",
};

/** Shape stored by HaramBlur 0.2.x under `chrome.storage.sync["hb-settings"]`. */
export interface LegacySettings {
	status?: boolean;
	blurryStartMode?: boolean;
	blurAmount?: number;
	gray?: boolean;
	strictness?: number;
	blurImages?: boolean;
	blurVideos?: boolean;
	blurMale?: boolean;
	blurFemale?: boolean;
	unblurImages?: boolean;
	unblurVideos?: boolean;
	whitelist?: string[];
}

const clamp = (n: number, lo: number, hi: number) => Math.min(hi, Math.max(lo, n));
const bool = (v: unknown, fallback: boolean): boolean => (typeof v === "boolean" ? v : fallback);
const num = (v: unknown, fallback: number, lo: number, hi: number): number =>
	typeof v === "number" && Number.isFinite(v) ? clamp(v, lo, hi) : fallback;

/**
 * Convert a stored HaramBlur 0.2.x `hb-settings` object into v1 settings.
 * Unknown types fall back to defaults; `whitelist` is renamed to `allowlist`;
 * strictness and blurAmount are clamped to their valid ranges.
 */
export function migrateLegacySettings(legacy: LegacySettings): Settings {
	return {
		status: bool(legacy.status, DEFAULT_SETTINGS.status),
		blurryStartMode: bool(legacy.blurryStartMode, DEFAULT_SETTINGS.blurryStartMode),
		blurAmount: num(legacy.blurAmount, DEFAULT_SETTINGS.blurAmount, 0, 100),
		gray: bool(legacy.gray, DEFAULT_SETTINGS.gray),
		strictness: num(legacy.strictness, DEFAULT_SETTINGS.strictness, 0, 1),
		blurImages: bool(legacy.blurImages, DEFAULT_SETTINGS.blurImages),
		blurVideos: bool(legacy.blurVideos, DEFAULT_SETTINGS.blurVideos),
		blurMale: bool(legacy.blurMale, DEFAULT_SETTINGS.blurMale),
		blurFemale: bool(legacy.blurFemale, DEFAULT_SETTINGS.blurFemale),
		strictModesty: DEFAULT_SETTINGS.strictModesty,
		skipChildren: DEFAULT_SETTINGS.skipChildren,
		unblurImages: bool(legacy.unblurImages, DEFAULT_SETTINGS.unblurImages),
		unblurVideos: bool(legacy.unblurVideos, DEFAULT_SETTINGS.unblurVideos),
		allowlist: Array.isArray(legacy.whitelist)
			? legacy.whitelist.filter((h): h is string => typeof h === "string")
			: [],
		failClosed: DEFAULT_SETTINGS.failClosed,
		regionBlur: DEFAULT_SETTINGS.regionBlur,
		coverageMode: DEFAULT_SETTINGS.coverageMode,
		smartCensor: DEFAULT_SETTINGS.smartCensor,
		escalateAreaPct: DEFAULT_SETTINGS.escalateAreaPct,
		wholeImages: DEFAULT_SETTINGS.wholeImages,
		censorStyle: DEFAULT_SETTINGS.censorStyle,
	};
}
