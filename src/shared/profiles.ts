import { DEFAULT_SETTINGS, type Settings } from "./settings";

/**
 * One-click settings presets (docs/PLAN.md Phase 3). Each profile is a full
 * patch over the tunable fields — applying one resets related knobs to the
 * preset's intent rather than merging onto whatever the user had.
 */
export interface Profile {
	id: "balanced" | "gaze" | "recovery";
	patch: Partial<Settings>;
}

export const PROFILES: readonly Profile[] = [
	{
		id: "balanced",
		patch: {
			regionBlur: true,
			coverageMode: "head",
			smartCensor: true,
			escalateAreaPct: 0.3,
			wholeImages: false,
			censorStyle: "blur",
			blurAmount: 20,
			gray: true,
			strictness: 0.5,
		},
	},
	{
		id: "gaze",
		patch: {
			regionBlur: true,
			coverageMode: "body",
			smartCensor: true,
			escalateAreaPct: 0.2,
			wholeImages: true,
			censorStyle: "static",
			blurAmount: 24,
			gray: true,
			strictness: 0.65,
		},
	},
	{
		id: "recovery",
		patch: {
			regionBlur: true,
			coverageMode: "body",
			smartCensor: true,
			escalateAreaPct: 0.15,
			wholeImages: true,
			censorStyle: "solid",
			blurAmount: 30,
			gray: true,
			strictness: 0.75,
			blurryStartMode: true,
			failClosed: true,
			skipChildren: true,
		},
	},
];

/** Merge a profile patch over the defaults — the value `applyProfile` writes. */
export function profileSettings(profile: Profile): Settings {
	return { ...DEFAULT_SETTINGS, ...profile.patch };
}
