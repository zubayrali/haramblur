import { describe, expect, it } from "vitest";
import {
	DEFAULT_SETTINGS,
	migrateLegacySettings,
	migrateV6Settings,
} from "../../src/shared/settings";

describe("migrateLegacySettings (HaramBlur 0.2.x hb-settings)", () => {
	it("maps a full legacy object, renaming whitelist to allowlist", () => {
		const migrated = migrateLegacySettings({
			status: false,
			blurryStartMode: true,
			blurAmount: 30,
			gray: false,
			strictness: 0.8,
			blurImages: false,
			blurVideos: false,
			blurMale: true,
			blurFemale: false,
			unblurImages: true,
			unblurVideos: true,
			whitelist: ["example.com", "foo.bar"],
		});
		expect(migrated).toEqual({
			...DEFAULT_SETTINGS,
			status: false,
			blurryStartMode: true,
			blurAmount: 30,
			gray: false,
			strictness: 0.8,
			blurImages: false,
			blurVideos: false,
			blurMale: true,
			blurFemale: false,
			revealModeImages: "hover",
			revealModeVideos: "hover",
			allowlist: ["example.com", "foo.bar"],
			failClosed: true,
		});
	});

	it("fills defaults for missing fields", () => {
		expect(migrateLegacySettings({})).toEqual(DEFAULT_SETTINGS);
	});

	it("clamps strictness and blurAmount into range", () => {
		const migrated = migrateLegacySettings({
			strictness: 42,
			blurAmount: -5,
		});
		expect(migrated.strictness).toBe(1);
		expect(migrated.blurAmount).toBe(0);
	});

	it("drops non-boolean and non-string values", () => {
		const migrated = migrateLegacySettings({
			// 0.2.x stored booleans; anything else is corrupt → default.
			status: 1 as unknown as boolean,
			whitelist: ["ok.com", 7 as unknown as string],
		});
		expect(migrated.status).toBe(DEFAULT_SETTINGS.status);
		expect(migrated.allowlist).toEqual(["ok.com"]);
	});
});

describe("migrateV6Settings (v6 → v7)", () => {
	it("maps the unblur booleans onto per-kind reveal modes", () => {
		const migrated = migrateV6Settings({
			...DEFAULT_SETTINGS,
			unblurImages: true,
			unblurVideos: false,
		} as never);
		expect(migrated.revealModeImages).toBe("hover");
		expect(migrated.revealModeVideos).toBe("off");
	});

	it("drops the legacy keys and fills the recovery-UX defaults", () => {
		const migrated = migrateV6Settings({
			...DEFAULT_SETTINGS,
			unblurImages: false,
			unblurVideos: true,
		} as never);
		expect(migrated).not.toHaveProperty("unblurImages");
		expect(migrated).not.toHaveProperty("unblurVideos");
		expect(migrated.settingsLock).toBe(false);
		expect(migrated.youtubeCaptionCover).toBe(true);
	});

	it("preserves every carried-over v6 value", () => {
		const migrated = migrateV6Settings({
			...DEFAULT_SETTINGS,
			strictness: 0.9,
			censorStyle: "solid",
			unblurImages: false,
			unblurVideos: false,
		} as never);
		expect(migrated.strictness).toBe(0.9);
		expect(migrated.censorStyle).toBe("solid");
	});
});
