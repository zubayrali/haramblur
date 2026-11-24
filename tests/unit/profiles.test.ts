import { describe, expect, it } from "vitest";
import { PROFILES, profileSettings } from "../../src/shared/profiles";
import { DEFAULT_SETTINGS } from "../../src/shared/settings";

describe("profiles", () => {
	it("exposes the three documented presets", () => {
		expect(PROFILES.map((p) => p.id)).toEqual(["balanced", "gaze", "recovery"]);
	});

	it("every profile resolves to a complete valid Settings object", () => {
		for (const p of PROFILES) {
			const s = profileSettings(p);
			// Every default key survives the merge.
			for (const key of Object.keys(DEFAULT_SETTINGS)) {
				expect(s).toHaveProperty(key);
			}
			expect(s.censorStyle).toMatch(/^(blur|solid|static|pattern)$/);
			expect(s.coverageMode).toMatch(/^(face|head|body)$/);
		}
	});

	it("stricter profiles escalate more aggressively than balanced", () => {
		const byId = Object.fromEntries(PROFILES.map((p) => [p.id, profileSettings(p)]));
		expect(byId.gaze!.escalateAreaPct).toBeLessThan(byId.balanced!.escalateAreaPct);
		expect(byId.recovery!.escalateAreaPct).toBeLessThan(byId.gaze!.escalateAreaPct);
		expect(byId.recovery!.strictness).toBeGreaterThan(byId.balanced!.strictness);
	});
});
