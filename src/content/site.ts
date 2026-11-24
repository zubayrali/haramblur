import { decide } from "../shared/policy";
import type { Detection, VerdictDecision } from "../shared/protocol";
import type { Settings } from "../shared/settings";

// Allowlist helpers live in shared/site.ts so the popup and video manager
// share the content pipeline's implementation; re-export for existing
// content-side importers.
export { isAllowlisted, normalizeHostEntry } from "../shared/site";

export interface ReasonEntry {
	label: Detection["label"];
	score: number;
	/** Normalized [x, y, w, h] over the source media; needed for region blur. */
	box?: Detection["box"];
	/** Estimated age when a face classifier provided one (skipChildren). */
	age?: number;
}

/** Serialize verdict reasons as `[{label, score, box?, age?}]` JSON for `data-hb-reasons`. */
export function reasonsToJson(detections: readonly Detection[]): string {
	const reasons: ReasonEntry[] = detections.map((d) => ({
		label: d.label,
		score: d.score,
		box: d.box,
		...(d.age != null ? { age: d.age } : {}),
	}));
	return JSON.stringify(reasons);
}

const isBox = (v: unknown): v is Detection["box"] =>
	Array.isArray(v) && v.length === 4 && v.every((n) => typeof n === "number" && Number.isFinite(n));

/**
 * Parse `data-hb-reasons` back into detections. Entries written before boxes
 * were persisted carry a zeroed box; `decide` never reads `box`.
 */
export function reasonsFromJson(json: string): Detection[] {
	try {
		const parsed: unknown = JSON.parse(json);
		if (!Array.isArray(parsed)) return [];
		return parsed.flatMap((item): Detection[] => {
			if (
				typeof item !== "object" ||
				item === null ||
				typeof (item as ReasonEntry).label !== "string" ||
				typeof (item as ReasonEntry).score !== "number"
			) {
				return [];
			}
			const { label, score, box, age } = item as ReasonEntry;
			return [
				{
					label,
					score,
					box: isBox(box) ? box : [0, 0, 0, 0],
					...(typeof age === "number" && Number.isFinite(age) ? { age } : {}),
				},
			];
		});
	} catch {
		return [];
	}
}

/** Re-run policy over detections stored in `data-hb-reasons`. */
export function redecide(reasonsJson: string, settings: Settings): VerdictDecision {
	return decide(reasonsFromJson(reasonsJson), settings);
}
