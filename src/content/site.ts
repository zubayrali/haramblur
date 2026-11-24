import { decide } from "../shared/policy";
import type { Detection, VerdictDecision } from "../shared/protocol";
import type { Settings } from "../shared/settings";

/**
 * Per-site allowlist + verdict-attr helpers. Pure functions so they stay
 * unit-testable without a browser (docs/PLAN.md "Policy": the per-site on/off
 * check lives in the content pipeline).
 */

/** Normalize an allowlist entry: strip scheme, path, port, lowercase. */
export function normalizeHostEntry(entry: string): string {
	let host = entry.trim().toLowerCase();
	if (host.length === 0) return "";
	// Strip a leading scheme and anything after the first "/" so entries like
	// "https://example.com/path" still work.
	host = host.replace(/^[a-z][a-z0-9+.-]*:\/\//, "");
	const slash = host.indexOf("/");
	if (slash >= 0) host = host.slice(0, slash);
	const colon = host.indexOf(":");
	if (colon >= 0) host = host.slice(0, colon);
	return host;
}

/**
 * `hostname` matches the allowlist when it equals an entry or is a subdomain of
 * one: "www.example.com" is covered by "example.com". Malformed entries are
 * ignored.
 */
export function isAllowlisted(hostname: string, allowlist: readonly string[]): boolean {
	const host = hostname.toLowerCase();
	for (const entry of allowlist) {
		const normalized = normalizeHostEntry(entry);
		if (normalized.length === 0) continue;
		if (host === normalized || host.endsWith(`.${normalized}`)) return true;
	}
	return false;
}

export interface ReasonEntry {
	label: Detection["label"];
	score: number;
}

/** Serialize verdict reasons as `[{label, score}]` JSON for `data-hb-reasons`. */
export function reasonsToJson(detections: readonly Detection[]): string {
	const reasons: ReasonEntry[] = detections.map((d) => ({ label: d.label, score: d.score }));
	return JSON.stringify(reasons);
}

/**
 * Parse `data-hb-reasons` back into detections. `decide` never reads `box`, so
 * parsed entries carry a zeroed box to keep the real `Detection` type.
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
			const { label, score } = item as ReasonEntry;
			return [{ label, score, box: [0, 0, 0, 0] }];
		});
	} catch {
		return [];
	}
}

/** Re-run policy over detections stored in `data-hb-reasons`. */
export function redecide(reasonsJson: string, settings: Settings): VerdictDecision {
	return decide(reasonsFromJson(reasonsJson), settings);
}
