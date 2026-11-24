/**
 * Per-site allowlist helpers shared by the content pipeline, the video
 * manager and the popup ("disable on this site"). Pure functions — no
 * storage or DOM access — so every context can use one implementation.
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
