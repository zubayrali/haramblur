/**
 * Hostname helpers shared by the popup ("disable on this site") and the
 * options allowlist editor.
 */

const HOSTNAME_RE =
	/^(?=.{1,253}\.?$)[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?(?:\.[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?)*\.?$/i;

/** Extract the hostname from a tab URL; null for extension/chrome/about pages. */
export function hostnameFromUrl(url: string | undefined): string | null {
	if (url == null) return null;
	try {
		const parsed = new URL(url);
		if (parsed.protocol !== "http:" && parsed.protocol !== "https:") return null;
		return parsed.hostname.toLowerCase();
	} catch {
		return null;
	}
}

/**
 * Normalize user input (a bare host or a pasted URL) to a hostname suitable
 * for the allowlist. Returns null when the input is not a valid hostname.
 */
export function normalizeHostname(input: string): string | null {
	const trimmed = input.trim().toLowerCase();
	if (trimmed === "") return null;
	let host = trimmed;
	if (trimmed.includes("://")) {
		try {
			host = new URL(trimmed).hostname;
		} catch {
			return null;
		}
	} else {
		// Strip a trailing path/port the user may have pasted without a scheme.
		host = host.split("/")[0]?.split(":")[0] ?? host;
	}
	host = host.replace(/\.$/, "");
	if (host === "" || !HOSTNAME_RE.test(host)) return null;
	return host;
}

/** Add or remove `host` from the allowlist, returning a new sorted list. */
export function toggleAllowlistHost(list: readonly string[], host: string): string[] {
	const next = list.includes(host) ? list.filter((h) => h !== host) : [...list, host];
	return next.sort();
}
