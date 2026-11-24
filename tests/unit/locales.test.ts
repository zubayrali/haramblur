import { readdirSync, readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { describe, expect, it } from "vitest";

const LOCALES_DIR = join(dirname(fileURLToPath(import.meta.url)), "..", "..", "public", "_locales");
const SRC_DIR = join(dirname(fileURLToPath(import.meta.url)), "..", "..", "src");

interface MessageEntry {
	message: string;
	placeholders?: Record<string, unknown>;
}
type Messages = Record<string, MessageEntry>;

const locales = readdirSync(LOCALES_DIR, { withFileTypes: true })
	.filter((d) => d.isDirectory())
	.map((d) => d.name);
const entries: [string, Messages][] = locales.map((loc) => [
	loc,
	JSON.parse(readFileSync(join(LOCALES_DIR, loc, "messages.json"), "utf-8")) as Messages,
]);
const tables = new Map(entries);
const en = tables.get("en");
if (en == null) throw new Error("en locale missing");

describe("_locales parity", () => {
	it("ships en, ar, tr, ur", () => {
		expect(locales.slice().sort()).toEqual(["ar", "en", "tr", "ur"]);
	});

	it.each(entries)("%s has every en key and no extras", (loc, table) => {
		expect(Object.keys(table).sort(), `locale ${loc}`).toEqual(Object.keys(en).sort());
	});

	it.each(entries)("%s declares the placeholders its messages reference", (loc, table) => {
		for (const [key, entry] of Object.entries(table)) {
			const refs = [...entry.message.matchAll(/\$([a-zA-Z]+)\$/g)].map((m) => m[1] ?? "");
			for (const ref of refs) {
				expect(entry.placeholders ?? {}, `${loc}.${key} uses $${ref}$`).toHaveProperty(ref);
			}
		}
	});
});

describe("i18n keys used in source", () => {
	it('every t("key") call resolves in en messages', () => {
		const missing: string[] = [];
		const stack = [SRC_DIR];
		const keyRe = /\bt\(\s*"([a-zA-Z0-9]+)"/g;
		while (stack.length > 0) {
			const dir = stack.pop();
			if (dir == null) break;
			for (const name of readdirSync(dir, { withFileTypes: true })) {
				const path = join(dir, name.name);
				if (name.isDirectory()) stack.push(path);
				else if (/\.(svelte|ts)$/.test(name.name)) {
					const text = readFileSync(path, "utf-8");
					for (const m of text.matchAll(keyRe)) {
						const key = m[1];
						if (key != null && en[key] == null) missing.push(`${name.name}: ${key}`);
					}
				}
			}
		}
		expect(missing).toEqual([]);
	});
});
