import { defineConfig } from "@playwright/test";

export default defineConfig({
	testDir: "tests/e2e",
	timeout: 60_000,
	retries: process.env.CI === "true" ? 1 : 0,
	workers: 1,
	reporter: process.env.CI === "true" ? "github" : "list",
});
