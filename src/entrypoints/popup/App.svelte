<script lang="ts">
	import { onMount } from "svelte";
	import { browser } from "wxt/browser";
	import { type I18nKey, t } from "../../shared/i18n";
	import { sendMessage } from "../../shared/protocol";
	import type { EngineState, EngineStatus, TabStats } from "../../shared/protocol";
	import { getSettings, settings, watchSettings } from "../../shared/settings";
	import type { Settings } from "../../shared/settings";
	import Toggle from "../../ui/Toggle.svelte";
	import { hostnameFromUrl, toggleAllowlistHost } from "../../ui/host";

	let s = $state<Settings | null>(null);
	let engine = $state<EngineStatus | null>(null);
	let stats = $state<TabStats | null>(null);
	let siteHost = $state<string | null>(null);

	const siteDisabled = $derived(s != null && siteHost != null && s.allowlist.includes(siteHost));
	const ENGINE_KEYS: Record<EngineState, I18nKey> = {
		booting: "engineBooting",
		ready: "engineReady",
		degraded: "engineDegraded",
		failed: "engineFailed",
	};
	const engineStateKey = $derived(
		engine == null ? "engineUnavailable" : ENGINE_KEYS[engine.state],
	);

	async function persist(patch: Partial<Settings>) {
		if (s == null) return;
		const next = { ...$state.snapshot(s), ...patch };
		await settings.setValue(next);
		s = next;
	}

	function toggleSite(enabled: boolean) {
		if (siteHost == null || s == null) return;
		const allowlist = toggleAllowlistHost(s.allowlist, siteHost);
		if (enabled !== allowlist.includes(siteHost)) return;
		void persist({ allowlist });
	}

	function openOptions() {
		void browser.runtime.openOptionsPage();
	}

	onMount(() => {
		const unwatch = watchSettings((next) => (s = next));
		void (async () => {
			s = await getSettings();
			const [tab] = await browser.tabs.query({ active: true, currentWindow: true });
			siteHost = hostnameFromUrl(tab?.url);
			const tabId = tab?.id;
			if (tabId != null) {
				try {
					stats = await sendMessage("getTabStats", { tabId });
				} catch {
					stats = null;
				}
			}
			try {
				engine = await sendMessage("engineStatus");
			} catch {
				engine = null;
			}
		})();
		return unwatch;
	});
</script>

<main
	data-testid="popup-root"
	class="w-72 bg-white p-3 font-sans text-gray-900 dark:bg-gray-900 dark:text-gray-100"
>
	<header class="flex items-center justify-between">
		<h1 class="text-base font-bold">{t("extName")}</h1>
		<button
			type="button"
			data-testid="open-options"
			onclick={openOptions}
			class="text-xs text-emerald-700 hover:underline dark:text-emerald-400"
		>
			{t("openSettings")}
		</button>
	</header>

	{#if s}
		<div class="mt-2 divide-y divide-gray-100 dark:divide-gray-800">
			<Toggle
				label={t("masterToggle")}
				bind:checked={s.status}
				testid="master-toggle"
				onchange={(v) => persist({ status: v })}
			/>
			{#if siteHost != null}
				<Toggle
					label={t("siteDetection", siteHost)}
					bind:checked={() => !siteDisabled, (v) => toggleSite(v)}
					testid="site-toggle"
				/>
			{/if}
		</div>

		<section class="mt-3 rounded-md bg-gray-50 p-2 text-xs dark:bg-gray-800" data-testid="status">
			<div class="flex justify-between">
				<span class="font-medium text-gray-600 dark:text-gray-300">{t("engineStatus")}</span>
				<span
					data-testid="engine-state"
					class="font-semibold {engine?.state === 'ready'
						? 'text-emerald-600'
						: engine?.state === 'failed'
							? 'text-red-600'
							: 'text-amber-600'}"
				>
					{t(engineStateKey)}{engine?.backend ? ` · ${engine.backend}` : ""}
				</span>
			</div>
			<div class="mt-1 flex justify-between">
				<span class="font-medium text-gray-600 dark:text-gray-300">{t("mediaOnThisPage")}</span>
				{#if stats == null || stats.total === 0}
					<span class="text-gray-500" data-testid="media-count">{t("noMedia")}</span>
				{:else}
					<span class="text-gray-700 dark:text-gray-200" data-testid="media-count">
						{stats.total}
						({stats.images}
						{t("blurImages").toLowerCase()} · {stats.videos} {t("blurVideos").toLowerCase()})
					</span>
				{/if}
			</div>
			{#if stats != null && stats.total > 0}
				<div class="mt-1 flex flex-wrap gap-1">
					{#each Object.entries(stats.byState) as [state, count] (state)}
						<span
							class="rounded bg-gray-200 px-1.5 py-0.5 text-[10px] text-gray-700 dark:bg-gray-700 dark:text-gray-200"
						>
							{state}: {count}
						</span>
					{/each}
				</div>
			{/if}
		</section>

		<h2 class="mt-3 text-xs font-semibold tracking-wide text-gray-500 uppercase">
			{t("mediaToBlur")}
		</h2>
		<div class="divide-y divide-gray-100 dark:divide-gray-800">
			<Toggle
				label={t("blurImages")}
				bind:checked={s.blurImages}
				testid="toggle-blur-images"
				onchange={(v) => persist({ blurImages: v })}
			/>
			<Toggle
				label={t("blurVideos")}
				bind:checked={s.blurVideos}
				testid="toggle-blur-videos"
				onchange={(v) => persist({ blurVideos: v })}
			/>
			<Toggle
				label={t("blurMale")}
				bind:checked={s.blurMale}
				testid="toggle-blur-male"
				onchange={(v) => persist({ blurMale: v })}
			/>
			<Toggle
				label={t("blurFemale")}
				bind:checked={s.blurFemale}
				testid="toggle-blur-female"
				onchange={(v) => persist({ blurFemale: v })}
			/>
		</div>
	{:else}
		<p class="mt-4 text-sm text-gray-500">…</p>
	{/if}
</main>
