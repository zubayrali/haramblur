<script lang="ts">
	import { onMount } from "svelte";
	import { browser } from "wxt/browser";
	import { t } from "../../shared/i18n";
	import { getSettings, settings, watchSettings } from "../../shared/settings-store";
	import type { Settings } from "../../shared/settings";
	import AllowlistEditor from "../../ui/AllowlistEditor.svelte";
	import Section from "../../ui/Section.svelte";
	import SliderRow from "../../ui/SliderRow.svelte";
	import Toggle from "../../ui/Toggle.svelte";

	interface LicenseRow {
		name: string;
		license: string;
		url: string;
	}

	const LICENSES: LicenseRow[] = [
		{
			name: "HaramBlur (this extension)",
			license: "AGPL-3.0",
			url: "https://github.com/zubayrali/haramblur",
		},
		{
			name: "NudeNet 320n (detector model)",
			license: "AGPL-3.0",
			url: "https://huggingface.co/deepghs/nudenet_onnx",
		},
		{
			name: "MediaPipe BlazeFace (face helper)",
			license: "Apache-2.0",
			url: "https://ai.google.dev/edge/mediapipe/solutions/vision/face_detector",
		},
		{ name: "onnxruntime-web", license: "MIT", url: "https://github.com/microsoft/onnxruntime" },
		{ name: "WXT", license: "MIT", url: "https://wxt.dev" },
		{ name: "Svelte", license: "MIT", url: "https://svelte.dev" },
		{ name: "Tailwind CSS", license: "MIT", url: "https://tailwindcss.com" },
		{ name: "XState", license: "MIT", url: "https://xstate.js.org" },
		{
			name: "@webext-core/messaging",
			license: "MIT",
			url: "https://github.com/aklinker1/webext-core",
		},
		{ name: "Playwright", license: "Apache-2.0", url: "https://playwright.dev" },
		{ name: "Biome", license: "MIT / Apache-2.0", url: "https://biomejs.dev" },
	];

	let s = $state<Settings | null>(null);

	const previewFilter = $derived(
		s == null
			? ""
			: `blur(${s.blurAmount}px) ${s.gray ? "grayscale(100%)" : ""}`.trim(),
	);

	async function persist(patch: Partial<Settings>) {
		if (s == null) return;
		const next = { ...$state.snapshot(s), ...patch };
		await settings.setValue(next);
		s = next;
	}

	onMount(() => {
		const unwatch = watchSettings((next) => (s = next));
		void getSettings().then((v) => (s = v));
		return unwatch;
	});
</script>

<main
	data-testid="options-root"
	class="mx-auto max-w-2xl space-y-4 p-6 font-sans text-gray-900 dark:bg-gray-900 dark:text-gray-100"
>
	<header>
		<h1 class="text-xl font-bold">{t("extName")}</h1>
		<p class="text-sm text-gray-500 dark:text-gray-400">
			{t("settings")} · {t("versionLabel")}
			{browser.runtime.getManifest().version}
		</p>
	</header>

	{#if s}
		<Section title={t("settings")} testid="section-general">
			<Toggle
				label={t("masterToggle")}
				bind:checked={s.status}
				testid="opt-status"
				onchange={(v) => persist({ status: v })}
			/>
			<Toggle
				label={t("blurryStart")}
				hint={t("blurryStartTooltip")}
				bind:checked={s.blurryStartMode}
				testid="opt-blurry-start"
				onchange={(v) => persist({ blurryStartMode: v })}
			/>
		</Section>

		<Section title={t("blurSection")} testid="section-blur">
			<SliderRow
				label={t("blurAmountLabel")}
				min={0}
				max={100}
				bind:value={s.blurAmount}
				testid="opt-blur-amount"
				onchange={(v) => persist({ blurAmount: v })}
			/>
			<Toggle
				label={t("grayscale")}
				bind:checked={s.gray}
				testid="opt-gray"
				onchange={(v) => persist({ gray: v })}
			/>
			<figure class="mt-3">
				<img
					src={browser.runtime.getURL("/sample-preview.jpg")}
					alt={t("blurPreviewCaption")}
					data-testid="blur-preview"
					class="w-full rounded-md"
					style:filter={previewFilter}
				/>
				<figcaption class="mt-1 text-xs text-gray-500 dark:text-gray-400">
					{t("blurPreviewCaption")}
				</figcaption>
			</figure>
		</Section>

		<Section title={t("mediaToBlur")} testid="section-media">
			<Toggle
				label={t("blurImages")}
				bind:checked={s.blurImages}
				testid="opt-blur-images"
				onchange={(v) => persist({ blurImages: v })}
			/>
			<Toggle
				label={t("blurVideos")}
				bind:checked={s.blurVideos}
				testid="opt-blur-videos"
				onchange={(v) => persist({ blurVideos: v })}
			/>
			<Toggle
				label={t("regionBlur")}
				hint={t("regionBlurHint")}
				bind:checked={s.regionBlur}
				testid="opt-region-blur"
				onchange={(v) => persist({ regionBlur: v })}
			/>
		</Section>

		<Section title={t("facesToBlur")} testid="section-faces">
			<Toggle
				label={t("blurMale")}
				bind:checked={s.blurMale}
				testid="opt-blur-male"
				onchange={(v) => persist({ blurMale: v })}
			/>
			<Toggle
				label={t("blurFemale")}
				bind:checked={s.blurFemale}
				testid="opt-blur-female"
				onchange={(v) => persist({ blurFemale: v })}
			/>
			<SliderRow
				label={t("strictnessLabel")}
				min={0}
				max={100}
				bind:value={() => Math.round(s!.strictness * 100), (v) =>
					persist({ strictness: v / 100 })}
				testid="opt-strictness"
			/>
		</Section>

		<Section title={t("revealSection")} testid="section-reveal">
			<p class="mb-1 text-xs text-gray-500 dark:text-gray-400">{t("unblurOnHover")}</p>
			<Toggle
				label={t("unblurImages")}
				bind:checked={s.unblurImages}
				testid="opt-unblur-images"
				onchange={(v) => persist({ unblurImages: v })}
			/>
			<Toggle
				label={t("unblurVideos")}
				bind:checked={s.unblurVideos}
				testid="opt-unblur-videos"
				onchange={(v) => persist({ unblurVideos: v })}
			/>
		</Section>

		<Section title={t("allowlistTitle")} testid="section-allowlist">
			<AllowlistEditor
				hosts={s.allowlist}
				testid="allowlist-editor"
				onchange={(hosts) => persist({ allowlist: hosts })}
			/>
		</Section>

		<Section title={t("advancedSection")} testid="section-advanced">
			<Toggle
				label={t("failClosed")}
				hint={t("failClosedHint")}
				bind:checked={s.failClosed}
				testid="opt-fail-closed"
				onchange={(v) => persist({ failClosed: v })}
			/>
		</Section>

		<Section title={t("aboutSection")} testid="section-about">
			<p class="text-sm text-gray-600 dark:text-gray-300">{t("extDescription")}</p>
			<h3 class="mt-3 text-xs font-semibold tracking-wide text-gray-500 uppercase">
				{t("licensesHeading")}
			</h3>
			<ul class="mt-1 divide-y divide-gray-100 dark:divide-gray-700">
				{#each LICENSES as row (row.name)}
					<li class="flex items-baseline justify-between gap-3 py-1 text-sm">
						<a
							href={row.url}
							target="_blank"
							rel="noreferrer"
							class="truncate text-emerald-700 hover:underline dark:text-emerald-400"
						>
							{row.name}
						</a>
						<span class="shrink-0 font-mono text-xs text-gray-500 dark:text-gray-400">
							{row.license}
						</span>
					</li>
				{/each}
			</ul>
			<p class="mt-3 text-sm text-gray-600 dark:text-gray-300">
				{t("feedbackLead")}
				<a
					href="https://forms.gle/mDDjh44MKRVemWX86"
					target="_blank"
					rel="noreferrer"
					class="text-emerald-700 hover:underline dark:text-emerald-400">{t("feedbackForm")}</a
				>
				{t("feedbackOr")}
				<a
					href="mailto:md.alganzory@gmail.com"
					class="text-emerald-700 hover:underline dark:text-emerald-400">{t("feedbackEmail")}</a
				>
				❤️
			</p>
		</Section>
	{/if}
</main>
