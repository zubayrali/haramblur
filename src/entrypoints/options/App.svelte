<script lang="ts">
	import { onMount } from "svelte";
	import { browser } from "wxt/browser";
	import { downloadPack, getModelStore, type InstalledModel } from "../../engine/model-store";
	import { t } from "../../shared/i18n";
	import { MODEL_SLOTS, type ModelSlot } from "../../shared/model-slots";
	import { packsForSlot, type ModelPack } from "../../shared/model-packs";
	import { PROFILES, profileSettings } from "../../shared/profiles";
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
			name: "HaramBlur 0.2.x (upstream)",
			license: "AGPL-3.0",
			url: "https://github.com/alganzory/HaramBlur",
		},
		{
			name: "NudeNet 320n (detector model)",
			license: "AGPL-3.0",
			url: "https://huggingface.co/deepghs/nudenet_onnx",
		},
		{
			name: "OpenCV YuNet (face detector)",
			license: "Apache-2.0",
			url: "https://github.com/opencv/opencv_zoo",
		},
		{
			name: "FastFace small112 (gender/age model)",
			license: "Apache-2.0",
			url: "https://huggingface.co/iFurySt/fastface",
		},
		{
			name: "nsfwjs MobileNetV2 (NSFW classifier)",
			license: "MIT",
			url: "https://github.com/infectedbyamy/nsfwjs",
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
	let installedModels = $state<InstalledModel[]>([]);
	let importSlot = $state<ModelSlot | null>(null);
	let fileInput = $state<HTMLInputElement | null>(null);

	const installedFor = (slot: ModelSlot) => installedModels.find((m) => m.slot === slot);

	async function refreshModels() {
		try {
			installedModels = await getModelStore().list();
		} catch {
			installedModels = [];
		}
	}

	function pickImport(slot: ModelSlot) {
		importSlot = slot;
		fileInput?.click();
	}

	async function onModelFile(e: Event) {
		const input = e.target as HTMLInputElement;
		const file = input.files?.[0];
		const slot = importSlot;
		input.value = "";
		if (file == null || slot == null) return;
		const bytes = new Uint8Array(await file.arrayBuffer());
		await getModelStore().put(slot, bytes, file.name);
		await refreshModels();
	}

	async function removeModel(slot: ModelSlot) {
		await getModelStore().remove(slot);
		await refreshModels();
	}

	let downloading = $state<Record<string, number>>({});
	let packError = $state<string | null>(null);

	async function download(pack: ModelPack) {
		packError = null;
		downloading = { ...downloading, [pack.id]: 0 };
		try {
			await downloadPack(pack, getModelStore(), (p) => {
				downloading = { ...downloading, [pack.id]: Math.round((p.received / p.total) * 100) };
			});
			await refreshModels();
		} catch (err) {
			packError = err instanceof Error ? err.message : String(err);
		} finally {
			const d = { ...downloading };
			delete d[pack.id];
			downloading = d;
		}
	}

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
		void refreshModels();
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
		<Section title={t("profiles")} testid="section-profiles">
			<div class="flex gap-2">
				{#each PROFILES as p (p.id)}
					<button
						type="button"
						class="flex-1 rounded-md border border-gray-300 px-2 py-1.5 text-sm font-medium text-gray-700 hover:bg-gray-100 dark:border-gray-600 dark:text-gray-200 dark:hover:bg-gray-800"
						data-testid={`opt-profile-${p.id}`}
						onclick={() => persist(profileSettings(p))}
					>
						{t(`profile_${p.id}`)}
					</button>
				{/each}
			</div>
			<p class="mt-1 text-xs text-gray-500 dark:text-gray-400">{t("profilesHint")}</p>
		</Section>

		<Section title={t("models")} testid="section-models">
			<p class="mb-2 text-xs text-gray-500 dark:text-gray-400">{t("modelsHint")}</p>
			<input
				bind:this={fileInput}
				type="file"
				accept=".onnx"
				class="hidden"
				data-testid="model-file-input"
				onchange={onModelFile}
			/>
			<ul class="space-y-2">
				{#each MODEL_SLOTS as slot (slot.id)}
					{@const inst = installedFor(slot.id)}
					<li class="rounded-md border border-gray-200 p-2 dark:border-gray-700" data-testid={`model-row-${slot.id}`}>
						<div class="flex items-center justify-between gap-2">
							<div class="min-w-0">
								<div class="truncate text-sm font-medium text-gray-800 dark:text-gray-100">
									{t(slot.labelKey)}
								</div>
								<div class="truncate text-xs text-gray-500 dark:text-gray-400">
									{inst != null
										? `${inst.meta.fileName} · ${(inst.meta.size / 1048576).toFixed(1)} MB`
										: `${t("modelBundled")} · ${slot.bundledFile}`}
								</div>
							</div>
							<div class="flex shrink-0 gap-1">
								<button
									type="button"
									class="rounded border border-gray-300 px-2 py-1 text-xs hover:bg-gray-100 dark:border-gray-600 dark:hover:bg-gray-800"
									data-testid={`model-import-${slot.id}`}
									onclick={() => pickImport(slot.id)}
								>
									{inst != null ? t("modelReplace") : t("modelImport")}
								</button>
								{#if inst != null}
									<button
										type="button"
										class="rounded border border-gray-300 px-2 py-1 text-xs hover:bg-gray-100 dark:border-gray-600 dark:hover:bg-gray-800"
										data-testid={`model-remove-${slot.id}`}
										onclick={() => removeModel(slot.id)}
									>
										{t("modelRemove")}
									</button>
								{/if}
							</div>
						</div>
						<p class="mt-1 text-[11px] text-gray-400 dark:text-gray-500">{t(slot.contractKey)}</p>
						{#each packsForSlot(slot.id) as pack (pack.id)}
							<div class="mt-1.5 flex items-center gap-2">
								<button
									type="button"
									class="rounded border border-gray-300 px-2 py-1 text-xs hover:bg-gray-100 disabled:opacity-50 dark:border-gray-600 dark:hover:bg-gray-800"
									data-testid={`pack-download-${pack.id}`}
									disabled={downloading[pack.id] != null}
									onclick={() => download(pack)}
								>
									{downloading[pack.id] != null
										? `${t("modelDownloading")} ${downloading[pack.id]}%`
										: t("modelDownload")}
								</button>
								<span class="text-[11px] text-gray-400 dark:text-gray-500">
									{t(pack.labelKey)} · {(pack.sizeBytes / 1048576).toFixed(1)} MB — {t(pack.descKey)}
								</span>
							</div>
						{/each}
					</li>
				{/each}
			</ul>
			{#if packError != null}
				<p class="mt-2 text-xs text-red-600 dark:text-red-400" data-testid="pack-error">{packError}</p>
			{/if}
			<p class="mt-2 text-xs text-gray-500 dark:text-gray-400">{t("modelsReloadNote")}</p>
		</Section>

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
			{#if s.regionBlur}
				<Toggle
					label={t("smartCensor")}
					hint={t("smartCensorHint")}
					bind:checked={s.smartCensor}
					testid="opt-smart-censor"
					onchange={(v) => persist({ smartCensor: v })}
				/>
				{#if s.smartCensor}
					<SliderRow
						label={t("escalateAreaPct")}
						min={5}
						max={90}
						bind:value={() => Math.round(s!.escalateAreaPct * 100), (v) =>
							persist({ escalateAreaPct: v / 100 })}
						testid="opt-escalate-area"
					/>
				{/if}
				<Toggle
					label={t("wholeImages")}
					hint={t("wholeImagesHint")}
					bind:checked={s.wholeImages}
					testid="opt-whole-images"
					onchange={(v) => persist({ wholeImages: v })}
				/>
			{/if}
			<div class="py-1.5" data-testid="opt-censor-style">
				<label class="block text-sm font-medium text-gray-800 dark:text-gray-100" for="censor-style">
					{t("censorStyle")}
				</label>
				<p class="mb-1 text-xs text-gray-500 dark:text-gray-400">{t("censorStyleHint")}</p>
				<select
					id="censor-style"
					class="w-full rounded-md border border-gray-300 bg-white px-2 py-1.5 text-sm text-gray-800 dark:border-gray-600 dark:bg-gray-800 dark:text-gray-100"
					value={s.censorStyle}
					onchange={(e) => persist({ censorStyle: (e.target as HTMLSelectElement).value as Settings["censorStyle"] })}
				>
					<option value="blur">{t("censorStyleBlur")}</option>
					<option value="solid">{t("censorStyleSolid")}</option>
					<option value="static">{t("censorStyleStatic")}</option>
					<option value="pattern">{t("censorStylePattern")}</option>
				</select>
			</div>
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
			<Toggle
				label={t("strictModesty")}
				hint={t("strictModestyHint")}
				bind:checked={s.strictModesty}
				testid="opt-strict-modesty"
				onchange={(v) => persist({ strictModesty: v })}
			/>
			<Toggle
				label={t("skipChildren")}
				hint={t("skipChildrenHint")}
				bind:checked={s.skipChildren}
				testid="opt-skip-children"
				onchange={(v) => persist({ skipChildren: v })}
			/>
			<div class="py-1.5" data-testid="opt-coverage-mode">
				<label class="block text-sm font-medium text-gray-800 dark:text-gray-100" for="coverage-mode">
					{t("coverageMode")}
				</label>
				<p class="mb-1 text-xs text-gray-500 dark:text-gray-400">{t("coverageModeHint")}</p>
				<select
					id="coverage-mode"
					class="w-full rounded-md border border-gray-300 bg-white px-2 py-1.5 text-sm text-gray-800 dark:border-gray-600 dark:bg-gray-800 dark:text-gray-100"
					value={s.coverageMode}
					onchange={(e) => persist({ coverageMode: (e.target as HTMLSelectElement).value as Settings["coverageMode"] })}
				>
					<option value="face">{t("coverageFace")}</option>
					<option value="head">{t("coverageHead")}</option>
					<option value="body">{t("coverageBody")}</option>
				</select>
			</div>
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
					href="https://github.com/zubayrali/haramblur/issues"
					target="_blank"
					rel="noreferrer"
					class="text-emerald-700 hover:underline dark:text-emerald-400"
					data-testid="feedback-link">{t("feedbackIssues")}</a
				>.
			</p>
		</Section>
	{/if}
</main>
