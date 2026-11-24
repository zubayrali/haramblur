<script lang="ts">
	import { browser } from "wxt/browser";
	import { getSettings, watchSettings } from "../../shared/settings";
	import type { Settings } from "../../shared/settings";

	let current = $state<Settings | null>(null);

	getSettings().then((s) => (current = s));
	watchSettings((s) => (current = s));
</script>

<main class="mx-auto max-w-xl p-6 font-sans">
	<h1 class="text-xl font-bold">{browser.i18n.getMessage("extName")}</h1>
	<p class="mt-1 text-sm text-gray-600">
		WP0 scaffold — the full options UI arrives in WP4.
	</p>
	{#if current}
		<pre
			class="mt-4 rounded border border-gray-200 bg-gray-50 p-3 text-xs">{JSON.stringify(current, null, 2)}</pre>
	{/if}
</main>
