<script lang="ts">
	import { browser } from "wxt/browser";
	import { sendMessage } from "../../shared/protocol";

	let pong = $state<string | null>(null);
	let error = $state<string | null>(null);

	async function ping() {
		error = null;
		try {
			const res = await sendMessage("ping");
			pong = `pong @ ${new Date(res.ts).toLocaleTimeString()}`;
		} catch (err) {
			error = String(err);
		}
	}
</script>

<main class="w-64 p-4 font-sans">
	<h1 class="text-lg font-bold">{browser.i18n.getMessage("extName")}</h1>
	<p class="mt-1 text-sm text-gray-600">
		WP0 scaffold — the settings UI arrives in WP4.
	</p>
	<button
		type="button"
		data-testid="ping"
		onclick={ping}
		class="mt-3 rounded bg-emerald-600 px-3 py-1 text-sm text-white"
	>
		Ping background
	</button>
	{#if pong}
		<p data-testid="pong" class="mt-2 text-xs text-emerald-700">{pong}</p>
	{/if}
	{#if error}
		<p data-testid="ping-error" class="mt-2 text-xs text-red-600">{error}</p>
	{/if}
</main>
