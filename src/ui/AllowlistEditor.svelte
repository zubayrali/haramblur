<script lang="ts">
	import { t } from "../shared/i18n";
	import { normalizeHostname } from "./host";

	interface Props {
		hosts: string[];
		testid?: string;
		onchange?: (hosts: string[]) => void;
	}

	let { hosts, testid, onchange }: Props = $props();

	let draft = $state("");
	let error = $state(false);

	function add() {
		const host = normalizeHostname(draft);
		if (host == null) {
			error = true;
			return;
		}
		error = false;
		draft = "";
		if (hosts.includes(host)) return;
		onchange?.([...hosts, host].sort());
	}

	function remove(host: string) {
		onchange?.(hosts.filter((h) => h !== host));
	}
</script>

<div data-testid={testid}>
	<p class="mb-2 text-xs text-gray-500 dark:text-gray-400">{t("allowlistDescription")}</p>
	<form
		class="flex gap-2"
		onsubmit={(e) => {
			e.preventDefault();
			add();
		}}
	>
		<input
			type="text"
			bind:value={draft}
			placeholder={t("allowlistPlaceholder")}
			data-testid="allowlist-input"
			class="min-w-0 flex-1 rounded-md border border-gray-300 px-2 py-1 text-sm dark:border-gray-600 dark:bg-gray-900 dark:text-gray-100"
		/>
		<button
			type="submit"
			data-testid="allowlist-add"
			class="rounded-md bg-emerald-600 px-3 py-1 text-sm text-white"
		>
			{t("addSite")}
		</button>
	</form>
	{#if error}
		<p data-testid="allowlist-error" class="mt-1 text-xs text-red-600">{t("invalidHostname")}</p>
	{/if}
	{#if hosts.length > 0}
		<ul class="mt-2 divide-y divide-gray-100 dark:divide-gray-700">
			{#each hosts as host (host)}
				<li class="flex items-center justify-between py-1 text-sm">
					<span class="font-mono text-gray-700 dark:text-gray-200">{host}</span>
					<button
						type="button"
						data-testid={`allowlist-remove-${host}`}
						onclick={() => remove(host)}
						class="text-xs text-red-600 hover:underline"
					>
						{t("removeSite")}
					</button>
				</li>
			{/each}
		</ul>
	{/if}
</div>
