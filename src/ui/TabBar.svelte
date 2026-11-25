<script lang="ts">
	import { tick } from "svelte";
	import { isRtl } from "../shared/i18n";

	interface Tab {
		id: string;
		label: string;
	}

	interface Props {
		tabs: readonly Tab[];
		active: string;
		testid?: string;
		onchange?: (id: string) => void;
	}

	let { tabs, active = $bindable(), testid, onchange }: Props = $props();

	let tablist = $state<HTMLDivElement | null>(null);

	function select(id: string) {
		if (id === active) return;
		active = id;
		onchange?.(id);
	}

	/** Arrow keys follow the *visual* order, so left/right flip under RTL. */
	async function onKeydown(e: KeyboardEvent) {
		const idx = tabs.findIndex((tab) => tab.id === active);
		if (idx < 0) return;
		const forward = isRtl() ? "ArrowLeft" : "ArrowRight";
		const backward = isRtl() ? "ArrowRight" : "ArrowLeft";
		let next = -1;
		if (e.key === forward) next = (idx + 1) % tabs.length;
		else if (e.key === backward) next = (idx - 1 + tabs.length) % tabs.length;
		else if (e.key === "Home") next = 0;
		else if (e.key === "End") next = tabs.length - 1;
		if (next < 0) return;
		e.preventDefault();
		const tab = tabs[next];
		if (tab == null) return;
		select(tab.id);
		await tick();
		tablist?.querySelector<HTMLButtonElement>(`[data-tab-id="${tab.id}"]`)?.focus();
	}
</script>

<div
	role="tablist"
	data-testid={testid}
	bind:this={tablist}
	class="flex overflow-x-auto"
>
	{#each tabs as tab (tab.id)}
		{@const selected = tab.id === active}
		<button
			type="button"
			role="tab"
			id={`hb-tab-${tab.id}`}
			data-tab-id={tab.id}
			data-testid={`tab-${tab.id}`}
			aria-selected={selected}
			aria-controls={`hb-panel-${tab.id}`}
			tabindex={selected ? 0 : -1}
			onclick={() => select(tab.id)}
			onkeydown={onKeydown}
			class="shrink-0 border-b-2 px-3 py-2 text-sm font-medium transition-colors {selected
				? 'border-emerald-600 text-emerald-700 dark:border-emerald-500 dark:text-emerald-400'
				: 'border-gray-200 text-gray-500 hover:border-gray-300 hover:text-gray-700 dark:border-gray-700 dark:text-gray-400 dark:hover:border-gray-600 dark:hover:text-gray-200'}"
		>
			{tab.label}
		</button>
	{/each}
	<!-- inactive button borders form the baseline; this fills the rest of the row -->
	<div aria-hidden="true" class="flex-1 border-b-2 border-gray-200 dark:border-gray-700"></div>
</div>
