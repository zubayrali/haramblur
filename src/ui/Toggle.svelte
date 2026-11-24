<script lang="ts">
	interface Props {
		label: string;
		hint?: string;
		checked: boolean;
		testid?: string;
		disabled?: boolean;
		onchange?: (checked: boolean) => void;
	}

	let {
		label,
		hint,
		checked = $bindable(false),
		testid,
		disabled = false,
		onchange,
	}: Props = $props();

	function toggle() {
		checked = !checked;
		onchange?.(checked);
	}
</script>

<div class="flex items-start justify-between gap-3 py-1.5">
	<div class="min-w-0">
		<span class="block text-sm font-medium text-gray-800 dark:text-gray-100">{label}</span>
		{#if hint}
			<span class="block text-xs text-gray-500 dark:text-gray-400">{hint}</span>
		{/if}
	</div>
	<button
		type="button"
		role="switch"
		aria-checked={checked}
		aria-label={label}
		data-testid={testid}
		{disabled}
		onclick={toggle}
		class="relative inline-flex h-5 w-9 shrink-0 cursor-pointer items-center rounded-full border-0 transition-colors disabled:cursor-not-allowed disabled:opacity-50 {checked
			? 'bg-emerald-600'
			: 'bg-gray-300 dark:bg-gray-600'}"
	>
		<span
			class="pointer-events-none inline-block h-4 w-4 transform rounded-full bg-white shadow transition-transform {checked
				? 'translate-x-4 rtl:-translate-x-4'
				: 'translate-x-0.5 rtl:-translate-x-0.5'}"
		></span>
	</button>
</div>
