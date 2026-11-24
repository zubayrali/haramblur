<script lang="ts">
	interface Props {
		label: string;
		min: number;
		max: number;
		step?: number;
		value: number;
		testid?: string;
		onchange?: (value: number) => void;
	}

	let { label, min, max, step = 1, value = $bindable(0), testid, onchange }: Props = $props();

	function onInput(event: Event) {
		value = Number((event.target as HTMLInputElement).value);
		onchange?.(value);
	}
</script>

<div class="py-1.5">
	<div class="flex items-center justify-between gap-3">
		<label class="text-sm font-medium text-gray-800 dark:text-gray-100" for={testid}>
			{label}
		</label>
		<span class="text-xs text-gray-500 tabular-nums dark:text-gray-400">{value}</span>
	</div>
	<input
		id={testid}
		data-testid={testid}
		type="range"
		{min}
		{max}
		{step}
		{value}
		oninput={onInput}
		class="mt-1 w-full accent-emerald-600"
	/>
</div>
