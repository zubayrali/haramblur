<script lang="ts">
	import { onDestroy } from "svelte";

	interface Props {
		label: string;
		hint?: string;
		/** Continuous hold time required to complete, ms. */
		durationMs?: number;
		testid?: string;
		oncomplete?: () => void;
	}

	let { label, hint, durationMs = 10_000, testid, oncomplete }: Props = $props();

	/**
	 * Press-and-hold confirmation (settings lock). The hold must be
	 * continuous: releasing the pointer/key, or the button losing focus,
	 * resets progress to zero. Friction for a deliberate decision — not a
	 * security control.
	 */
	let progress = $state(0);
	let holding = $state(false);
	let startedAt = 0;
	let interval: ReturnType<typeof setInterval> | null = null;

	function stop() {
		holding = false;
		progress = 0;
		startedAt = 0;
		if (interval != null) {
			clearInterval(interval);
			interval = null;
		}
	}

	function start() {
		if (holding) return;
		holding = true;
		startedAt = Date.now();
		interval = setInterval(() => {
			progress = Math.min(1, (Date.now() - startedAt) / durationMs);
			if (progress >= 1) {
				stop();
				oncomplete?.();
			}
		}, 50);
	}

	function onKeydown(e: KeyboardEvent) {
		if ((e.key === " " || e.key === "Enter") && !e.repeat) start();
	}

	function onKeyup(e: KeyboardEvent) {
		if (e.key === " " || e.key === "Enter") stop();
	}

	onDestroy(stop);
</script>

<button
	type="button"
	data-testid={testid}
	aria-label={hint != null ? `${label} — ${hint}` : label}
	class="relative w-full overflow-hidden rounded-md border border-amber-300 bg-amber-50 px-3 py-2 text-sm font-medium text-amber-900 select-none dark:border-amber-700 dark:bg-amber-950 dark:text-amber-200"
	onpointerdown={start}
	onpointerup={stop}
	onpointercancel={stop}
	onpointerleave={stop}
	onkeydown={onKeydown}
	onkeyup={onKeyup}
	onblur={stop}
>
	<span
		class="absolute inset-y-0 left-0 bg-amber-200 transition-none dark:bg-amber-800"
		style:width="{progress * 100}%"
	></span>
	<span class="relative">
		{label}
		{#if holding}
			<span data-testid={testid != null ? `${testid}-progress` : undefined}>
				{Math.ceil((1 - progress) * (durationMs / 1000))}s
			</span>
		{/if}
	</span>
	{#if hint != null}
		<span class="relative mt-0.5 block text-xs font-normal text-amber-700 dark:text-amber-400">
			{hint}
		</span>
	{/if}
</button>
