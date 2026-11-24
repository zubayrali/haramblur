<script lang="ts">
	import type { Snippet } from "svelte";

	interface Props {
		title: string;
		testid?: string;
		children: Snippet;
	}

	let { title, testid, children }: Props = $props();
</script>

<section
	data-testid={testid}
	class="rounded-lg border border-gray-200 bg-white p-4 dark:border-gray-700 dark:bg-gray-800"
>
	<h2 class="mb-2 text-sm font-semibold tracking-wide text-gray-700 uppercase dark:text-gray-200">
		{title}
	</h2>
	{@render children()}
</section>
