/**
 * LRU verdict cache keyed by image URL (docs/PLAN.md — cap 500).
 * `Map` iteration order is insertion order, so the first key is the oldest.
 */
export class LruCache<K, V> {
	private map = new Map<K, V>();

	constructor(private readonly cap: number) {}

	get(key: K): V | undefined {
		const value = this.map.get(key);
		if (value === undefined) return undefined;
		// Refresh recency.
		this.map.delete(key);
		this.map.set(key, value);
		return value;
	}

	set(key: K, value: V): void {
		if (this.map.has(key)) {
			this.map.delete(key);
		} else if (this.map.size >= this.cap) {
			const oldest = this.map.keys().next();
			if (!oldest.done) this.map.delete(oldest.value);
		}
		this.map.set(key, value);
	}

	has(key: K): boolean {
		return this.map.has(key);
	}

	get size(): number {
		return this.map.size;
	}

	clear(): void {
		this.map.clear();
	}
}
