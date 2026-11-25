/**
 * Compact wire format for binary person masks. Browser runtime messaging is
 * JSON-only (no transferables), so masks cross the extension boundary as a
 * base64-encoded bitstream: row-major, most-significant bit first, 1 = inside
 * the person silhouette. Shared by the engine host (encoder) and the content
 * region portal (decoder); no storage access.
 */

export function bytesToBase64(bytes: Uint8Array): string {
	let binary = "";
	for (let off = 0; off < bytes.length; off += 0x8000) {
		binary += String.fromCharCode(...bytes.subarray(off, off + 0x8000));
	}
	return btoa(binary);
}

export function base64ToBytes(encoded: string): Uint8Array {
	const binary = atob(encoded);
	const out = new Uint8Array(binary.length);
	for (let i = 0; i < binary.length; i++) out[i] = binary.charCodeAt(i);
	return out;
}

/** Pack RGBA pixels' alpha channel into row-major MSB-first bits. */
export function packAlphaBits(rgba: Uint8ClampedArray, width: number, height: number): Uint8Array {
	const pixels = width * height;
	const out = new Uint8Array(Math.ceil(pixels / 8));
	for (let i = 0; i < pixels; i++) {
		if ((rgba[i * 4 + 3] ?? 0) > 0) out[i >> 3]! |= 0x80 >> (i & 7);
	}
	return out;
}

/** Expand base64 MSB-first bits into white RGBA pixels for canvas painting. */
export function maskBitsToRgba(
	encoded: string,
	width: number,
	height: number,
): Uint8ClampedArray<ArrayBuffer> {
	const bytes = base64ToBytes(encoded);
	const pixels = width * height;
	const rgba = new Uint8ClampedArray(pixels * 4);
	for (let i = 0; i < pixels; i++) {
		if (((bytes[i >> 3] ?? 0) & (0x80 >> (i & 7))) !== 0) {
			const o = i * 4;
			rgba[o] = 255;
			rgba[o + 1] = 255;
			rgba[o + 2] = 255;
			rgba[o + 3] = 255;
		}
	}
	return rgba;
}
