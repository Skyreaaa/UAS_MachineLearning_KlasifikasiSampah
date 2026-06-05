// Warna per kategori sampah (selaras dengan backend TRASH_INFO).
export const CATEGORY_COLORS: Record<string, string> = {
  Organik: '#10B981', // hijau
  Anorganik: '#F59E0B', // kuning
  B3: '#EF4444', // merah
  Residu: '#6B7280', // abu-abu
}

/** Warna aksen untuk sebuah kategori (fallback abu-abu). */
export function categoryColor(category: string | undefined | null): string {
  if (!category) return '#6B7280'
  return CATEGORY_COLORS[category] ?? '#6B7280'
}

/** Warna bar kepercayaan berdasarkan nilai 0–100. */
export function confidenceColor(confidence: number): string {
  if (confidence >= 90) return '#10B981'
  if (confidence >= 70) return '#F59E0B'
  return '#EF4444'
}
