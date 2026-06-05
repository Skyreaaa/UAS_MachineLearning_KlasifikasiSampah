// ─────────────────────────────────────────────────────────────
//  Tipe data yang dipakai bersama frontend
//  Bentuk response mengikuti backend FastAPI (backend/main.py)
// ─────────────────────────────────────────────────────────────

/** Salah satu kemungkinan kategori dari prediksi (Top-3). */
export interface Top3Item {
  lvl2: string
  confidence: number
}

/** Response dari endpoint POST /classify. */
export interface ClassifyResult {
  label: string // nama tampilan, mis. "Anorganik"
  code: string // kode singkat, mis. "ANO"
  category: string // kategori, mis. "Anorganik" | "Organik" | "B3" | "Residu"
  decompose: string // estimasi waktu terurai
  bin: string // warna tempat sampah, mis. "Kuning"
  binHex: string // warna hex tempat sampah, mis. "#F59E0B"
  tip: string // tips pengelolaan
  confidence: number // 0–100
  low_confidence: boolean // true jika di bawah ambang yakin
  unknown: boolean // true jika sampah tidak dikenali
  lvl2: string
  lvl1: string
  top3: Top3Item[]
  method: string // mis. "TTA"
}

/** Response dari endpoint GET /health. */
export interface HealthResponse {
  status: string
  model: string
  classes: number
  kategori: string[]
}

export type ClassifyMode = 'upload' | 'webcam'
export type BackendStatus = 'checking' | 'online' | 'offline'
