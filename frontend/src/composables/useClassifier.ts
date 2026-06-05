import { ref } from 'vue'
import type { ClassifyResult } from '../types'

/**
 * Mengelola seluruh siklus klasifikasi:
 * preview gambar, progress, hasil, dan error.
 * Dibuat sekali di App.vue lalu state-nya diteruskan ke komponen anak.
 */
export function useClassifier() {
  const previewUrl = ref<string | null>(null)
  const result = ref<ClassifyResult | null>(null)
  const scanning = ref(false)
  const progress = ref(0)
  const error = ref<string | null>(null)

  let timer: ReturnType<typeof setInterval> | null = null
  let objectUrl: string | null = null

  function stopProgress(): void {
    if (timer !== null) {
      clearInterval(timer)
      timer = null
    }
  }

  // Progress bar simulasi — backend tidak streaming progress, jadi kita perkirakan.
  function startProgress(max = 88): void {
    let pct = 0
    timer = setInterval(() => {
      pct += Math.random() * 13
      if (pct >= max) {
        pct = max
        stopProgress()
      }
      progress.value = Math.round(Math.min(pct, max))
    }, 130)
  }

  function revokePreview(): void {
    if (objectUrl) {
      URL.revokeObjectURL(objectUrl)
      objectUrl = null
    }
  }

  async function classify(file: File | null | undefined): Promise<void> {
    if (!file) return

    stopProgress()
    revokePreview()

    objectUrl = URL.createObjectURL(file)
    previewUrl.value = objectUrl
    result.value = null
    error.value = null
    scanning.value = true
    progress.value = 0
    startProgress(88)

    try {
      const form = new FormData()
      form.append('file', file)

      const res = await fetch('/api/classify', { method: 'POST', body: form })
      if (!res.ok) {
        const body = (await res.json().catch(() => ({}))) as { detail?: string }
        throw new Error(body.detail ?? `Server error ${res.status}`)
      }

      const data = (await res.json()) as ClassifyResult
      stopProgress()
      progress.value = 100
      // Jeda singkat agar transisi progress 100% terlihat halus.
      await new Promise((resolve) => setTimeout(resolve, 280))
      result.value = data
    } catch (e) {
      error.value = e instanceof Error ? e.message : 'Gagal terhubung ke server'
      progress.value = 0
    } finally {
      stopProgress()
      scanning.value = false
    }
  }

  function reset(): void {
    stopProgress()
    revokePreview()
    previewUrl.value = null
    result.value = null
    error.value = null
    scanning.value = false
    progress.value = 0
  }

  return { previewUrl, result, scanning, progress, error, classify, reset }
}
