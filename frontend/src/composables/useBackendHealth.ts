import { ref } from 'vue'
import type { BackendStatus, HealthResponse } from '../types'

/** Memeriksa apakah backend FastAPI hidup lewat GET /api/health. */
export function useBackendHealth() {
  const status = ref<BackendStatus>('checking')
  const info = ref<HealthResponse | null>(null)

  async function check(): Promise<void> {
    status.value = 'checking'
    try {
      const res = await fetch('/api/health')
      if (!res.ok) throw new Error(`HTTP ${res.status}`)
      info.value = (await res.json()) as HealthResponse
      status.value = 'online'
    } catch {
      info.value = null
      status.value = 'offline'
    }
  }

  return { status, info, check }
}
