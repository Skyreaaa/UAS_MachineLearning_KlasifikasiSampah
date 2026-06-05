<script setup lang="ts">
import { onUnmounted, ref } from 'vue'

const emit = defineEmits<{
  (e: 'file', file: File): void
}>()

type CamState = 'idle' | 'loading' | 'active' | 'error'

const videoRef = ref<HTMLVideoElement | null>(null)
const canvasRef = ref<HTMLCanvasElement | null>(null)

const camState = ref<CamState>('idle')
const errorMsg = ref('')
const cameras = ref<MediaDeviceInfo[]>([])
const currentDeviceId = ref<string | null>(null)
const countdown = ref(0)
const flash = ref(false)

let stream: MediaStream | null = null
let countdownTimer: ReturnType<typeof setInterval> | null = null

function stopStream(): void {
  stream?.getTracks().forEach((t) => t.stop())
  stream = null
}

function clearCountdown(): void {
  if (countdownTimer !== null) {
    clearInterval(countdownTimer)
    countdownTimer = null
  }
  countdown.value = 0
}

async function listCameras(): Promise<void> {
  try {
    const devices = await navigator.mediaDevices.enumerateDevices()
    cameras.value = devices.filter((d) => d.kind === 'videoinput')
  } catch {
    /* enumerasi gagal — abaikan */
  }
}

async function start(deviceId?: string): Promise<void> {
  camState.value = 'loading'
  errorMsg.value = ''
  stopStream()

  try {
    const video: MediaTrackConstraints = deviceId
      ? { deviceId: { exact: deviceId }, width: { ideal: 1280 }, height: { ideal: 720 } }
      : { facingMode: 'environment', width: { ideal: 1280 }, height: { ideal: 720 } }

    stream = await navigator.mediaDevices.getUserMedia({ video, audio: false })

    if (videoRef.value) {
      videoRef.value.srcObject = stream
      await videoRef.value.play().catch(() => undefined)
    }
    currentDeviceId.value = stream.getVideoTracks()[0]?.getSettings().deviceId ?? deviceId ?? null
    camState.value = 'active'
    await listCameras()
  } catch (err) {
    const name = err instanceof DOMException ? err.name : ''
    errorMsg.value =
      name === 'NotAllowedError'
        ? 'Akses kamera ditolak. Izinkan kamera di pengaturan browser.'
        : name === 'NotFoundError'
          ? 'Kamera tidak ditemukan di perangkat ini.'
          : 'Gagal membuka kamera.'
    camState.value = 'error'
  }
}

function stop(): void {
  clearCountdown()
  stopStream()
  if (videoRef.value) videoRef.value.srcObject = null
  camState.value = 'idle'
}

function snapshot(): void {
  const video = videoRef.value
  const canvas = canvasRef.value
  if (!video || !canvas) return
  const w = video.videoWidth
  const h = video.videoHeight
  if (!w || !h) return

  canvas.width = w
  canvas.height = h
  const ctx = canvas.getContext('2d')
  if (!ctx) return
  ctx.drawImage(video, 0, 0, w, h)

  flash.value = true
  setTimeout(() => (flash.value = false), 160)

  canvas.toBlob(
    (blob) => {
      if (blob) emit('file', new File([blob], `webcam-${Date.now()}.jpg`, { type: 'image/jpeg' }))
    },
    'image/jpeg',
    0.95,
  )
}

function startCountdown(): void {
  if (camState.value !== 'active' || countdown.value > 0) return
  countdown.value = 3
  countdownTimer = setInterval(() => {
    countdown.value -= 1
    if (countdown.value <= 0) {
      clearCountdown()
      snapshot()
    }
  }, 1000)
}

function shortLabel(label: string, index: number): string {
  const name = label || `Kamera ${index + 1}`
  return name.length > 24 ? name.slice(0, 24) + '…' : name
}

onUnmounted(stop)
</script>

<template>
  <section class="flex flex-col gap-4">
    <!-- Frame kamera 4:3 -->
    <div class="relative w-full overflow-hidden rounded-md border border-neutral-200 bg-neutral-950 dark:border-neutral-800" style="aspect-ratio: 4 / 3">
      <video
        ref="videoRef"
        autoplay
        playsinline
        muted
        class="absolute inset-0 h-full w-full object-cover transition-opacity duration-200"
        :class="camState === 'active' ? 'opacity-100' : 'opacity-0'"
      />
      <canvas ref="canvasRef" class="hidden" />

      <!-- Kotak panduan -->
      <div v-if="camState === 'active' && countdown === 0" class="pointer-events-none absolute inset-0 flex items-center justify-center">
        <div class="rounded border-2 border-dashed border-white/70" style="width: 64%; height: 64%" />
        <span class="absolute bottom-3 left-1/2 -translate-x-1/2 rounded bg-neutral-950/70 px-2.5 py-1 text-[11px] font-medium text-white">
          Posisikan sampah di dalam kotak
        </span>
      </div>

      <!-- Hitung mundur -->
      <div v-if="countdown > 0" class="pointer-events-none absolute inset-0 flex items-center justify-center bg-neutral-950/45">
        <span class="font-display text-7xl font-semibold text-white">{{ countdown }}</span>
      </div>

      <!-- Kilat -->
      <div v-if="flash" class="pointer-events-none absolute inset-0 z-20 bg-white/80" />

      <!-- Idle -->
      <div v-if="camState === 'idle'" class="absolute inset-0 flex flex-col items-center justify-center gap-2 text-center">
        <svg viewBox="0 0 24 24" class="h-10 w-10 text-white/45" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round">
          <path d="M23 7l-7 5 7 5V7z" />
          <rect x="1" y="5" width="15" height="14" rx="2" ry="2" />
        </svg>
        <p class="text-sm font-medium text-white/70">Kamera belum aktif</p>
      </div>

      <!-- Loading -->
      <div v-else-if="camState === 'loading'" class="absolute inset-0 flex flex-col items-center justify-center gap-3">
        <svg viewBox="0 0 24 24" class="animate-spin-slow h-8 w-8 text-white" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round">
          <path d="M21 12a9 9 0 1 1-6.2-8.5" />
        </svg>
        <p class="text-sm font-medium text-white/80">Membuka kamera</p>
      </div>

      <!-- Error -->
      <div v-else-if="camState === 'error'" class="absolute inset-0 flex items-center justify-center px-6">
        <p class="max-w-xs text-center text-sm text-white">{{ errorMsg }}</p>
      </div>
    </div>

    <!-- Pilih kamera -->
    <div v-if="camState === 'active' && cameras.length > 1" class="flex gap-2 overflow-x-auto pb-0.5">
      <button
        v-for="(cam, i) in cameras"
        :key="cam.deviceId || i"
        type="button"
        @click="start(cam.deviceId)"
        class="shrink-0 cursor-pointer rounded-md border px-3 py-1.5 text-xs font-medium transition-colors"
        :class="
          cam.deviceId === currentDeviceId
            ? 'border-accent text-accent'
            : 'border-neutral-300 text-neutral-600 hover:bg-neutral-100 dark:border-neutral-700 dark:text-neutral-300 dark:hover:bg-neutral-800'
        "
      >
        {{ shortLabel(cam.label, i) }}
      </button>
    </div>

    <!-- Tombol aksi -->
    <div class="flex gap-2.5">
      <button
        v-if="camState === 'idle' || camState === 'error'"
        type="button"
        @click="start()"
        class="bg-accent text-on-accent hover:bg-accent-hover flex-1 cursor-pointer rounded-md py-2.5 text-sm font-semibold transition-colors"
      >
        Buka kamera
      </button>

      <template v-if="camState === 'active'">
        <button
          type="button"
          @click="snapshot"
          :disabled="countdown > 0"
          class="bg-accent text-on-accent hover:bg-accent-hover flex-1 cursor-pointer rounded-md py-2.5 text-sm font-semibold transition-colors disabled:opacity-50"
        >
          Ambil foto
        </button>
        <button
          type="button"
          @click="startCountdown"
          :disabled="countdown > 0"
          class="cursor-pointer rounded-md border border-neutral-300 px-4 py-2.5 text-sm font-medium text-neutral-700 transition-colors hover:bg-neutral-100 disabled:opacity-50 dark:border-neutral-700 dark:text-neutral-300 dark:hover:bg-neutral-800"
          title="Ambil foto dengan hitung mundur 3 detik"
        >
          Timer 3s
        </button>
        <button
          type="button"
          @click="stop"
          class="cursor-pointer rounded-md border border-neutral-300 px-4 py-2.5 text-sm text-neutral-700 transition-colors hover:bg-neutral-100 dark:border-neutral-700 dark:text-neutral-300 dark:hover:bg-neutral-800"
          aria-label="Tutup kamera"
          title="Tutup kamera"
        >
          Tutup
        </button>
      </template>
    </div>
  </section>
</template>
