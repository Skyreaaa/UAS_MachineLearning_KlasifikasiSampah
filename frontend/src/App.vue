<script setup lang="ts">
import { computed, onMounted, ref, watch } from 'vue'
import type { ClassifyMode } from './types'
import { useClassifier } from './composables/useClassifier'
import { useBackendHealth } from './composables/useBackendHealth'
import AppNavbar from './components/AppNavbar.vue'
import ModeTabs from './components/ModeTabs.vue'
import UploadZone from './components/UploadZone.vue'
import WebcamCapture from './components/WebcamCapture.vue'
import ResultPanel from './components/ResultPanel.vue'

const { previewUrl, result, scanning, progress, error, classify, reset } = useClassifier()
const { status, check } = useBackendHealth()

const mode = ref<ClassifyMode>('upload')

onMounted(check)

// Reset state saat berpindah mode (Unggah ↔ Webcam)
watch(mode, () => reset())

function handleFile(file: File): void {
  classify(file)
}

const backendOffline = computed(() => status.value === 'offline')
</script>

<template>
  <div class="min-h-screen bg-white text-neutral-900 dark:bg-neutral-950 dark:text-neutral-100">
    <AppNavbar :status="status" />

    <main class="mx-auto w-full max-w-6xl">
      <!-- Hero -->
      <section class="px-4 py-12 sm:px-6 sm:py-16">
        <span class="block h-1 w-12 bg-accent" aria-hidden="true" />
        <h1
          class="font-display mt-6 max-w-3xl text-[2.6rem] leading-[1.04] font-semibold tracking-tight text-balance sm:text-6xl lg:text-7xl"
        >
          Foto sampahnya. Tahu cara buangnya.
        </h1>
        <p class="mt-5 max-w-xl text-base leading-relaxed text-neutral-600 sm:text-lg dark:text-neutral-400">
          Unggah gambar atau nyalakan webcam. Model mengenali kategori sampah, tempat
          pembuangan, estimasi waktu terurai, dan tips pengelolaannya, lengkap dengan
          tingkat keyakinannya.
        </p>
      </section>

      <!-- Area kerja: input | hasil, dipisah garis -->
      <section class="grid grid-cols-1 border-t border-neutral-200 md:grid-cols-2 dark:border-neutral-800">
        <!-- Kolom input -->
        <div class="border-b border-neutral-200 px-4 py-8 sm:px-6 md:border-r md:border-b-0 md:py-10 dark:border-neutral-800">
          <ModeTabs v-model="mode" />
          <div class="mt-7">
            <UploadZone
              v-if="mode === 'upload'"
              :preview-url="previewUrl"
              :scanning="scanning"
              :progress="progress"
              :result="result"
              @file="handleFile"
              @reset="reset"
            />
            <WebcamCapture v-else @file="handleFile" />
          </div>
        </div>

        <!-- Kolom hasil -->
        <div class="px-4 py-8 sm:px-6 md:py-10">
          <ResultPanel
            :result="result"
            :scanning="scanning"
            :error="error"
            :backend-offline="backendOffline"
          />
        </div>
      </section>

      <!-- Footer -->
      <footer class="border-t border-neutral-200 px-4 py-6 sm:px-6 dark:border-neutral-800">
        <div class="flex flex-wrap items-center gap-x-3 gap-y-1 text-xs text-neutral-500 dark:text-neutral-500">
          <span>TrashTrack · Proyek UAS Machine Learning</span>
          <span aria-hidden="true">·</span>
          <span>Model YOLO11-cls, 4 kategori: anorganik, organik, B3, residu</span>
        </div>
      </footer>
    </main>
  </div>
</template>
