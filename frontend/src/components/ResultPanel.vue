<script setup lang="ts">
import { computed } from 'vue'
import type { ClassifyResult } from '../types'
import { categoryColor, confidenceColor } from '../lib/categories'
import StatCard from './StatCard.vue'

const props = defineProps<{
  result: ClassifyResult | null
  scanning: boolean
  error: string | null
  backendOffline: boolean
}>()

const confidenceLabel = computed(() => {
  const c = props.result?.confidence ?? 0
  if (c >= 90) return 'tinggi'
  if (c >= 70) return 'sedang'
  return 'rendah'
})
</script>

<template>
  <!-- Error -->
  <div v-if="error" class="flex min-h-72 flex-col justify-center gap-2">
    <h2 class="font-display text-2xl font-semibold tracking-tight text-neutral-900 dark:text-neutral-50">
      Gagal mengklasifikasi
    </h2>
    <p class="max-w-md text-sm leading-relaxed text-neutral-600 dark:text-neutral-400">{{ error }}</p>
    <p v-if="backendOffline" class="max-w-md text-sm text-neutral-500 dark:text-neutral-500">
      Pastikan backend berjalan, lalu coba lagi:
    </p>
    <code
      v-if="backendOffline"
      class="mt-1 block w-fit rounded bg-neutral-100 px-2.5 py-1.5 font-mono text-xs text-neutral-700 dark:bg-neutral-900 dark:text-neutral-300"
    >
      uvicorn backend.main:app --reload --port 8000
    </code>
  </div>

  <!-- Memproses -->
  <div v-else-if="scanning" class="flex min-h-72 flex-col justify-center gap-3">
    <div class="flex items-center gap-3">
      <svg viewBox="0 0 24 24" class="animate-spin-slow h-5 w-5 text-accent" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round">
        <path d="M21 12a9 9 0 1 1-6.2-8.5" />
      </svg>
      <h2 class="font-display text-2xl font-semibold tracking-tight text-neutral-900 dark:text-neutral-50">
        Menganalisis gambar
      </h2>
    </div>
    <p class="text-sm text-neutral-500 dark:text-neutral-400">
      Model sedang menentukan kategori sampah dan tingkat keyakinannya.
    </p>
  </div>

  <!-- Hasil -->
  <div v-else-if="result" :key="result.code + result.confidence" class="animate-fade-in flex flex-col gap-6">
    <!-- Nama + keyakinan -->
    <div class="flex items-start justify-between gap-4">
      <div class="min-w-0">
        <p class="text-sm font-medium text-neutral-500 dark:text-neutral-400">Terdeteksi sebagai</p>
        <h2 class="font-display mt-1 text-3xl font-semibold tracking-tight text-balance text-neutral-900 sm:text-4xl dark:text-neutral-50">
          {{ result.label }}
        </h2>
        <p class="mt-2 inline-flex items-center gap-2 text-sm font-medium text-neutral-700 dark:text-neutral-300">
          <span class="h-2.5 w-2.5 rounded-full" :style="{ background: categoryColor(result.category) }" aria-hidden="true" />
          Kategori {{ result.category }}
        </p>
      </div>
      <div class="shrink-0 text-right">
        <p class="font-display text-4xl font-semibold tabular-nums text-neutral-900 sm:text-5xl dark:text-neutral-50">
          {{ result.confidence.toFixed(0) }}<span class="text-2xl text-neutral-400 dark:text-neutral-500">%</span>
        </p>
        <p class="text-xs font-medium text-neutral-500 dark:text-neutral-400">keyakinan {{ confidenceLabel }}</p>
      </div>
    </div>

    <!-- Bar keyakinan -->
    <div class="h-1.5 w-full overflow-hidden rounded-full bg-neutral-200 dark:bg-neutral-800">
      <div
        class="h-full rounded-full transition-[width] duration-500"
        :style="{ width: result.confidence + '%', background: confidenceColor(result.confidence) }"
      />
    </div>

    <!-- Peringatan keyakinan rendah -->
    <div
      v-if="result.low_confidence"
      class="rounded-md border border-amber-300 bg-amber-50 px-4 py-3 dark:border-amber-500/40 dark:bg-amber-500/10"
    >
      <p class="text-sm font-semibold text-amber-800 dark:text-amber-300">Model kurang yakin</p>
      <p class="mt-0.5 text-sm leading-relaxed text-amber-800/90 dark:text-amber-200/80">
        Objek mungkin tidak ada di dataset, atau gambar kurang jelas. Coba foto ulang dengan
        pencahayaan yang lebih baik dan objek lebih dekat.
      </p>
    </div>

    <!-- Kemungkinan lain -->
    <div v-if="result.low_confidence && result.top3.length > 1" class="flex flex-col gap-2.5">
      <p class="text-xs font-medium text-neutral-500 dark:text-neutral-400">Kemungkinan lain</p>
      <div v-for="alt in result.top3.slice(1)" :key="alt.lvl2" class="flex items-center gap-3">
        <span class="w-24 shrink-0 text-sm text-neutral-700 capitalize dark:text-neutral-300">
          {{ alt.lvl2.replace(/_/g, ' ') }}
        </span>
        <div class="h-1 flex-1 overflow-hidden rounded-full bg-neutral-200 dark:bg-neutral-800">
          <div class="h-full rounded-full bg-neutral-400 dark:bg-neutral-600" :style="{ width: alt.confidence + '%' }" />
        </div>
        <span class="w-10 shrink-0 text-right text-sm font-medium tabular-nums text-neutral-600 dark:text-neutral-400">
          {{ alt.confidence.toFixed(0) }}%
        </span>
      </div>
    </div>

    <!-- Data -->
    <dl class="grid grid-cols-2 gap-3">
      <StatCard label="Waktu terurai" :value="result.decompose" />
      <StatCard label="Tempat sampah" :value="result.bin" :swatch="result.binHex" />
    </dl>

    <!-- Tips -->
    <div class="border-t border-neutral-200 pt-4 dark:border-neutral-800">
      <p class="text-sm font-semibold text-neutral-900 dark:text-neutral-100">Tips pengelolaan</p>
      <p class="mt-1.5 text-sm leading-relaxed text-pretty text-neutral-600 dark:text-neutral-300">{{ result.tip }}</p>
    </div>
  </div>

  <!-- Kosong -->
  <div v-else class="flex min-h-72 flex-col justify-center gap-2">
    <h2 class="font-display text-2xl font-semibold tracking-tight text-neutral-400 dark:text-neutral-600">
      Belum ada hasil
    </h2>
    <p class="max-w-md text-sm leading-relaxed text-neutral-500 dark:text-neutral-400">
      Unggah gambar sampah atau ambil foto lewat webcam. Hasil klasifikasi, tingkat keyakinan, dan
      tips pembuangan akan muncul di sini.
    </p>
  </div>
</template>
