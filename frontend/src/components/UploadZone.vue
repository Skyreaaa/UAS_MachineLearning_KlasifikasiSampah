<script setup lang="ts">
import { ref } from 'vue'
import type { ClassifyResult } from '../types'
import { categoryColor } from '../lib/categories'

const props = defineProps<{
  previewUrl: string | null
  scanning: boolean
  progress: number
  result: ClassifyResult | null
}>()

const emit = defineEmits<{
  (e: 'file', file: File): void
  (e: 'reset'): void
}>()

const inputRef = ref<HTMLInputElement | null>(null)
const dragging = ref(false)

function openPicker(): void {
  if (!props.previewUrl) inputRef.value?.click()
}

function onSelect(e: Event): void {
  const target = e.target as HTMLInputElement
  const file = target.files?.[0]
  if (file) emit('file', file)
  target.value = '' // izinkan memilih file yang sama lagi
}

function onDrop(e: DragEvent): void {
  dragging.value = false
  const file = e.dataTransfer?.files?.[0]
  if (file && file.type.startsWith('image/')) emit('file', file)
}
</script>

<template>
  <section class="flex flex-col gap-4">
    <!-- Dropzone -->
    <div
      @click="openPicker"
      @drop.prevent="onDrop"
      @dragover.prevent="dragging = true"
      @dragenter.prevent="dragging = true"
      @dragleave.prevent="dragging = false"
      class="relative flex min-h-72 items-center justify-center overflow-hidden rounded-md transition-colors"
      :class="[
        previewUrl
          ? 'cursor-default border border-neutral-200 dark:border-neutral-800'
          : 'cursor-pointer border border-dashed',
        !previewUrl && dragging
          ? 'border-accent bg-accent/5'
          : !previewUrl
            ? 'border-neutral-300 hover:border-accent dark:border-neutral-700'
            : '',
      ]"
    >
      <input ref="inputRef" type="file" accept="image/*" class="hidden" @change="onSelect" />

      <!-- Kosong -->
      <div v-if="!previewUrl" class="flex flex-col items-center gap-3 px-8 py-12 text-center">
        <svg viewBox="0 0 24 24" class="h-8 w-8 text-neutral-400 dark:text-neutral-500" fill="none" stroke="currentColor" stroke-width="1.75" stroke-linecap="round" stroke-linejoin="round">
          <path d="M12 16V4M7 9l5-5 5 5" />
          <path d="M5 16v2a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2v-2" />
        </svg>
        <div>
          <p class="text-base font-semibold text-neutral-800 dark:text-neutral-100">Seret gambar ke sini</p>
          <p class="mt-1 text-sm text-neutral-500 dark:text-neutral-400">atau klik untuk memilih dari perangkat</p>
        </div>
        <p class="text-xs text-neutral-400 dark:text-neutral-500">JPG, PNG, atau WEBP</p>
      </div>

      <!-- Preview -->
      <div v-else class="relative w-full">
        <img :src="previewUrl" alt="Pratinjau gambar yang diunggah" class="block h-72 w-full object-cover sm:h-80" />

        <!-- Overlay saat menganalisis -->
        <div v-if="scanning" class="absolute inset-0 flex flex-col items-center justify-center gap-3 bg-neutral-950/70">
          <p class="animate-pulse-soft text-sm font-medium text-white">Menganalisis {{ progress }}%</p>
          <div class="h-1 w-48 overflow-hidden rounded-full bg-white/25">
            <div class="h-full rounded-full bg-white transition-[width] duration-150" :style="{ width: progress + '%' }" />
          </div>
        </div>

        <!-- Chip hasil (titik warna + label, kontras tinggi) -->
        <div
          v-else-if="result"
          class="animate-fade-in absolute bottom-3 left-3 flex items-center gap-2 rounded-md bg-neutral-950/90 px-3 py-1.5 text-xs font-medium text-white"
        >
          <span class="h-2.5 w-2.5 rounded-full" :style="{ background: categoryColor(result.category) }" />
          <span>{{ result.label }}</span>
          <span class="text-white/60">{{ result.confidence.toFixed(0) }}%</span>
        </div>
      </div>
    </div>

    <!-- Reset -->
    <button
      v-if="previewUrl && !scanning"
      type="button"
      @click="emit('reset')"
      class="cursor-pointer self-start text-sm font-medium text-neutral-500 underline-offset-4 transition-colors hover:text-accent hover:underline dark:text-neutral-400"
    >
      Ganti gambar
    </button>
  </section>
</template>
