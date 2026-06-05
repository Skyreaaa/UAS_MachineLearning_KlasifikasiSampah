<script setup lang="ts">
import type { ClassifyMode } from '../types'

const model = defineModel<ClassifyMode>({ required: true })

const tabs: ReadonlyArray<{ id: ClassifyMode; label: string }> = [
  { id: 'upload', label: 'Unggah Foto' },
  { id: 'webcam', label: 'Webcam' },
]
</script>

<template>
  <div class="flex gap-7 border-b border-neutral-200 dark:border-neutral-800">
    <button
      v-for="tab in tabs"
      :key="tab.id"
      type="button"
      @click="model = tab.id"
      class="-mb-px cursor-pointer border-b-2 pt-1 pb-3 text-sm font-medium transition-colors"
      :class="
        model === tab.id
          ? 'border-accent text-neutral-900 dark:text-neutral-50'
          : 'border-transparent text-neutral-500 hover:text-neutral-800 dark:text-neutral-400 dark:hover:text-neutral-100'
      "
      :aria-pressed="model === tab.id"
    >
      {{ tab.label }}
    </button>
  </div>
</template>
