<script setup lang="ts">
import { computed } from 'vue'
import type { BackendStatus } from '../types'
import { useTheme } from '../composables/useTheme'

const props = defineProps<{
  status: BackendStatus
}>()

const { isDark, toggle } = useTheme()

const statusMeta = computed(() => {
  switch (props.status) {
    case 'online':
      return { dot: 'bg-emerald-600 dark:bg-emerald-400', label: 'Backend siap' }
    case 'offline':
      return { dot: 'bg-accent', label: 'Backend offline' }
    default:
      return { dot: 'bg-amber-500 animate-pulse-soft', label: 'Menghubungkan' }
  }
})
</script>

<template>
  <header class="border-b border-neutral-200 dark:border-neutral-800">
    <div class="mx-auto flex w-full max-w-6xl items-center gap-3 px-4 py-3.5 sm:px-6">
      <!-- Mark + wordmark -->
      <span class="h-5 w-5 shrink-0 bg-accent" aria-hidden="true" />
      <div class="flex items-baseline gap-2">
        <span class="font-display text-xl font-semibold tracking-tight text-neutral-900 dark:text-neutral-50">
          TrashTrack
        </span>
        <span class="hidden text-xs text-neutral-500 sm:inline dark:text-neutral-400">
          Klasifikasi Sampah
        </span>
      </div>

      <div class="flex-1" />

      <!-- Status backend (titik + teks, ramah buta warna) -->
      <span class="flex items-center gap-2 text-xs font-medium text-neutral-600 dark:text-neutral-300">
        <span class="h-2 w-2 rounded-full" :class="statusMeta.dot" />
        <span class="hidden sm:inline">{{ statusMeta.label }}</span>
      </span>

      <!-- Toggle tema -->
      <button
        type="button"
        @click="toggle"
        class="flex h-9 w-9 cursor-pointer items-center justify-center rounded-md border border-neutral-300 text-neutral-700 transition-colors hover:bg-neutral-100 dark:border-neutral-700 dark:text-neutral-300 dark:hover:bg-neutral-800"
        :aria-label="isDark ? 'Aktifkan mode terang' : 'Aktifkan mode gelap'"
        :title="isDark ? 'Mode terang' : 'Mode gelap'"
      >
        <svg v-if="isDark" viewBox="0 0 24 24" class="h-4.5 w-4.5" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
          <circle cx="12" cy="12" r="4" />
          <path d="M12 2v2M12 20v2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M2 12h2M20 12h2M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4" />
        </svg>
        <svg v-else viewBox="0 0 24 24" class="h-4.5 w-4.5" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
          <path d="M21 12.8A9 9 0 1 1 11.2 3a7 7 0 0 0 9.8 9.8Z" />
        </svg>
      </button>
    </div>
  </header>
</template>
