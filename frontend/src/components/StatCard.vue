<script setup lang="ts">
defineProps<{
  label: string
  value: string
  /** Swatch warna opsional sebelum nilai (mis. warna tempat sampah). */
  swatch?: string
}>()
</script>

<template>
  <div class="border border-neutral-200 p-4 dark:border-neutral-800">
    <dt class="text-xs font-medium text-neutral-500 dark:text-neutral-400">{{ label }}</dt>
    <dd class="mt-1.5 flex items-center gap-2 text-sm font-semibold text-neutral-900 dark:text-neutral-100">
      <span
        v-if="swatch"
        class="h-3.5 w-3.5 rounded-sm ring-1 ring-black/10 dark:ring-white/20"
        :style="{ background: swatch }"
        aria-hidden="true"
      />
      <span>{{ value }}</span>
    </dd>
  </div>
</template>
