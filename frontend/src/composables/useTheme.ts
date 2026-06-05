import { computed, ref, watch } from 'vue'
import type { Ref } from 'vue'

type Theme = 'light' | 'dark'

const STORAGE_KEY = 'trashtrack-theme'

function getInitial(): Theme {
  try {
    const saved = localStorage.getItem(STORAGE_KEY)
    if (saved === 'light' || saved === 'dark') return saved
    return window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light'
  } catch {
    return 'light'
  }
}

function apply(value: Theme): void {
  document.documentElement.classList.toggle('dark', value === 'dark')
  try {
    localStorage.setItem(STORAGE_KEY, value)
  } catch {
    /* localStorage tidak tersedia — abaikan */
  }
}

// State tunggal (singleton) — dibagi ke semua komponen yang memakai composable ini.
const theme: Ref<Theme> = ref(getInitial())
const isDark = computed(() => theme.value === 'dark')

apply(theme.value)
watch(theme, apply)

export function useTheme() {
  const toggle = (): void => {
    theme.value = theme.value === 'dark' ? 'light' : 'dark'
  }
  return { theme, isDark, toggle }
}
