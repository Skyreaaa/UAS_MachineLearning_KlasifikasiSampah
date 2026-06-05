# Design

Sistem visual TrashTrack. Register: **product** (UI melayani tugas). Arah:
**editorial / tipografi tegas** — dipimpin huruf, kontras tinggi, warna minim,
garis hairline alih-alih kartu.

## Theme

Dual mode (terang/gelap), berbasis class `.dark` pada `<html>`. Bukan
glassmorphism, bukan bayangan tebal. Permukaan polos (kertas/tinta), dipisah
oleh garis 1px. Sudut tegas (radius kecil 4–8px), tanpa blob dekoratif.

## Color (OKLCH)

Strategi: **Restrained** — netral murni (chroma 0) + satu aksen.

| Role | Light | Dark |
|---|---|---|
| Background | `oklch(1 0 0)` (putih murni) | `oklch(0.145 0 0)` (near-black) |
| Ink (teks utama) | `neutral-900` | `neutral-100` |
| Ink muted (sekunder) | `neutral-600` (≥4.5:1) | `neutral-400` |
| Line (hairline) | `neutral-200` / `neutral-300` | `neutral-800` / `neutral-700` |
| Accent (crimson) | `oklch(0.52 0.205 27)` | `oklch(0.66 0.19 27)` |
| On-accent | putih | near-black |

Aksen dipakai hanya untuk: aksi primer, tab aktif, tautan, cincin fokus, dan
garis aksen editorial. Bukan dekorasi.

**Warna fungsional kategori** (dari backend, dipertahankan, selalu + label/ikon):
anorganik `#F59E0B` (kuning), organik `#10B981` (hijau), B3 `#EF4444` (merah),
residu `#6B7280` (abu).

## Typography

Dua keluarga pada sumbu kontras (serif display + sans body):

- **Display**: `Fraunces` (variable, opsz) — judul hero & nama kategori hasil.
- **Body/UI**: `Inter` — semua teks, label, tombol, data.

Aturan: skala rem tetap (bukan fluid untuk UI), rasio langkah ~1.2–1.3, kontras
bobot kuat (400 body, 600/700 heading, 900 hero). Hero clamp max ≤ 6rem.
Tracking display ≥ -0.02em. Tanpa body ALL CAPS. `text-wrap: balance` pada
heading, `pretty` pada prosa.

## Components

- **Masthead**: wordmark serif + mark aksen, garis bawah hairline; status backend
  (titik warna + teks) dan toggle tema di kanan.
- **Tabs (mode)**: teks dengan indikator garis bawah; tab aktif bergaris aksen.
- **Dropzone**: area bergaris putus-putus 1px, instruksi jelas, bukan kartu.
- **Tombol**: primer = aksen solid + teks on-accent; sekunder = outline hairline.
  Radius kecil konsisten. State: default/hover/focus/active/disabled.
- **Hasil**: nama kategori serif besar, angka keyakinan besar + bar, daftar data
  bergaris (waktu terurai, tempat sampah), tips sebagai prosa, swatch warna +
  label.
- **State lengkap**: empty (mengajari), loading (progress determinate), error,
  low-confidence (peringatan + kemungkinan lain).

## Layout

- Lebar konten maks ~72rem, kiri-rata (editorial), bukan center semua.
- Area kerja: dua kolom dipisah garis vertikal (input | hasil); menumpuk di
  ponsel dengan garis horizontal.
- Spasi bervariasi untuk ritme; whitespace berlimpah.

## Motion

150–250ms, menyampaikan keadaan (hover, fokus, reveal hasil, progress). Tanpa
orkestrasi page-load. Ease-out. Semua punya alternatif `prefers-reduced-motion`.
