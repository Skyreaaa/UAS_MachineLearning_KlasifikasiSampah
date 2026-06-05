# Product

## Register

product

## Users

Masyarakat umum berbahasa Indonesia yang ingin tahu cara memilah dan membuang
sampah dengan benar. Konteksnya: di rumah atau di lokasi, sering lewat ponsel,
sambil memegang satu objek sampah dan ragu ini masuk kategori atau tempat sampah
mana. Juga dipakai sebagai demo proyek UAS Machine Learning, jadi sebagian
pengguna adalah dosen/penguji yang menilai kualitas dan kejujuran sistem.

## Product Purpose

Memotret atau mengunggah satu gambar sampah, lalu model (YOLO11-cls) menentukan
kategorinya (anorganik, organik, B3, residu), warna tempat sampah, estimasi waktu
terurai, dan tips pengelolaan. Sukses = pengguna langsung paham "ini sampah apa
dan harus dibuang ke mana" dalam hitungan detik, dengan keyakinan model yang
transparan, termasuk jujur saat model tidak yakin.

## Brand Personality

Tegas, jelas, dapat dipercaya. Terasa seperti terbitan editorial yang serius,
bukan mainan eco yang lucu dan bukan SaaS klise. Tiga kata: editorial, presisi,
jujur.

## Anti-references

- Glassmorphism: kartu kaca buram dan teks gradient. Dilarang.
- Refleks "aplikasi sampah = hijau lembut, daun, gelembung" yang generik.
- Tampak template AI: eyebrow uppercase di tiap section, grid kartu seragam,
  pola hero-metric SaaS, blob dekoratif mengambang.
- Dua ekstrem yang sama buruknya: terlalu ramai/warna-warni, atau terlalu polos
  dan membosankan.

## Design Principles

1. Tipografi memimpin. Hierarki dibangun dari skala dan bobot huruf, bukan dari
   kartu dan bayangan.
2. Kontras dulu. Teks selalu mudah dibaca (WCAG AA); warna dipakai untuk makna,
   bukan dekorasi.
3. Jujur soal keyakinan. Tampilkan tingkat kepercayaan dan kemungkinan lain;
   jangan paksa hasil saat model ragu.
4. Warna fungsional punya makna tetap. Warna kategori dan tempat sampah selalu
   didampingi label dan ikon, supaya ramah buta warna.
5. Gerak menyampaikan keadaan, bukan pertunjukan. Singkat, dan selalu punya
   alternatif untuk reduced-motion.

## Accessibility & Inclusion

WCAG AA untuk kontras teks (body minimal 4.5:1, teks besar/tebal minimal 3:1).
Tidak pernah mengandalkan warna saja untuk menyampaikan kategori (selalu ada
teks dan ikon pendamping). Mendukung `prefers-reduced-motion`. Responsif di
ponsel dan desktop. Fokus keyboard selalu terlihat; target sentuh memadai.
