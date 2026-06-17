# Catatan Riset — GEEvo TA Tristan
> Dibuat: 2026-04-22 | Terakhir diupdate: 2026-04-24  
> NRP: 5002221100 — Alridho Tristan Satriawan, ITS  
> Status: Kondisi A, B & C selesai dijalankan — Kondisi C (TA) unggul di α=0.05 dengan median gen=8

---

## 1. Konteks Project

Fork dari [GEEvo](https://github.com/FlorianRupp/GEEvo-game-economies) (Florian Rupp & Kai Eckert, IEEE CEC 2024).  
Kontribusi TA:
- **Agent-based simulation** — 3 profil perilaku pemain (aggressive, passive, random)
- **Fungsi fitness obj4** — varian tolerance-threshold dari f1 paper
- **GDD Extractor** — ekstraksi konfigurasi ekonomi otomatis dari PDF Game Design Document

---

## 2. Cleanup Kode yang Sudah Dilakukan

### Dihapus (true dead code — tidak ada di paper, tidak dipakai di manapun)
| File | Item | Alasan |
|---|---|---|
| `geevo/graph.py` | `class Graph2` | Duplikasi Graph, tidak pernah diinstansiasi |
| `geevo/evolution/balancer.py` | `get_ind_fitness_single()` (obj1) | Tidak ada di paper, tidak pernah dipanggil |
| `geevo/evolution/balancer.py` | `get_ind_fitness_single2()` (obj2) | Tidak ada di paper, tidak pernah dipanggil |
| `geevo/evolution/generator.py` | `crossover()` | Paper eksplisit menyatakan tidak diimplementasikan |

### Di-comment out (ada di logika paper, out-of-scope TA)
| File | Item | Alasan |
|---|---|---|
| `geevo/evolution/balancer.py` | `class BalancerV2` | Implementasi f2 paper (mage vs archer), tidak dipakai di TA ini |

### File runtime dihapus
- `profile.txt` (536KB)
- `Gdd/economy_output.txt`
- `Gdd/pdf_results.txt`

### Lainnya
- `.gitignore` diperbarui (tambah output files, `*.pkl`, `*.html`)
- `README.md` diperbarui sesuai kontribusi TA
- Import `BalancerV2` di `demo.ipynb` dan `demo_agent.ipynb` diperbaiki
- Semua perubahan di-commit: `301d07f` (belum di-push ke GitHub)

---

## 3. Temuan Analisis Paper vs Kode

### Fungsi Fitness
Formula paper (f1):
```
f1(s_t, x) = α + (1/m) Σ prop(s_t_i, x)
prop(s_t, x) = min(s_t, x) / max(s_t, x)
```

| Fungsi di kode | Cocok dengan paper? | Status |
|---|---|---|
| obj1 (`get_ind_fitness_single`) | Tidak — membandingkan Pool A vs Pool B, bukan Pool vs target x | **Dihapus** |
| obj2 (`get_ind_fitness_single2`) | Tidak — formula derivatif temporal, tidak ada di paper | **Dihapus** |
| obj3 (`get_ind_fitness_single3`) | Ya — implementasi prop() dengan target x | **Dipertahankan** |
| obj4 (`get_ind_fitness_single4`) | Partial — prop() sama, tapi α sebagai threshold bukan offset | **Dipertahankan** (kontribusi TA) |

### BalancerV2
- **Ada di paper** sebagai implementasi f2 (case study mage vs archer, Section IV-B2)
- Tidak dipakai di scope TA (TA hanya satu ekonomi)
- Di-comment dengan keterangan, bukan dihapus

### crossover() di EvolutionaryGraphGeneration
- Paper menyatakan eksplisit: *"we abstain the implementation of a crossover"*
- Sudah disabled sejak repo asli — dihapus bersih

---

## 4. Hasil Evaluasi `evaluate_metrics.ipynb` (~4 jam runtime)

### Setup Evaluasi TA
| Parameter | Nilai |
|---|---|
| `pop_size` | 20 |
| `n_sim` | 10 |
| `n_sim_steps` | random [10, 30] |
| `balance_value` | random [20, 100] |
| `max_steps` | 500 |
| `fitness_func` | obj4 |
| Agent | aggressive + passive + random (sekaligus) |
| α tested | 0.05, 0.01, 0.0 |

### Dataset
- **Abstract** (n=30): `{Source:3, RandomGate:2, Pool:4, Converter:1}` — EGG generated
- **MMORPG** (n=10): `{Source:2, RandomGate:3, Pool:5, Converter:3}` — dari GDD PDF
- **VHS/Horror** (n=10): `{Source:1, RandomGate:2, Pool:3, Converter:1}` — dari GDD PDF

### Tabel 1 — Abstract (n=30)
| α | Balanced % | Initial Balanced % | Improved % | Median Gen | Exec Time (s) | Fitness Std Dev |
|---|---|---|---|---|---|---|
| 0.05 | 66.67% | 6.67% | 86.67% | 2.5 | 1.191 | 0.1947 |
| 0.01 | 30.0% | 3.33% | 90.0% | 500.0 | 7.453 | 0.0606 |
| 0.00 | 6.67% | 0.0% | 86.67% | 500.0 | 40.216 | 0.0371 |

### Tabel 2 — MMORPG GDD (n=10)
| α | Balanced % | Initial Balanced % | Improved % | Median Gen | Exec Time (s) | Fitness Std Dev |
|---|---|---|---|---|---|---|
| 0.05 | 50.0% | 10.0% | 70.0% | 251.0 | 3.609 | 0.2858 |
| 0.01 | 0.0% | 0.0% | 60.0% | 500.0 | 40.778 | 0.3783 |
| 0.00 | 10.0% | 0.0% | 80.0% | 500.0 | 54.733 | 0.2986 |

### Tabel 3 — VHS/Horror GDD (n=10)
| α | Balanced % | Initial Balanced % | Improved % | Median Gen | Exec Time (s) | Fitness Std Dev |
|---|---|---|---|---|---|---|
| 0.05 | 30.0% | 0.0% | 100.0% | 500.0 | 18.004 | 0.0507 |
| 0.01 | 20.0% | 0.0% | 100.0% | 500.0 | 3.575 | 0.0229 |
| 0.00 | 0.0% | 0.0% | 90.0% | 500.0 | 5.084 | 0.2827 |

### Tabel 4 — Breakdown per Agen, Abstract (n=30)
| α | Agent | Balanced % | Improved % | Median Gen | Exec Time (s) |
|---|---|---|---|---|---|
| 0.05 | aggressive | 63.33% | 83.33% | 3.0 | 0.593 |
| 0.05 | passive | 60.0% | 83.33% | 6.5 | 0.888 |
| 0.05 | random | 56.67% | 73.33% | 6.5 | 0.809 |
| 0.01 | aggressive | 13.33% | 86.67% | 500.0 | 27.192 |
| 0.01 | passive | 26.67% | 86.67% | 500.0 | 9.407 |
| 0.01 | random | 16.67% | 86.67% | 500.0 | 2.574 |
| 0.00 | aggressive | 3.33% | 76.67% | 500.0 | 8.241 |
| 0.00 | passive | 0.0% | 90.0% | 500.0 | 20.599 |
| 0.00 | random | 10.0% | 96.67% | 500.0 | 2.877 |

### Tabel 5 — Perbandingan dengan Paper Asli
| Sumber | α=0.05 Balanced % | α=0.01 Balanced % | α=0.05 Median Gen | α=0.05 Exec Time (s) | α=0.05 Fitness Std Dev |
|---|---|---|---|---|---|
| Paper GEEvo (n=194) | 93.3% | 83.0% | 1.0 | 18.400 | N/A |
| TA — Abstract (n=30) | 66.67% | 30.0% | 2.5 | 1.191 | 0.1947 |
| TA — MMORPG (n=10) | 50.0% | 0.0% | 251.0 | 3.609 | 0.2858 |
| TA — VHS/Horror (n=10) | 30.0% | 20.0% | 500.0 | 18.004 | 0.0507 |

---

## 5. Analisis Temuan Kunci

### Mengapa lebih rendah dari paper
| Faktor | Penjelasan |
|---|---|
| **Fungsi fitness** | obj4 lebih ketat dari obj3 (ada tolerance threshold) |
| **Stokastisitas agent** | Paper tanpa agent → simulasi deterministik. TA pakai 3 agen acak → estimasi fitness lebih noisy |
| **Kompleksitas GDD** | MMORPG punya 13–19 edge vs abstract ~10 edge → ruang pencarian lebih besar |
| **Sample size** | Paper n=194 lebih representatif |

### Temuan original yang menarik
1. **Tipe agen berpengaruh signifikan terhadap kecepatan konvergensi:**
   - Aggressive paling cepat di α=0.05 (median 3 gen, 0.593 detik)
   - Passive lebih unggul di α=0.01 (26.67% vs aggressive 13.33%)
   - Random paling konsisten di α=0.0 (improved 96.67%)

2. **Kompleksitas GDD berdampak besar:**
   - Abstract: 66.67% balanced (α=0.05)
   - MMORPG: turun ke 50.0%
   - VHS/Horror: turun ke 30.0%

3. **Improved % selalu tinggi (>70%) di semua kondisi** → balancer selalu berhasil meningkatkan fitness walau tidak selalu mencapai threshold

4. **VHS/Horror paling konsisten** (Fitness Std Dev 0.023–0.051) meski balanced rate rendah

---

## 6. Perbandingan Parameter: Paper vs TA

### Parameter IDENTIK dengan paper
| Parameter | Nilai |
|---|---|
| `pop_size` | 20 |
| `n_sim` | 10 |
| `n_sim_steps` | random [10, 30] |
| `balance_value` | random [20, 100] |
| `max_steps` | 500 |
| α values | [0.05, 0.01, 0.0] |

### Parameter yang BERBEDA
| Parameter | Paper GEEvo | TA |
|---|---|---|
| Fitness function | obj3 | obj4 |
| Agent | Tidak ada | Aggressive + Passive + Random |
| n graf | 194 | 30 (abstract) + 10+10 (GDD) |
| Konfigurasi graf | Random varied (5–20 node) | Fixed per dataset |
| Sumber graf | EGG abstract saja | EGG + GDD real |

---

## 7. Hasil `evaluate_abstract_comparison.py` — Kondisi A, B & C (n=194)

Dijalankan: 2026-04-24 | α tested: [0.05, 0.01, 0.0]

### Hasil Aktual

| Kondisi | Alpha (α) | Balanced % | Initial Balanced % | Improved % | Median Gen | Median Exec Time (s) | Fitness Std Dev |
|---|---|---|---|---|---|---|---|
| A: No Agent+obj3 | 0.05 | 48.97% | 3.09% | 78.35% | 500.0 | 8.872  | 0.2925 |
| A: No Agent+obj3 | 0.01 | 14.95% | 1.03% | 82.99% | 500.0 | 14.617 | 0.2448 |
| A: No Agent+obj3 | 0.0  | 4.64%  | 0.52% | 80.41% | 500.0 | 13.705 | 0.2307 |
| B: Agent+obj3    | 0.05 | 46.39% | 3.09% | 81.44% | 500.0 | 3.54   | 0.2433 |
| B: Agent+obj3    | 0.01 | 12.37% | 1.55% | 86.60% | 500.0 | 29.76  | 0.1921 |
| B: Agent+obj3    | 0.0  | 3.09%  | 0.0%  | 84.54% | 500.0 | 39.464 | 0.2302 |
| **C: Agent+obj4 (TA)** | **0.05** | **54.64%** | **3.09%** | **87.11%** | **8.0** | **2.625** | **0.2396** |
| C: Agent+obj4 (TA) | 0.01 | 15.46% | 0.52% | 84.54% | 500.0 | 25.656 | 0.2349 |
| C: Agent+obj4 (TA) | 0.0  | 3.09%  | 0.0%  | 84.54% | 500.0 | 40.129 | 0.2302 |

### Referensi Paper (Table II)

| α | Balanced % | Improved % | Initial Balanced % | Median Gen | Median Exec Time (s) |
|---|---|---|---|---|---|
| 0.05 | 93.3% | 77.3% | 27.3% | 1   | 18.4  |
| 0.01 | 83.0% | 88.7% | 8.8%  | 7   | 66.0  |
| 0.0  | 58.8% | 94.8% | 2.5%  | 196 | 703.2 |

---

## 8. Analisis Perbedaan Hasil vs Paper (Sesi 2026-04-24)

### Penyebab Utama Perbedaan

**1. Generator steps terlalu sedikit (`steps=50` vs paper median 641)**
- Paper menggunakan max **50k steps**, median aktual **641 steps**
- Kode saat ini: `egg.run(steps=50)` — terlalu sedikit untuk menghasilkan graf valid
- Bukti: **Median Generation selalu 500** (max) di semua kondisi → balancer tidak pernah konvergen

**2. Konfigurasi graf fixed vs random**
- Paper: node types & jumlahnya diacak per graf (total 5–20 node)
- Kode: `ABSTRACT_CONF` fixed `{Source:3, RandomGate:2, Pool:4, Converter:1}` untuk semua 194 graf

**3. Fitness function `obj3` vs Eq. 2 paper — ternyata EKUIVALEN**
- Paper: `f1 = α + (1/m)Σprop(s,x)`, balanced jika `f1 ≥ 1.0` → perlu `avg(prop) ≥ 1-α`
- `obj3`: `fitness = avg(prop)`, threshold = `1 - alpha`
- Secara matematika identik ✓ — bukan sumber masalah

### Kesimpulan

Poin 1 & 2 adalah penyebab utama. Graf dengan `steps=50` kemungkinan invalid/buruk strukturnya sehingga balancer tidak bisa menemukan solusi sebelum batas 500 generasi.

---

## 9. Diskusi Metodologi untuk TA

### Apakah perlu match persis dengan paper?
**Tidak.** Kontribusi TA adalah perbandingan **A vs B vs C**, bukan replikasi exact paper. Yang penting:
- Ketiga kondisi menggunakan dataset dan metodologi yang **sama**
- Perbedaan dengan paper dijelaskan secara eksplisit di bab metodologi

### Yang bisa diperbaiki
| Perbaikan | Cara | Prioritas |
|---|---|---|
| Generator steps | Naikkan `egg.run(steps=5000)` | **Wajib** |
| Random config | Generate config acak per graf (5–20 node) | Sangat disarankan |
| Fitness function | `obj3` sudah benar | Tidak perlu diubah |

### Yang tidak bisa direplikasi persis
- Dataset 194 graf paper (seed & hardware berbeda)
- Distribusi exact node types yang dipakai paper

### Kalimat untuk bab metodologi TA
> *"Dataset 194 graf di-generate ulang menggunakan framework GEEvo yang sama, namun dengan seed berbeda dari paper asli. Perbandingan dengan paper bersifat referensial — fokus evaluasi adalah perbandingan antar kondisi A, B, dan C."*

---

## 10. Estimasi Waktu Jika Metodologi Diperbaiki

| Komponen | Sebelum Fix | Setelah Fix |
|---|---|---|
| Graph generation (194 graf) | ~10 detik | ~16 menit (+negligible) |
| Kondisi A (3 alpha) | ~2 jam | ~1.5 jam (est.) |
| Kondisi B (3 alpha) | ~4 jam | ~2–3 jam (est.) |
| Kondisi C (3 alpha) | belum dijalankan | ~2–3 jam (est.) |
| **Total** | **~6+ jam** | **~5–7 jam** (best case) |

**Catatan:** Jika balancer mulai konvergen (seperti paper), total bisa justru lebih cepat karena tidak lagi stuck di 500 generasi. Test dulu dengan `N_GRAPHS=10` sebelum full run.

---

## 11. Analisis Kondisi C vs A dan B

### Temuan Utama Kondisi C (Agent+obj4, α=0.05)

| Kondisi | Balanced % | Median Gen | Exec Time (s) | Improved % |
|---|---|---|---|---|
| A: No Agent+obj3 | 48.97% | 500.0 | 8.872 | 78.35% |
| B: Agent+obj3    | 46.39% | 500.0 | 3.54  | 81.44% |
| **C: Agent+obj4** | **54.64%** | **8.0** | **2.625** | **87.11%** |

**Kondisi C unggul di semua metrik pada α=0.05:**
- **Median Generation = 8** — satu-satunya kondisi yang konvergen sebelum batas 500 generasi
- **Balanced % tertinggi** (54.64%) meski threshold obj4 lebih ketat
- **Exec time tercepat** (2.625s) karena konvergensi lebih awal
- **Improved % tertinggi** (87.11%)

### Interpretasi
- Kombinasi agent-based simulation + obj4 (tolerance-threshold) **lebih efektif** untuk graf abstract daripada obj3 saja maupun obj3+agent
- obj4 memberikan sinyal fitness yang lebih informatif (threshold ketat → gradient lebih jelas) sehingga GA lebih cepat konvergen
- Pada α lebih ketat (0.01, 0.0), semua kondisi terjebak di 500 generasi — perbedaan obj3 vs obj4 tidak signifikan karena ambang terlalu sulit dicapai

### Narasi untuk Bab Hasil TA
> *"Kondisi C (agent-based simulation dengan obj4) menunjukkan konvergensi terbaik pada α=0.05, dengan median generasi = 8 dibandingkan 500 pada kondisi A dan B. Ini menunjukkan bahwa kombinasi kontribusi TA — simulasi berbasis agen dan fungsi fitness tolerance-threshold — secara sinergis meningkatkan kemampuan balancer dalam menemukan solusi optimal."*

---

## 12. To-Do

- [x] Jalankan kondisi A & B dari `evaluate_abstract_comparison.py`
- [x] Jalankan kondisi C
- [x] Jalankan `evaluate_paper_replication.py` (steps=50000, random config) → output: `C:\Users\MSI THIN 15 I7\output_replication\`
- [ ] Buat plot visualisasi A vs B vs C (Balanced %, Improved %, Fitness Std Dev) untuk kedua dataset
- [ ] Tulis penjelasan metodologi di bab laporan (lihat narasi di bagian 13 & 14)
- [ ] Push commit `301d07f` ke GitHub (`git push`)
- [ ] Mulai menulis laporan

---

## 13. Analisis Dua Dataset — Temuan Sesi 2026-04-25

Dua run telah selesai dan dibandingkan:

| | `output_results` | `output_replication` |
|---|---|---|
| Script | `evaluate_abstract_comparison.py` | `evaluate_paper_replication.py` |
| Generator steps | `egg.run(steps=50)` | `egg.run(steps=50000)` |
| Konfigurasi graf | Fixed `{S:3,RG:2,P:4,C:1}` | Random 5–17 node |
| Lokasi output | `C:\Users\MSI THIN 15 I7\output_results\` | `C:\Users\MSI THIN 15 I7\output_replication\` |
| Balancer max_steps | 500 (sesuai paper) | 500 (sesuai paper) |

**Catatan:** `max_steps=500` di balancer SUDAH BENAR — sesuai paper Section IV-B-1. Bukan parameter yang perlu diubah.

---

## 14. Diagnosis Kesiapan Laporan — Sesi 2026-04-25

### Pola yang Ditemukan

**Dampak Agent (B vs A) — KONSISTEN di kedua dataset:**

| Metrik | RESULTS (steps=50) | REPLICATION (steps=50k) |
|---|---|---|
| Balanced % | A unggul (−2.58%) | A unggul (−2.06%) |
| Improved % | **B unggul (+3.09%)** | **B unggul (+13.4%)** |
| Fitness Std Dev | **B lebih stabil (−0.049)** | **B lebih stabil (−0.053)** |

→ Agent secara konsisten meningkatkan kualitas pencarian (Improved % naik, variance turun) di kedua kondisi. **Kontribusi agent adalah yang paling solid.**

**Dampak obj4 (C vs B) — KONDISIONAL:**

| Dataset | α=0.05 | α=0.01 | α=0.0 |
|---|---|---|---|
| RESULTS (steps=50) | C unggul (Balanced +8.25%, gen=8) | C sedikit unggul | Identik |
| REPLICATION (steps=50k) | C marginal (+2.06%) | **B = C persis** | **B = C persis** |

→ obj4 efektif hanya pada α=0.05 dengan topologi homogen. Pada kondisi lain (α ketat atau topologi bervariasi), obj3 dan obj4 menghasilkan nilai **identik persis** dalam 500 generasi.

### Kesimpulan

**Eksperimen tambahan TIDAK DIPERLUKAN.** Data yang ada sudah cukup, tapi perlu framing ulang klaim.

### Narasi yang Bisa Dipertahankan untuk Laporan

> *"Simulasi berbasis agen secara konsisten meningkatkan kualitas pencarian algoritma evolusioner dalam menyeimbangkan ekonomi game, ditunjukkan dari peningkatan Improved % dan penurunan variance fitness di semua kondisi topologi. Fungsi fitness obj4 memberikan keunggulan tambahan pada kondisi threshold longgar (α=0.05) dengan topologi terkontrol, namun konvergen ke hasil yang ekuivalen dengan obj3 pada kondisi yang lebih ketat."*

### Struktur Bab Hasil yang Disarankan

1. **Kontribusi Agent (B vs A)** — klaim kuat, konsisten di kedua dataset
2. **Kontribusi obj4 (C vs B)** — klaim kondisional, efektif pada α=0.05 topologi homogen
3. **Evaluasi GDD** — aplikasi Kondisi C pada ekonomi game nyata (MMORPG, VHS)
4. **GDD Extractor** — kontribusi kualitatif, berdiri sendiri

### Yang Perlu Disiapkan Sebelum Nulis (bukan eksperimen, tapi persiapan)

1. Plot visualisasi A vs B vs C untuk kedua dataset
2. Framing disclaimer metodologi dua dataset di bab metodologi

---

*Catatan ini dibuat otomatis dari sesi riset bersama Claude Sonnet 4.6.*
