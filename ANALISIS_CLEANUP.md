# Analisis Cleanup Kode — GEEvo TA Tristan
> Dibuat: 2026-04-21  
> Referensi repo asli: https://github.com/FlorianRupp/GEEvo-game-economies  
> Status: CATATAN SAJA — belum ada aksi yang dilakukan

---

## Konteks

Project ini adalah fork dari GEEvo (Florian Rupp, IEEE CEC 2024) yang dikembangkan sebagai
Tugas Akhir ITS oleh Alridho Tristan Satriawan (NRP 5002221100). Tristan menambahkan lapisan
simulasi agen (aggressive/passive/random), ekstraksi GDD otomatis, dan fungsi fitness baru (obj4).

Analisis ini membandingkan kode lokal dengan repo asli untuk mengidentifikasi:
1. Kode tidak berguna (dead code / unused)
2. File output yang tersasar ke repo
3. Kode yang perlu diperbarui (.gitignore, __pycache__)

---

## BAGIAN 1 — Asal-usul Setiap Komponen

### Berasal dari Repo Asli Florian Rupp (upstream original)

| File | Komponen | Catatan |
|---|---|---|
| `geevo/graph.py` | Class `Graph` | Core, dipakai |
| `geevo/graph.py` | Class `Graph2` (baris 188–283) | Ada di original, tidak dipakai di TA |
| `geevo/evolution/balancer.py` | Class `Balancer` (inti) | Core, dipakai |
| `geevo/evolution/balancer.py` | Class `BalancerV2` (baris 320–478) | Ada di original, Tristan modifikasi minor (tambah agent1/agent2), tapi tidak pernah dipanggil |
| `geevo/evolution/balancer.py` | `get_ind_fitness_single()` obj1 (baris 67–97) | Ada di original, Tristan modifikasi untuk support agent, tapi tidak dipanggil `get_ind_fitness()` |
| `geevo/evolution/balancer.py` | `get_ind_fitness_single2()` obj2 (baris 99–129) | Ada di original, Tristan modifikasi untuk support agent, tapi tidak dipanggil `get_ind_fitness()` |
| `geevo/evolution/balancer.py` | `get_ind_fitness_single3()` obj3 | Ada di original — INI YANG DIPAKAI sebagai default |
| `geevo/evolution/generator.py` | `crossover()` method (baris 114–151) | Ada di original, sudah di-comment out di original juga (`# self.crossover()`) |
| `geevo/nodes.py` | Semua class node | Core, dipakai |
| `geevo/simulation.py` | Class `Simulator` | Core, dipakai |
| `demo.ipynb` | Notebook demo asli | Dipakai |
| `coc_machinations.gml` | Data contoh GML | Dipakai di demo.ipynb |

### Murni Tambahan Tristan (tidak ada di original)

| File | Komponen | Status |
|---|---|---|
| `geevo/agent_simulation.py` | Seluruh file — class `Agent` + `AgentSimulator` | DIPAKAI — inti TA |
| `geevo/gdd_extractor.py` | Seluruh file — class `GDDGraphExtractor` | DIPAKAI — fitur TA |
| `geevo/evolution/balancer.py` | `get_ind_fitness_single4()` obj4 (Eq 2 Paper) | DIPAKAI — fungsi baru TA |
| `geevo/evolution/balancer.py` | Parameter `fitness_func`, `agent`, `_fit_cache` di `Balancer` | DIPAKAI |
| `geevo/graph.py` | Method `set_edge_weights()` | DIPAKAI — in-place weight update performa lebih baik |
| `geevo/graph.py` | Parameter `agent` di `Graph.simulate()` | DIPAKAI |
| `geevo/nodes.py` | `is_auto`, `force_agent` di `Converter` | DIPAKAI |
| `geevo/nodes.py` | Perubahan `RandomGate.consume()`: `np.random.choice` → `np.random.multinomial` | DIPAKAI |
| `demo_agent.ipynb` | Notebook demo dengan AgentSimulator | DIPAKAI |
| `evaluate_metrics.ipynb` | Notebook evaluasi metrik | DIPAKAI |
| `test_integration.py` | Integration test | DIPAKAI |
| `profile.txt` | Output profiling (535.8KB, 3565 baris) | TIDAK PERLU — file runtime |
| `Gdd/economy_output.txt` | Output teks dari gdd_extractor | TIDAK PERLU — file runtime |
| `Gdd/pdf_results.txt` | Output teks analisis PDF GDD | TIDAK PERLU — file runtime |
| `img/archer.png`, `img/mage.png` | Gambar untuk notebook | DIPAKAI di notebook |

---

## BAGIAN 2 — Daftar Item yang Dapat Dihapus/Dibersihkan

### A. FILE — Hapus dari repo

| File | Alasan | Ukuran |
|---|---|---|
| `profile.txt` | Output profiling dari test_integration.py, bukan kode | 535.8 KB, 3565 baris |
| `Gdd/economy_output.txt` | Output runtime dari gdd_extractor, bisa di-regenerasi | — |
| `Gdd/pdf_results.txt` | Output runtime analisis PDF, bisa di-regenerasi | — |

> `update_nb_gdd.py` sudah terhapus (tercatat sebagai `D` di git status) — tinggal di-stage.

### B. KODE — Hapus dari file .py

#### 1. `geevo/graph.py` — Hapus class `Graph2` (baris 188–283)

**Alasan:** Duplikasi dari `Graph` dengan perbedaan minor (tidak ada normalisasi Dirichlet yang
benar untuk RandomGate, range bobot `[1,10]` bukan `[1,4]`). Tidak pernah diimport atau
diinstansiasi di manapun — tidak ada `Graph2(...)` di seluruh notebook maupun file `.py`.
`demo_agent.ipynb` sendiri sudah mencatatnya sebagai unused.

**Asal:** Ada di repo asli Florian Rupp — bukan kesalahan Tristan.

#### 2. `geevo/evolution/balancer.py` — Hapus class `BalancerV2` (baris 320–478)

**Alasan:** Untuk menyeimbangkan 2 graf sekaligus — fitur yang tidak masuk scope TA.
Diimport di `demo.ipynb` dan `demo_agent.ipynb` tapi tidak pernah diinstansiasi
(tidak ada `BalancerV2(...)` di manapun). Tristan memang memodifikasinya (tambah
parameter `agent1`/`agent2`) tapi tidak pernah memakainya.

**Asal:** Ada di repo asli Florian Rupp — bukan kesalahan Tristan.

#### 3. `geevo/evolution/balancer.py` — Hapus `get_ind_fitness_single()` obj1 (baris 67–97)

**Alasan:** Fungsi fitness obj1 (minimasi rasio kesenjangan 2 Pool berbeda). Tidak pernah
dipanggil oleh `get_ind_fitness()` — yang hanya memanggil obj3 atau obj4 berdasarkan
parameter `fitness_func`. Tidak ada panggilan langsung dari notebook manapun.

**Asal:** Ada di repo asli Florian Rupp. Tristan memodifikasinya untuk support agent
(menggunakan `self.g.set_edge_weights(ind)` dan loop agen), tapi tetap tidak dipakai.

#### 4. `geevo/evolution/balancer.py` — Hapus `get_ind_fitness_single2()` obj2 (baris 99–129)

**Alasan:** Fungsi fitness obj2 (minimasi derivatif deviasi antar Pool per waktu diskrit).
Sama tidak terpanggilnya seperti obj1. `get_ind_fitness()` tidak pernah memilih obj2.
`demo_agent.ipynb` sudah mencatatnya sebagai unused.

**Asal:** Ada di repo asli Florian Rupp. Sama seperti obj1 — Tristan modifikasi untuk
agent support tapi tidak dipakai.

#### 5. `geevo/evolution/generator.py` — Pertimbangkan hapus method `crossover()` (baris 114–151)

**Alasan:** Method ini ada tapi sengaja di-comment out dalam `run()`:
`# self.crossover()  -> Dimatikan/Komentar`. Ini dead code yang sudah dinonaktifkan
sejak repo asli (Florian Rupp juga sudah comment out). Bisa dihapus jika crossover
memang tidak masuk metodologi TA. Atau biarkan kalau ingin menjaga referensi eksperimental.

**Asal:** Ada di repo asli — sudah disabled sejak original.

**Rekomendasi:** Hapus saja, karena sudah tidak aktif dan tidak ada dalam metodologi TA.

### C. `.gitignore` — Perlu Diperbarui

File `.gitignore` saat ini hanya berisi:
```
__pycache__/
*.py[cod]
*$py.class
.idea/
*.pyc
```

**Entry yang perlu ditambahkan:**
```
# Output files runtime
profile.txt
Gdd/economy_output.txt
Gdd/pdf_results.txt

# Serialized graph files
*.pkl

# OS
.DS_Store
```

### D. `__pycache__/` — Untrack dari Git

`.gitignore` sudah mendaftarkan `__pycache__/` tapi folder-folder ini sudah terlanjur
di-commit ke repo sehingga masih ditracking oleh git. Perlu dijalankan:

```bash
git rm -r --cached geevo/__pycache__
git rm -r --cached geevo/evolution/__pycache__
```

---

## BAGIAN 3 — Yang TETAP DIPERTAHANKAN

Semua ini relevan dan dipakai oleh TA:

| File/Komponen | Alasan Dipertahankan |
|---|---|
| `geevo/nodes.py` | Core — semua tipe node dipakai |
| `geevo/graph.py` → class `Graph` saja | Dipakai di balancer, simulator, test |
| `geevo/simulation.py` | Base class Simulator |
| `geevo/agent_simulation.py` | Kontribusi utama TA (Agent + AgentSimulator) |
| `geevo/gdd_extractor.py` | Kontribusi TA (ekstraksi GDD otomatis) |
| `geevo/evolution/generator.py` | EGG algorithm — dipakai |
| `geevo/evolution/balancer.py` → class `Balancer` + obj3 + obj4 saja | Inti algoritma optimasi TA |
| `test_integration.py` | Regression test — penting untuk verifikasi |
| `demo.ipynb` | Demo asli GEEvo |
| `demo_agent.ipynb` | Demo dengan AgentSimulator — kontribusi TA |
| `evaluate_metrics.ipynb` | Evaluasi metrik fitness — kontribusi TA |
| `coc_machinations.gml` | Data contoh asli GEEvo, dipakai di demo.ipynb |
| `img/archer.png`, `img/mage.png` | Dipakai di notebook |
| `Gdd/*.pdf` | 4 file GDD sebagai dataset input gdd_extractor |

---

## BAGIAN 4 — Ringkasan Aksi yang Direncanakan (Belum Dilakukan)

```
HAPUS FILE:
  - profile.txt
  - Gdd/economy_output.txt
  - Gdd/pdf_results.txt

HAPUS KODE:
  - geevo/graph.py           : class Graph2 (baris 188–283)
  - geevo/evolution/balancer.py : class BalancerV2 (baris 320–478)
  - geevo/evolution/balancer.py : method get_ind_fitness_single() obj1 (baris 67–97)
  - geevo/evolution/balancer.py : method get_ind_fitness_single2() obj2 (baris 99–129)
  - geevo/evolution/generator.py: method crossover() (baris 114–151) [opsional]

PERBARUI .gitignore:
  - Tambah profile.txt, Gdd/economy_output.txt, Gdd/pdf_results.txt, *.pkl

UNTRACK __pycache__:
  - git rm -r --cached geevo/__pycache__
  - git rm -r --cached geevo/evolution/__pycache__

STAGE DELETION:
  - git add -u  (untuk menangkap update_nb_gdd.py yang sudah dihapus)
```

---

*Catatan: Analisis ini tidak mengubah apapun. Semua aksi di atas menunggu persetujuan.*
