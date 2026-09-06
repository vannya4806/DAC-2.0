# DAC 2026 — NHPA Claim Fraud Detection

## Ringkasan Proyek
Model prediksi fraud probability untuk klaim asuransi kesehatan, dioptimalkan
untuk membantu alokasi kapasitas audit terbatas (3%, 5%, 7%) milik NHPA.

## Pipeline
1. **EDA** (`01_EDA.ipynb`) — eksplorasi data, deteksi duplikat, leakage, sparsity
2. **Preprocessing** (`02_preprocessing.ipynb`) — encoding, transform, split
3. **Modeling** (`03_modeling_randomforest.ipynb`) — 33 konfigurasi Random Forest,
   5-fold Stratified CV (OOF), 3 metode ensemble (simple avg, rank, weighted rank)
4. **Audit Allocation** (`04_audit_allocation.ipynb`) — Task C
5. **Fairness Evaluation** (`05_fairness_evaluation.ipynb`) — Task D

## Temuan Kunci dari EDA
- Label biner ~50/50 (bukan probabilitas kontinu seperti disebut brief) — nyaris tidak ada class imbalance.
- Ditemukan 1.737 grup kombinasi fitur identik dengan label fraud berbeda-beda (5.361 baris) — indikasi **irreducible error**: ada batas atas performa model yang tidak bisa dilewati dengan fitur yang tersedia saat ini.
- Kolom high-cardinality: `kdkc` (126 kategori), `dati2` (486 kategori), `typeppk`, `cmg`, `diagprimer` → ditangani via target encoding dengan smoothing.
- Kolom count (`dx2_*`, `proc*`) sangat sparse (>99% nol pada banyak kolom) → ditransformasi jadi indikator biner.

## Model Final
**Weighted Rank Ensemble** dari top-10 (dari 33 kandidat) konfigurasi Random Forest, dipilih berdasarkan performa OOF dan hasil leaderboard.

- **NormalizedRecall@5% (OOF): 0.9683**

## Hasil Task C — Audit Allocation

| Kapasitas | Klaim Diaudit | Fraud Tertangkap | Fraud Terlewat | Klaim Jujur Teraudit | Recall | Precision | NormalizedRecall@k | Lift vs Random |
|---|---|---|---|---|---|---|---|---|
| 3% | 4.805 | 4.706 | 75.497 | 99 | 5.87% | 97.94% | 0.9794 | 1.96x |
| 5% | 8.008 | 7.754 | 72.449 | 254 | 9.67% | 96.83% | 0.9683 | 1.93x |
| 7% | 11.212 | 10.748 | 69.455 | 464 | 13.40% | 95.86% | 0.9586 | 1.91x |

**Interpretasi:**
- Model secara konsisten menangkap fraud hampir **2x lebih banyak** dibanding audit acak di semua tingkat kapasitas.
- Presisi audit sangat tinggi (95.9%–97.9%) — dari klaim yang direkomendasikan untuk diaudit, sebagian besar benar-benar fraud, sehingga sumber daya audit NHPA tidak banyak terbuang ke klaim yang ternyata jujur.
- NormalizedRecall menurun sedikit seiring kapasitas audit membesar (0.979 → 0.968 → 0.959) — wajar karena semakin banyak klaim diaudit, semakin sulit menjaga presisi ranking di posisi-posisi selanjutnya.
- **Sensitivitas cut-off:** gap skor tepat di batas top-5% (0.000004) lebih kecil dari rata-rata gap antar skor keseluruhan (0.000006) — artinya keputusan audit di batas 5% ini **cukup sensitif**: sedikit perubahan pada model atau data bisa menggeser klaim mana yang tepat masuk/keluar dari cutoff. Ini perlu dicatat sebagai keterbatasan, terutama untuk klaim yang skornya berada persis di sekitar garis batas.
- Total 2.002 klaim direkomendasikan untuk diaudit dari test set pada kapasitas 5%.

## Hasil Task D — Fairness

### Gender (jkpst)
| | Proporsi Populasi | Proporsi Diaudit | Selisih |
|---|---|---|---|
| P (Perempuan) | 53.62% | 54.37% | +0.75% |
| L (Laki-laki) | 46.38% | 45.63% | -0.75% |

- Base rate fraud (dari label asli) **identik** antara L dan P (0.5007 keduanya) — tidak ada bias pada label sumber terhadap gender.
- Precision audit sedikit berbeda: L (97.04%) vs P (96.65%) — selisih kecil (~0.4%), tidak signifikan secara praktis.
- **Kesimpulan:** disparitas audit berdasarkan gender sangat kecil (<1%) dan tidak menunjukkan pola bias sistematis.

### Age Group
| | Proporsi Populasi | Proporsi Diaudit | Selisih |
|---|---|---|---|
| 40-59 | 29.87% | 30.33% | +0.46% |
| 18-39 | 25.84% | 26.21% | +0.37% |
| 0-17 | 25.05% | 25.62% | +0.57% |
| 60+ | 19.24% | 17.83% | **-1.41%** |

- Precision audit relatif merata antar kelompok umur (18-39: 96.67%, 40-59: 96.95%, 60+: 96.92%).
- **Kelompok 60+ diaudit sedikit LEBIH JARANG** dibanding proporsi populasinya (selisih -1.41%, yang terbesar di antara semua kelompok umur) — ini pola yang perlu didiskusikan lebih lanjut. Ini **bukan berarti diskriminasi terhadap lansia**; kemungkinan penyebabnya termasuk pola klinis yang berbeda pada kelompok usia lanjut (misal jenis diagnosis/prosedur yang secara natural kurang mencolok dalam fitur yang tersedia), bukan bias langsung terhadap usia itu sendiri.

### Interaksi Gender × Age Group
Audit rate berkisar 4.56%–5.30% di seluruh 8 kombinasi gender×age group — variasinya kecil dan tidak menunjukkan subgroup tertentu yang secara mencolok kurang/lebih terwakili, kecuali kombinasi (L, 60+) yang audit rate-nya sedikit lebih rendah (4.56%) dibanding kombinasi lain.

## Keterbatasan
- Ada **batas atas performa (irreducible error)** akibat 1.737 kombinasi fitur identik dengan label berbeda-beda, ditemukan di EDA — NormalizedRecall@5% tidak mungkin mencapai 1.0 sempurna dengan fitur yang tersedia.
- **Sensitivitas cut-off tinggi** di batas 5% — keputusan audit untuk klaim yang skornya persis di sekitar cutoff cukup rentan berubah dengan variasi model/data kecil.
- Disparitas audit pada kelompok usia 60+ (-1.41%) perlu investigasi lebih lanjut untuk memastikan ini mencerminkan pola klinis yang wajar, bukan keterbatasan fitur yang secara tidak sengaja merugikan kelompok ini.
- Model tidak menggunakan data eksternal, sesuai aturan kompetisi yang diperbarui.

## Tools & Library
Python 3.13, pandas, numpy, scikit-learn (RandomForestClassifier, StratifiedKFold), scipy (rankdata), matplotlib, seaborn.

## Struktur Proyek
```
DAC/
├── Dataset/{RawDataset, Processed}/
├── Notebooks/{01_EDA, 02_preprocessing, 03_modeling_randomforest,
│              04_audit_allocation, 05_fairness_evaluation}.ipynb
├── outputs/{figures,submissions}/
├── results/{audit_allocation, fairness,predictions}/
└── README.md
└── requirements.txt