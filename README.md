# UTS Machine Learning — Linear Regression & Logistic Regression

## Identitas Mahasiswa

- **Nama:** Fachri
- **NIM:** 241011400621
- **Kelas:** Reg B
- **Random state:** `21`

## Tujuan

Tugas ini menerapkan Linear Regression untuk memperkirakan penjualan berdasarkan biaya iklan dan Logistic Regression untuk mengklasifikasikan email sebagai spam atau bukan spam. Implementasi utama pembagian data, model, dan metrik dibuat menggunakan NumPy sesuai instruksi tugas. Scikit-learn hanya digunakan pada bagian bonus sebagai pembanding.

## Dataset

- `Ads_Data.csv`: 40 baris dengan kolom `BiayaIklan` dan `Penjualan`. Nilai biaya iklan dan penjualan dinyatakan dalam juta rupiah.
- `Email_Data.csv`: 60 baris dengan kolom `JumlahLink`, `KataPromo`, dan `Spam`. Label `Spam=1` berarti spam dan `Spam=0` berarti bukan spam. Dataset berisi 25 email spam dan 35 email bukan spam.

## Bagian A — Linear Regression

Data dibagi menjadi 32 baris training dan 8 baris testing. Model Normal Equation menghasilkan persamaan:

`ŷ = 27.8884 + 9.3474 × BiayaIklan`

Koefisien positif menunjukkan bahwa biaya iklan yang lebih besar berkaitan dengan prediksi penjualan yang lebih tinggi pada dataset ini.

| Data | MAE | MSE | RMSE | R² |
|---|---:|---:|---:|---:|
| Training | 4.6088 | 39.2516 | 6.2651 | 0.9433 |
| Testing | 3.9922 | 23.1709 | 4.8136 | 0.9406 |

RMSE testing sekitar Rp4,81 juta dan R² testing 0,9406. Ini menunjukkan hubungan linear cukup kuat pada data testing, tetapi kesalahan prediksi tetap ada dan performa testing lebih rendah daripada training.

### Eksperimen Gradient Descent

| Learning rate | MSE akhir | Hasil |
|---:|---:|---|
| 0.001 | 41.2635 | Turun, tetapi belum konvergen setelah 5.000 iterasi |
| 0.02 | 39.2516 | Konvergen mendekati hasil Normal Equation |
| 0.1 | Tidak hingga (NaN) | Divergen/tidak stabil |

Learning rate yang terlalu kecil membutuhkan lebih banyak iterasi, sedangkan learning rate 0,1 terlalu besar untuk skala fitur dan target pada eksperimen ini. Prediksi pada biaya iklan Rp12 juta adalah ekstrapolasi karena melampaui rentang dataset (Rp0,8–Rp9,9 juta), sehingga lebih tidak pasti daripada prediksi di dalam rentang data.

## Bagian B — Logistic Regression

Data dibagi menjadi 45 baris training dan 15 baris testing. Model menggunakan Gradient Descent dengan learning rate 0,1 dan 10.000 iterasi, menghasilkan persamaan:

`P(Spam) = sigmoid(-12.491747 + 1.098748 × JumlahLink + 1.032337 × KataPromo)`

Log loss turun dari 0,6840 menjadi 0,1766 selama pelatihan.

### Evaluasi testing pada threshold 0,5

| Metrik | Hasil |
|---|---:|
| Accuracy | 0.8000 (80%) |
| Precision | 0.8000 |
| Recall | 0.6667 |
| F1-score | 0.7273 |

Confusion matrix dengan urutan `[[TN, FP], [FN, TP]]`:

```text
[[10, 0],
 [1, 4]]
```

Artinya, 8 email bukan spam dan 4 email spam diklasifikasikan dengan benar; terdapat 1 false positive dan 2 false negative.

### Eksperimen threshold

| Threshold | TN | FP | FN | TP | Precision | Recall | Accuracy |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.3 | 7 | 2 | 2 | 4 | 0.6667 | 0.6667 | 0.7333 |
| 0.5 | 8 | 1 | 2 | 4 | 0.8000 | 0.6667 | 0.8000 |
| 0.7 | 9 | 0 | 3 | 3 | 1.0000 | 0.5000 | 0.8000 |

Threshold 0,5 dipilih untuk prediksi email baru karena memberikan keseimbangan antara precision dan recall pada eksperimen ini. Threshold 0,7 menghilangkan false positive pada data testing ini, tetapi meningkatkan false negative dan menurunkan recall.

### Prediksi email baru

Dengan threshold 0,5, model mengklasifikasikan E1 dan E2 sebagai bukan spam, serta E3 dan E4 sebagai spam.

| Email | Jumlah link | Kata promo | Probabilitas Spam (±) | Prediksi |
|---|---:|---:|---:|---|
| E1 | 1 | 2 | 0.0001 | Bukan Spam |
| E2 | 5 | 6 | 0.3092 | Bukan Spam |
| E3 | 9 | 10 | 0.9996 | Spam |
| E4 | 2 | 11 | 0.7430 | Spam |

## Bonus — Perbandingan dengan scikit-learn

Pada Linear Regression, implementasi NumPy dan scikit-learn memberikan hasil yang sama pada presisi yang ditampilkan: intercept 27,8884, koefisien 9,3474, MAE testing 6,4464, MSE testing 80,7492, dan R² testing 0,8241.

Pada Logistic Regression, metrik klasifikasi testing di threshold 0,5 sama pada eksperimen ini: accuracy 0,80, precision 0,80, recall 0,6667, dan F1-score 0,7273. Koefisiennya berbeda; implementasi NumPy menghasilkan intercept sekitar -12,4917, sedangkan scikit-learn sekitar -15,1846. Perbedaan dapat dipengaruhi regularisasi, solver, serta kriteria konvergensi scikit-learn. Bagian bonus tidak menggantikan implementasi utama NumPy.

## Kesimpulan

Linear Regression menunjukkan hubungan positif antara biaya iklan dan penjualan, dengan R² testing 0,8241. Gradient Descent memperlihatkan bahwa pemilihan learning rate berpengaruh besar terhadap konvergensi. Logistic Regression memperoleh akurasi testing 80% pada threshold 0,5, tetapi masih melakukan kesalahan klasifikasi sehingga belum dapat dianggap sempurna. Hasil ini berlaku untuk dataset yang digunakan dan tidak menjamin performa yang sama pada data baru.

## Struktur Repository

```text
uts-machine-learning-241011400621/
├── README.md
├── tugas_UTS.ipynb
├── Ads_Data.csv
├── Email_Data.csv
└── requirements.txt
```

## Cara Menjalankan

1. Buka folder repository di VS Code atau Jupyter Notebook.
2. Pasang dependensi menggunakan `python -m pip install -r requirements.txt`.
3. Buka `tugas_UTS.ipynb`, pilih kernel Python yang sesuai, lalu jalankan **Run All**.

Notebook yang disertakan sudah dieksekusi dari awal hingga akhir dan output tabel, metrik, serta grafik disimpan di dalam notebook.
