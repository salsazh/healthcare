# Healthcare Data Mining Dashboard

Dashboard interaktif berbasis **Streamlit** untuk menganalisis data layanan kesehatan dan menerapkan tiga metode data mining, yaitu **regresi, klasifikasi, dan clustering**. Dashboard menampilkan ringkasan data, visualisasi, prediksi, evaluasi model, insight, serta rekomendasi untuk mendukung pengambilan keputusan organisasi pelayanan kesehatan.


## Tujuan Proyek

Proyek ini bertujuan untuk:

- menyajikan informasi utama mengenai data pasien;
- memperkirakan biaya perawatan pasien;
- mengklasifikasikan risiko biaya perawatan tinggi;
- mengelompokkan pasien berdasarkan karakteristik yang serupa;
- menghasilkan insight dan rekomendasi untuk mendukung keputusan operasional.

## Fitur Dashboard

Dashboard memiliki empat menu utama:

### 1. Dashboard Utama

Menampilkan:

- total pasien;
- rata-rata biaya perawatan;
- rata-rata lama rawat;
- persentase hasil tes abnormal;
- tren jumlah pasien berdasarkan tahun;
- distribusi kondisi medis;
- distribusi provider asuransi;
- insight otomatis dan rekomendasi keputusan.

### 2. Regresi Biaya

Menggunakan **Random Forest Regressor** untuk memprediksi nilai `Billing Amount`.

Fitur yang digunakan:

- `Age`
- `Gender`
- `Blood Type`
- `Medical Condition`
- `Insurance Provider`
- `Admission Type`
- `Medication`
- `Test Results`
- `Length of Stay`

Target:

- `Billing Amount`

Evaluasi model:

- Mean Absolute Error atau MAE;
- Root Mean Squared Error atau RMSE;
- R-squared atau R²;
- grafik aktual dibandingkan prediksi;
- feature importance.

Fungsi untuk pengambilan keputusan:

- memberikan estimasi awal biaya pasien;
- membantu perencanaan anggaran;
- membantu pemantauan pasien dengan estimasi biaya relatif tinggi;
- menunjukkan faktor yang paling berpengaruh terhadap hasil model.

### 3. Klasifikasi Risiko Biaya

Menggunakan **Random Forest Classifier** untuk mengklasifikasikan pasien ke dalam:

- Risiko Biaya Normal;
- Risiko Biaya Tinggi.

Target klasifikasi dibuat dari nilai `Billing Amount` asli. Pasien dikategorikan berisiko biaya tinggi apabila nilai biayanya berada pada atau di atas kuartil ke-75 dataset.

Fitur yang digunakan sama dengan fitur regresi. Kolom `Billing Amount` tidak dimasukkan sebagai fitur sehingga tidak terjadi target leakage.

Evaluasi model:

- Accuracy;
- Precision;
- Recall;
- F1-score;
- ROC-AUC;
- confusion matrix;
- kurva ROC;
- feature importance.

Fungsi untuk pengambilan keputusan:

- membantu mengidentifikasi pasien yang berpotensi memiliki biaya tinggi;
- mendukung verifikasi pembiayaan dan asuransi lebih awal;
- membantu pemantauan biaya selama perawatan;
- mendukung prioritas pengawasan administrasi dan keuangan.

### 4. Clustering Pasien

Menggunakan algoritma **K-Means Clustering** untuk mengelompokkan pasien berdasarkan:

- `Age`;
- `Length of Stay`;
- `Billing Amount`.

Sebelum clustering, data numerik distandardisasi menggunakan `StandardScaler`. Hasil clustering divisualisasikan menggunakan PCA dua dimensi dan grafik tiga dimensi.

Evaluasi dan hasil:

- jumlah cluster dapat dipilih antara 2 sampai 6;
- Silhouette Score;
- persentase variasi PCA;
- jumlah pasien pada setiap cluster;
- rata-rata usia;
- rata-rata lama rawat;
- rata-rata biaya;
- kondisi medis dominan;
- jenis admisi dominan;
- profil dan rekomendasi setiap cluster.

Fungsi untuk pengambilan keputusan:

- membantu segmentasi kelompok pasien;
- mendukung perencanaan kapasitas tempat tidur;
- membantu identifikasi kelompok dengan biaya tinggi;
- membantu evaluasi kelompok dengan lama rawat tinggi;
- mendukung penyusunan strategi pelayanan yang berbeda untuk setiap cluster.

## Cara Kerja Data

Tahapan pengolahan data:

1. Membaca file `healthcare_dataset.csv`.
2. Memeriksa keberadaan kolom wajib.
3. Mengubah kolom usia dan biaya menjadi data numerik.
4. Mengubah tanggal masuk dan tanggal keluar menjadi format tanggal.
5. Menghitung `Length of Stay` dari selisih tanggal keluar dan tanggal masuk.
6. Membuat `Admission Year` dari tahun tanggal masuk.
7. Menghapus data kosong, biaya tidak valid, lama rawat tidak valid, dan data duplikat.
8. Mengubah fitur kategorikal menggunakan `OneHotEncoder`.
9. Melakukan standardisasi fitur numerik menggunakan `StandardScaler`.
10. Membagi data latih dan data uji untuk model regresi dan klasifikasi.
11. Melatih model, menghitung evaluasi, dan menampilkan hasil dalam dashboard.

## Struktur File

```text
project/
├── app.py
├── healthcare_dataset.csv
├── requirements.txt
└── README.md
```

Nama dataset wajib:

```text
healthcare_dataset.csv
```

## Kesimpulan

Dashboard ini telah menerapkan tiga metode data mining:

1. **Random Forest Regressor** untuk prediksi biaya perawatan.
2. **Random Forest Classifier** untuk klasifikasi risiko biaya tinggi.
3. **K-Means Clustering** untuk segmentasi pasien.

Ketiga metode tersebut dilengkapi dengan visualisasi, evaluasi, insight, dan rekomendasi sehingga dapat digunakan untuk mendukung perencanaan anggaran, pengawasan biaya, pengelolaan kapasitas, dan segmentasi pelayanan pasien.