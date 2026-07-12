# Healthcare Data Mining Dashboard

Healthcare Data Mining Dashboard adalah aplikasi berbasis **Streamlit** untuk menampilkan ringkasan data pasien serta menerapkan tiga analisis data mining pada dataset healthcare. Dashboard menggunakan tampilan interaktif dengan visualisasi Plotly dan nilai biaya ditampilkan dalam **dolar ($)**.

## Fitur Dashboard

### 1. Dashboard Utama
Menampilkan gambaran umum data healthcare, meliputi:

- total pasien;
- rata-rata biaya perawatan;
- rata-rata lama rawat;
- persentase hasil tes abnormal;
- tren jumlah pasien berdasarkan tahun admisi;
- jumlah pasien berdasarkan kondisi medis;
- proporsi pasien berdasarkan provider asuransi;
- rata-rata biaya berdasarkan kondisi medis;
- filter gender, kondisi medis, dan tahun admisi;
- insight otomatis dan rekomendasi keputusan;
- tabel contoh data pasien.

### 2. Regresi Biaya
Digunakan untuk memperkirakan biaya perawatan pasien.

Fitur yang tersedia:

- formulir input karakteristik pasien;
- prediksi biaya perawatan;
- kategori biaya rendah, sedang, atau tinggi;
- perbandingan hasil prediksi dengan rata-rata dataset;
- evaluasi model menggunakan MAE, RMSE, dan R²;
- grafik nilai aktual dibandingkan nilai prediksi;
- visualisasi fitur yang paling berpengaruh;
- insight dan rekomendasi berdasarkan hasil prediksi.

### 3. Klasifikasi Risiko
Digunakan untuk mengelompokkan pasien ke dalam risiko sumber daya normal atau tinggi.

Fitur yang tersedia:

- formulir input karakteristik pasien;
- hasil klasifikasi risiko;
- probabilitas risiko sumber daya tinggi;
- gauge probabilitas;
- evaluasi menggunakan Accuracy, Precision, Recall, F1-score, dan ROC-AUC;
- confusion matrix;
- kurva ROC;
- visualisasi fitur yang paling berpengaruh;
- insight dan rekomendasi berdasarkan tingkat risiko.

### 4. Clustering Pasien
Digunakan untuk membentuk kelompok pasien berdasarkan kemiripan usia, lama rawat, dan biaya perawatan.

Fitur yang tersedia:

- pilihan jumlah cluster;
- Silhouette Score;
- visualisasi cluster dua dimensi menggunakan PCA;
- visualisasi cluster tiga dimensi;
- tabel profil setiap cluster;
- jumlah dan persentase pasien pada setiap cluster;
- rata-rata usia, lama rawat, dan biaya;
- kondisi medis dan jenis admisi dominan;
- insight dan rekomendasi untuk setiap cluster.

## Struktur File

```text
project/
├── app.py
├── healthcare_dataset.csv
├── README.md
└── requirements.txt
```

