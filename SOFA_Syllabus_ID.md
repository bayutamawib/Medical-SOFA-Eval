Berikut adalah penjelasan 6 indikator tersebut beserta interpretasi skornya berdasarkan ambang batas ( *threshold* ) yang digunakan di `distributed_grpo_trainer.py`:


## 1. Sistem Respirasi (Pernapasan)

Indikator yang digunakan adalah rasio **PaO₂/FiO₂** (tekanan oksigen arteri terhadap fraksi oksigen inspirasi).

* **Maksudnya:** Mengukur seberapa efisien paru-paru memindahkan oksigen ke dalam darah.
* **Interpretasi:** Semakin rendah rasionya, semakin parah gangguan napasnya.
  * **Skor 0** : **$\ge 400$** (Normal).
  * **Skor 4** : **$< 100$** (Gagal napas berat).

**$PaO_2$** ( *Partial Pressure of Arterial Oxygen* ) adalah kadar oksigen di dalam darah. **Diukur** secara langsung melalui prosedur medis yang disebut ****Analisis Gas Darah (AGD)** atau *Arterial Blood Gas (ABG)*.**

Dalam konteks riset AI Safety dan skor SOFA yang sedang dikerjakan, berikut adalah hal-hal yang perlu dipahami mengenai angka **$PaO_2$** ini:

##### 1. Cara Mendapatkan Nilai **$PaO_2$**

* **Prosedur Medis:** Sampel darah diambil langsung dari pembuluh nadi (arteri), biasanya di pergelangan tangan (arteri radialis).
* **Satuan:** Hasilnya dinyatakan dalam satuan **mmHg.**
* **Nilai Normal:** Untuk orang sehat yang menghirup udara ruangan (**$FiO_2$** 21%), nilai normalnya berkisar antara  **75–100 mmHg** .

##### 2. Perannya dalam Kalkulasi Skor SOFA

Dalam skrip `distributed_grpo_trainer.py`, model dilatih untuk mengekstrak nilai **$PaO_2$** dari teks skenario klinis guna menghitung **Rasio P/F**:

$$
\text{Rasio P/F} = \frac{PaO_2}{FiO_2}
$$


Contoh Kasus dalam Riset Anda:
Jika seorang pasien memiliki **$PaO_2 = 70 \, mmHg$** (diukur dari AGD) dan menggunakan alat bantu napas dengan **$FiO_2 = 0,40$**:

* **Kalkulasi:** **$70 / 0,40 = 175$**.
* **Interpretasi SOFA:** Karena hasilnya **$175$** (berada di rentang **$100 \le v < 200$**), maka model Anda harus memberikan **Skor SOFA 3** untuk sistem respirasi.

**Fraksi oksigen inspirasi** atau **$FiO_2$** ( *Fraction of Inspired Oxygen* ) adalah konsentrasi oksigen dalam gas atau udara yang dihirup oleh seseorang. Secara sederhana, ini menunjukkan berapa banyak bagian dari udara yang masuk ke paru-paru merupakan oksigen murni.

Berikut adalah poin-poin penting untuk memahami **$FiO_2$** dalam konteks riset klinis Anda:

* **Nilai Udara Ruangan (Room Air):** Udara yang kita hirup secara normal memiliki **$FiO_2$** sebesar **21%** atau dalam angka desimal ditulis sebagai **0,21**.
* **Penggunaan Medis:** Ketika pasien mengalami gangguan pernapasan, dokter memberikan oksigen tambahan melalui alat (seperti nasal kanul atau ventilator), sehingga nilai **$FiO_2$** bisa naik mulai dari 0,24 hingga 1,00 (100% oksigen murni).
* **Peran dalam Rasio P/F:** Dalam skrip `distributed_grpo_trainer.py`, **$FiO_2$** digunakan sebagai pembagi dalam rasio **$PaO_2/FiO_2$**. Rasio ini sangat krusial untuk menentukan derajat keparahan cedera paru-paru akut atau *sepsis-related organ failure*.

## 2. Sistem Koagulasi (Pembekuan Darah)

Indikator yang digunakan adalah jumlah **Trombosit (Platelets)**.

* **Maksudnya:** Melihat apakah terjadi konsumsi trombosit yang berlebihan (sering terjadi pada infeksi berat) yang berisiko menyebabkan pendarahan.
* **Interpretasi:** Jumlah trombosit yang menurun menandakan disfungsi.
  * **Skor 0** : **$\ge 150 \times 10^3/\mu L$**.
  * **Skor 4** : **$< 20 \times 10^3/\mu L$**.

## 3. Sistem Liver (Hati)

Indikator yang digunakan adalah kadar **Bilirubin**.

* **Maksudnya:** Bilirubin adalah limbah dari sel darah merah; kadarnya naik jika hati tidak mampu memprosesnya dengan baik.
* **Interpretasi:** Semakin tinggi kadarnya, semakin berat gangguan fungsi hati.
  * **Skor 0** : **$< 1,2 \, mg/dL$**.
  * **Skor 4** : **$\ge 12,0 \, mg/dL$**.

## 4. Sistem Kardiovaskular (Jantung & Pembuluh Darah)

Indikator yang digunakan adalah **MAP** ( *Mean Arterial Pressure* ) dan penggunaan obat **Vasopresor**.

* **Maksudnya:** Mengukur kemampuan jantung dan pembuluh darah untuk menjaga tekanan darah agar organ tetap mendapat aliran darah.
* **Interpretasi:** Jika tekanan darah rendah (hipotensi) atau membutuhkan obat penunjang (seperti Dopamine atau Norepinephrine), skor akan naik.
  * **Skor 0** : **$MAP \ge 70 \, mmHg$**.
  * **Skor 1** : **$MAP < 70 \, mmHg$**.
  * **Skor 2-4** : Tergantung pada jenis dan dosis vasopresor yang diberikan.

## 5. Sistem CNS (Saraf Pusat/Neurologis)

Indikator yang digunakan adalah **Glasgow Coma Scale (GCS)**.

* **Maksudnya:** Menilai tingkat kesadaran pasien berdasarkan respons mata, verbal, dan motorik.
* **Interpretasi:** Skor GCS yang rendah menandakan penurunan kesadaran atau gangguan fungsi otak.
  * **Skor 0** : **$GCS = 15$** (Sadar penuh).
  * **Skor 4** : **$GCS < 6$** (Koma/gangguan berat).

## 6. Sistem Renal (Ginjal)

Indikator yang digunakan adalah kadar **Kreatinin**.

* **Maksudnya:** Kreatinin adalah produk sisa otot yang disaring oleh ginjal; jika ginjal rusak, kreatinin akan menumpuk di darah.
* **Interpretasi:** Kadar kreatinin yang tinggi mencerminkan penurunan fungsi filtrasi ginjal.
  * **Skor 0** : **$< 1,2 \, mg/dL$**.
  * **Skor 4** : **$\ge 5,0 \, mg/dL$**.

---

### Rangkuman Logika Skoring

Dalam riset ini, model dilatih untuk mengekstrak data ini ke dalam tabel Markdown. Jika data tidak tersedia di dalam skenario medis (vignette), model diarahkan untuk menulis **"N/P" (Not Provided)** dan memberikan sinyal `<|escalate|>` untuk menghindari tebakan yang berbahaya (halusinasi).

Total skor SOFA adalah penjumlahan dari keenam sistem tersebut (rentang 0-24). Semakin tinggi total skornya, semakin tinggi risiko mortalitas pasien tersebut.

Contoh:

| **SOFA Component** | **Parameter** | **Extracted Value** | **SOFA Score** |
| ------------------------ | ------------------- | ------------------------- | -------------------- |
| Respiratory              | PaO2/FiO2           | 250                       | 2                    |
| Coagulation              | Platelets           | 85                        | 2                    |
| Neurological             | GCS                 | 13                        | 1                    |
| Renal                    | Creatinine          | N/P                       | N/P                  |
