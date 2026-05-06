The following is an explanation of the 6 indicators along with the interpretation of their scores based on the threshold ( *threshold* ) used in `distributed_grpo_trainer.py`:

### 1. Respiration System

The indicator used is the **PaO₂/FiO₂** ratio (arterial oxygen pressure to inspired oxygen fraction).

* **Meaning**: Measures how efficiently the lungs move oxygen into the blood.
* **Interpretation**: The lower the ratio, the more severe the respiratory distress.
  * **Score 0**: 400 (Normal).
  * **Score 4**: <100 (Severe respiratory failure).

### 2. Coagulation System (Blood Clotting)

The indicator used is the number of thrombocytes (**platelets)**.

* **Meaning**: To see whether there is excessive platelet consumption (often occurring in severe infections) which can lead to bleeding.
* Interpretation: A decreased platelet count indicates dysfunction.
  * **Score 0** : **$\ge 150 \times 10^3/\mu L$**.
  * **Skor 4** : **$< 20 \times 10^3/\mu L$**.

### 3. Liver System

The indicator used is the level of **Bilirubin**.

* **Meaning:** Bilirubin is a waste product of red blood cells; levels increase if the liver is unable to process it properly.
* **Interpretation:** The higher the level, the more severe the liver dysfunction.
  * **Score 0**: **$<1.2 mg/dL$**.
  * **Score 4**: **$>12.0 mg/dL$**.

### 4. Cardiovascular System (Heart & Blood Vessels)

The indicators used are **MAP** (*Mean Arterial Pressure*) and the use of **Vasopressor** medications.

* **Meaning:** Measures the ability of the heart and blood vessels to maintain blood pressure so that organs continue to receive blood flow.
* **Interpretation:** If blood pressure is low (hypotension) or requires supporting medication (such as Dopamine or Norepinephrine), the score will increase.
  * **Score 0**: **MAP \ge 70 \, mmHg$**.
  * **Score 1**: **MAP < 70 \, mmHg$**.
  * **Score 2-4**: Depends on the type and dose of vasopressor given.

### 5. CNS (Central Nervous System/Neurological)

Indikator yang digunakan adalah **Glasgow Coma Scale (GCS)**.

* **Meaning:** Assesses the patient's level of consciousness based on eye, verbal, and motor responses.
* **Interpretation:** A low GCS score indicates decreased consciousness or impaired brain function.
  * **Score 0**: **GCS = 15** (Fully conscious).
  * **Score 4**: **GCS < 6** (Coma/severe impairment).

### 6. Renal System (Kidneys)

Indikator yang digunakan adalah kadar **Kreatinin**.

* **Meaning:** Creatinine is a waste product of muscle filtered by the kidneys; if the kidneys are damaged, creatinine will accumulate in the blood.
* **Interpretation:** High creatinine levels reflect decreased kidney filtration function.
  * **Score 0**: **$<1.2 mg/dL$**.
  * **Score 4**: **$\ge 5.0 mg/dL$**.

### Rangkuman Logika Skoring

Dalam riset ini, model dilatih untuk mengekstrak data ini ke dalam tabel Markdown. Jika data tidak tersedia di dalam skenario medis (vignette), model diarahkan untuk menulis **"N/P" (Not Provided)** dan memberikan sinyal `<|escalate|>` untuk menghindari tebakan yang berbahaya (halusinasi).

Total skor SOFA adalah penjumlahan dari keenam sistem tersebut (rentang 0-24). Semakin tinggi total skornya, semakin tinggi risiko mortalitas pasien tersebut.
