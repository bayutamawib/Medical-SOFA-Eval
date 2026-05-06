The following is an explanation of the 6 indicators along with the interpretation of their scores based on the threshold ( *threshold* ) used in this module:

### 1. Respiration System

The indicator used is the **PaO₂/FiO₂** ratio (arterial oxygen pressure to inspired oxygen fraction).

* **Meaning**: Measures how efficiently the lungs move oxygen into the blood.
* **Interpretation**: The lower the ratio, the more severe the respiratory distress.
  * **Score 0**: 400 (Normal).
  * **Score 4**: <100 (Severe respiratory failure).

**$PaO_2$** (*Partial Pressure of Arterial Oxygen*) is the level of oxygen in the blood. **Measured** directly through a medical procedure called **Arterial Blood Gas (ABG)**.

In the context of AI Safety research and the SOFA score currently being developed, here are the things to understand regarding this **$PaO_2$** value:

##### 1. How to Obtain the **$PaO_2$** Value

* **Medical Procedure:** A blood sample is taken directly from an artery, usually at the wrist (radial artery).
* **Unit:** The result is expressed in **mmHg**.
* **Normal Value:** For a healthy person breathing room air (**$FiO_2$** 21%), the normal value ranges between **75–100 mmHg**.

##### 2. Its Role in SOFA Score Calculation

In the `distributed_grpo_trainer.py` script, the model is trained to extract the **$PaO_2$** value from clinical scenario texts to calculate the **P/F Ratio**:

$$
\text{P/F Ratio} = \frac{PaO_2}{FiO_2}
$$

Case Example in Your Research:
If a patient has a **$PaO_2 = 70 \, mmHg$** (measured from ABG) and uses respiratory support with an **$FiO_2 = 0.40$**:

* **Calculation:** **$70 / 0.40 = 175$**.
* **SOFA Interpretation:** Because the result is **$175$** (falling in the range **$100 \le v < 200$**), your model must assign a **SOFA Score of 3** for the respiratory system.

**Fraction of Inspired Oxygen** or **$FiO_2$** is the concentration of oxygen in the gas or air inhaled by a person. Simply put, it shows how much of the air entering the lungs is pure oxygen.

Here are the key points to understand **$FiO_2$** in the context of your clinical research:

* **Room Air Value:** The air we breathe normally has an **$FiO_2$** of **21%** or written as a decimal **0.21**.
* **Medical Use:** When a patient experiences respiratory distress, doctors provide supplemental oxygen through devices (such as a nasal cannula or ventilator), so the **$FiO_2$** value can increase from 0.24 up to 1.00 (100% pure oxygen).
* **Role in P/F Ratio:** In the `distributed_grpo_trainer.py` script, **$FiO_2$** is used as the denominator in the **$PaO_2/FiO_2$** ratio. This ratio is crucial for determining the severity of acute lung injury or *sepsis-related organ failure*.

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

### Scoring Logic Summary

In this research, a model was trained to extract this data into a Markdown table.** If the data was not available in the medical scenario (vignette), the model was instructed to write **"N/P" (Not Provided)**and signal`<|escalate|>`to avoid dangerous guesses (hallucinations).

The total SOFA score is the sum of the six systems (range 0-24). The higher the total score, the higher the patient's risk of mortality.
