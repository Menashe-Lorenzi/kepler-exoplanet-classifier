# Kepler-Exoplanet-Classification  
**Classifying Exoplanets from NASA’s Kepler Dataset: A Machine Learning Comparative Study**  

## Project Overview  
This project tackles the challenge of distinguishing confirmed exoplanets from false positives using the **NASA Kepler Exoplanet Search Dataset**. By comparing three different machine learning approaches — **Multilayer Perceptron (MLP)**, **Random Forest (RF)**, and **k-Nearest Neighbors (KNN)** — the study highlights trade-offs between accuracy, probability calibration, and inference speed in astrophysical classification tasks.  

The ultimate goal is to identify which model best fits different operational contexts — from high-stakes confirmation to rapid, low-latency screening in real-time telescope pipelines.  

---

## 🔄 Project Versions & Reproducibility

This repository contains two versions of the experiment:

1.  **Legacy Version (`Kepler_Exoplanet_ML_Classitication_paper_notebook.ipynb`):**
    The original code used for the paper (pdf).

2.  **Updated Version (`Kepler_Exoplanet_ML_Classitication.ipynb`):**
    A refined version executed on newer hardware : Apple M3 Max
RAM GB: 36.0.
    * **Updates:** Model "Personality" & Edge Cases Analysis.
    * **Note:** While core findings remain consistent, slight variances in performance metrics may appear due to hardware differences and library updates.

---

## Research Questions  
1. Which machine learning model achieves the highest classification accuracy and reliability for identifying exoplanets?  
2. How does probability calibration differ between MLP, RF, and KNN, and in which scenarios is it most important?  
3. What are the trade-offs between accuracy and inference speed when applying these models to real-world astronomical pipelines?  

---

## Dataset  
- **Source**: [NASA Kepler Exoplanet Search Results](https://www.kaggle.com/datasets/nasa/kepler-exoplanet-search-results)  
- **Size**: 9,564 objects with physical and observational parameters  
- **Target Variable**: Binary label derived from `koi_disposition` (`CONFIRMED` = 1, `FALSE POSITIVE` = 0, with `CANDIDATE` removed)  
- **Key Features**: Orbital period, transit duration, transit depth, signal-to-noise ratio, and NASA false positive flags  

---

## Methodology  

**Preprocessing**  
- Dropped irrelevant identifiers and high-missingness columns  
- Removed potential leakage features like `koi_score`  
- Median imputation for missing numeric values  
- Outlier handling: created both “cleaned” and “no-treatment” datasets  
- Standardized numeric features  

**Modeling Approach**  
- **MLP**: Regularization (L2, Dropout), early stopping, hyperparameter tuning via KerasTuner  
- **RF**: OOB-based hyperparameter optimization, class balancing  
- **KNN**: Stratified k-fold cross-validation to select optimal k  

**Evaluation Metrics**  
- Primary: **LogLoss** (binary cross-entropy) for probability quality  
- Secondary: Accuracy, AUC, Average Precision (AP), Expected Calibration Error (ECE), inference time

### Advanced Evaluation Strategy
Beyond standard metrics, we conducted a **"Model vs. NASA Flags" stress test** to evaluate performance on edge cases:
1.  **"Clean Imposters" (Scenario A):** Noise signals that NASA flags missed (Flags=0, Label=False Positive).
2.  **"Wrongly Flagged" (Scenario B):** Real planets that NASA flags discarded (Flags=1, Label=Confirmed).
3.  **Confidence Analysis:** Analyzed the probability distribution of errors to understand the "personality" of each model (High-Confidence vs. Low-Confidence mistakes).

---

## Key Results  
| Model | Accuracy | AUC | LogLoss | ECE | Inference Speed |
|-------|----------|-----|---------|-----|----------------|
| **Random Forest** | 96.45% | 0.9906 | 0.1098 | 0.0194 | 0.57 ms/sample |
| **MLP** | 95.08% | 0.9860 | 0.1452 | 0.0173 | 0.029 ms/sample |
| **KNN** | 93.03% | 0.9736 | 0.2925 | 0.0121 | 0.035 ms/sample |


---

## Deep Dive: Model "Personality" & Edge Cases

While Random Forest achieved the highest global accuracy, our detailed error analysis revealed distinct "superpowers" for each model, suggesting they solve different parts of the astronomical puzzle:

### 1. MLP as the "Gatekeeper" (Detecting Hidden Noise)
In scenarios where NASA's automated flags failed (**Scenario A**: Flags=0 but Label=Noise), the **MLP correctly identified 100% of the "Clean Imposters."**
* **Insight:** This proves the neural network learned complex physical patterns (transit shape, depth consistency) rather than relying solely on the provided flags, making it an excellent auditor for "clean" candidates.

### 2. KNN as the "Scout" (Recovering Lost Planets)
Despite lower overall accuracy, **KNN successfully "rescued" 25% of real planets** that were wrongly flagged as noise by NASA (**Scenario B**).
* **Insight:** Its similarity-based approach allowed it to identify rare planetary candidates that looked like noise to rule-based systems but had neighbors with confirmed planetary status.

### 3. Confidence Analysis: Conservative vs. Bold
* **Random Forest (Conservative):** When RF errors, it does so with low confidence (~0.5-0.6), making it a safe, stable baseline for general classification.
* **KNN (Bold):** Exhibits high-confidence errors (~0.9-1.0). While risky as a standalone model, this trait makes it sensitive to outliers and useful for flagging anomalies.

### Physical Challenges & Failure Modes

Beyond the reliance on NASA flags, our error analysis identified specific physical characteristics that consistently confused the models. These "blind spots" highlight the inherent difficulty of exoplanet detection:

* **Extreme Physics (Radius & Heat):**
    * **Small Radius (`koi_prad`):** All models struggled with Earth-sized or smaller candidates. The signal produced by small planets is often indistinguishable from stellar variability.
    * **Extreme Heat (`koi_insol`):** High insolation flux caused failures, particularly for Random Forest. The models often misclassified scorching hot planets as stellar binaries due to their intense energy signatures.

* **Signal Quality (Depth & SNR):**
    * **Shallow Transits (`koi_depth`):** MLP failed significantly when transit depth was very low. The neural network struggled to lock onto the faint "dip" patterns, treating them as background noise.
    * **Low Signal-to-Noise (`koi_model_snr`):** KNN performance degraded heavily in low-SNR environments. Since KNN relies on distance metrics, "fuzzy" or noisy signals created random neighbors, leading to misclassification.

* **Temporal Dynamics (The "Fast & Slow" Paradox):**
    * **Rapid Orbits:** MLP tended to flag very short-period candidates (< 3 days) as false positives, confusing them with high-frequency instrumental noise.
    * **Long Orbits:** KNN failed on long-period candidates (> 60 days). Long orbits imply fewer observed transit events (sparse data), making it difficult for the algorithm to find statistically similar "neighbors."

---

## Strategic Conclusion: The Hybrid Pipeline

Based on these findings, we propose a multi-stage automated discovery pipeline instead of a single-model solution:

1.  **Stage 1 (RF):** Use Random Forest for rapid, high-accuracy initial classification.
2.  **Stage 2 (MLP Auditor):** Pass all "Confirmed" candidates through MLP to filter out subtle noise ("Clean Imposters") that RF might miss.
3.  **Stage 3 (KNN Discovery):** Pass all "False Positives" through KNN to flag potential "Lost Planets" for manual scientific review.

This ensemble approach leverages the **stability** of RF, the **pattern-recognition** of MLP, and the **anomaly detection** of KNN to maximize scientific discovery.

---
## Tech Stack
Python | scikit-learn | TensorFlow | KerasTuner | NumPy

---
## Repository Structure

kepler-exoplanet-classifier/

Kepler_Exoplanet_ML_Classification.ipynb -> main research notebook
Kepler_Exoplanet_ML_Classitication_paper_notebook.ipynb -> old research renotebook (Without "Model Personality Analysis & Edge Cases")
Kepler_Exoplanet_Classification.pdf -> full research report(Without "Model Personality Analysis & Edge Cases")

data/
  cumulative.csv
  kepler_cleaned_scaled.csv
  kepler_model_ready.csv
  kepler_with_metadata.csv
  kepler_with_outliers_scaled.csv

models/
  best_mlp.keras
  best_model.keras
  best_rf_params.json

figures/
  KNN/
  MLP/
  RF/
  compare/

tuner_logs/
  mlp_kepler/
  mlp_tuning/
  mlp_tuning_fix/

requirements.txt   -> python dependency list
README.md
LICENSE

---
## How To Run
pip install -r requirements.txt
python Kepler_Exoplanet_ML_Classitication.ipynb

---
## Full Report
The full research methodology, detailed experiments,
and analysis are documented in:

Kepler_Exoplanet_Classification.pdf

---
## Expected Impact  
- Improve the automation of exoplanet confirmation  
- Enable faster and more reliable pre-selection for telescope follow-up  
- Serve as a benchmark for applying ML to astronomical datasets  

---

Contact
LinkedIn: [Menashe Lorenzi](https://www.linkedin.com)   | Email: menashelorenzi@gmail.com

---

## License  
MIT License

Copyright (c) 2025 Menashe Lorenzi

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
