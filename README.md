# Kepler-Exoplanet-Classification  
**Classifying Exoplanets from NASA’s Kepler Dataset: A Machine Learning Comparative Study**  

## Project Overview  
This project tackles the challenge of distinguishing confirmed exoplanets from false positives using the **NASA Kepler Exoplanet Search Dataset**. By comparing three different machine learning approaches — **Multilayer Perceptron (MLP)**, **Random Forest (RF)**, and **k-Nearest Neighbors (KNN)** — the study highlights trade-offs between accuracy, probability calibration, and inference speed in astrophysical classification tasks.  

The ultimate goal is to identify which model best fits different operational contexts — from high-stakes confirmation to rapid, low-latency screening in real-time telescope pipelines.  

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

---

## Key Results  
| Model | Accuracy | AUC | LogLoss | ECE | Inference Speed |
|-------|----------|-----|---------|-----|----------------|
| **Random Forest** | 96.45% | 0.9906 | 0.1098 | 0.0194 | 0.114 ms/sample |
| **MLP** | 95.36% | 0.9868 | 0.1414 | **0.0115** | 0.096 ms/sample |
| **KNN** | 93.03% | 0.9736 | 0.2925 | 0.0121 | **0.078 ms/sample** |

**Insights**  
- **RF**: Best overall for accuracy and precision–recall performance in high-stakes confirmation  
- **MLP**: Most stable and best-calibrated for probability ranking tasks  
- **KNN**: Fastest, ideal for real-time candidate filtering  

---
## Tech Stack
Python | scikit-learn | TensorFlow | KerasTuner | NumPy

---
## Repository Structure

kepler-exoplanet-classifier/

Kepler_Exoplanet_ML_Classification.ipynb -> main research notebook
Kepler_Exoplanet_Classification.pdf      -> full research report

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
