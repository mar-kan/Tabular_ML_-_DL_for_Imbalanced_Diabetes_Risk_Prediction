# Evaluating Tabular Deep Learning Architectures against Classical Machine Learning for Imbalanced Diabetes Risk Prediction

**Author:** Marianna Kanellaki  
**Institution:** German University of Digital Science  
**Contact:** marianna@kanellakis.eu  

---

## 📌 Project Overview
This repository contains the code and notebooks for evaluating the efficacy of classical machine learning algorithms versus state-of-the-art tabular deep learning architectures on highly imbalanced public health data. 

The objective is to predict multi-class diabetes risk (Healthy, Prediabetes, Diabetes) using the **Behavioral Risk Factor Surveillance System (BRFSS) 2015 dataset**. The project addresses severe class imbalance (84% healthy subjects) and high multicollinearity among health indicators, successfully mitigating the "Accuracy Paradox" through algorithmic class weighting.

## 📂 Repository Structure

* `Part1_Exploration_Proposal.ipynb`: 
  * Dataset origin, structure, and relevance.
  * Extensive Exploratory Data Analysis (EDA) (Univariate and Bivariate analysis).
  * Problem definition and identification of challenges (e.g., severe class imbalance).
  * Feature selection utilizing XGBoost feature importance (selecting the top 16 critical features).
* `Part2_Modeling_Implementation.ipynb`:
  * End-to-end experimental pipeline and data preprocessing (Scaling, Stratified Splitting).
  * Handling class imbalance (Benchmarking SMOTE vs. Class Weighting).
  * Model development, hyperparameter tuning (via Optuna and GridSearch), and training.
  * Comprehensive benchmarking and comparative analysis of 5 distinct architectures.
  * Evaluation metrics mapping (Accuracy, Precision, Recall, Macro F1, and Multiclass One-vs-Rest ROC-AUC).

## 📊 Dataset
* **Source:** CDC's Behavioral Risk Factor Surveillance System (BRFSS) 2015.
* **Total Instances:** 253,680 survey responses.
* **Target Variable:** 3-class ordinal feature:
  * `0`: No Diabetes (84% / 213,703 cases)
  * `1`: Prediabetes (2% / 4,631 cases)
  * `2`: Diabetes (14% / 35,346 cases)

## 🧠 Architectures Evaluated
1. **Multinomial Logistic Regression (Baseline):** Utilizing L2 (Ridge) regularization.
2. **XGBoost:** Gradient-boosted tree ensemble for native non-linear mapping.
3. **Multi-Layer Perceptron (MLP):** PyTorch-based foundational deep learning benchmark.
4. **TabNet:** Sequential attention-based neural network incorporating sparsity regularization.
5. **FT-Transformer:** State-of-the-art transformer architecture tokenizing continuous and categorical features.

## 🏆 Key Results
* **Best Macro F1-Score:** **XGBoost (0.5119)** demonstrated superior stability in inter-class classification and explicit interpretability.
* **Best ROC-AUC:** **PyTorch MLP (0.7803)** achieved the highest overall discriminative ability across all thresholds.
* **Conclusion:** Algorithmic class weighting is strictly necessary to prevent minority class omission. While XGBoost is optimal for immediate clinical applications requiring transparency, deep tabular architectures (MLP, TabNet, FT-Transformer) demonstrate highly competitive discriminative capacities.

## ⚙️ Requirements & Installation

To run the notebooks locally, ensure you have Python 3.12 and the library versions provided in **requirements.txt**.
