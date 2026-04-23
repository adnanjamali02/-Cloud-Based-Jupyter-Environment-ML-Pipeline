# ML Pipeline on Iris Dataset using Cloud-Based Jupyter Environment



## 📌 Overview

This project implements a complete machine learning pipeline using the classic **Iris dataset** 🌸.  
The entire workflow is executed in a **cloud-based Jupyter environment** (Google Colab) with GPU acceleration (T4 GPU). It covers:

- Data loading and exploratory data analysis
- Model training (Random Forest Classifier)
- Performance evaluation (accuracy, confusion matrix, classification report)
- Advanced visualizations: ROC curves, learning curve, feature importance
- Model persistence (save/load using Google Drive)
- Inference on new data with confidence bar chart

---

## 🚀 Features

| Component               | Description                                                                 |
|------------------------|-----------------------------------------------------------------------------|
| **GPU Runtime**        | T4 GPU enabled for faster training (falls back to CPU if unavailable)      |
| **Google Drive Mount** | Persistent file storage; model and figures are saved to Drive              |
| **EDA Visualizations** | Pairplot, box plots, violin plots, correlation heatmap                     |
| **Model**              | Random Forest Classifier (`n_estimators=100`, `random_state=42`)           |
| **Evaluation Metrics** | Accuracy, precision, recall, F1‑score, confusion matrix                    |
| **Feature Importance** | Bar plot of Gini importance                                                |
| **Learning Curve**     | 5‑fold cross‑validation, training vs validation accuracy                   |
| **ROC Curves**         | One‑vs‑Rest ROC curves with AUC scores                                     |
| **Model Persistence**  | `joblib` serialization; model saved to Google Drive                        |
| **New Data Prediction**| Inference on a custom flower sample with confidence bar chart              |

---

## 📊 Dataset

The **Iris dataset** is built into `scikit-learn`:

- **Samples**: 150  
- **Features**: 4 (sepal length, sepal width, petal length, petal width)  
- **Classes**: setosa, versicolor, virginica (50 samples each – perfectly balanced)  
- **Train/Test Split**: 80% / 20% (stratified)

---

## 🛠️ Requirements

All dependencies are installed automatically inside the notebook.  
Key libraries:

- `tensorflow` – GPU detection  
- `scikit-learn` – Random Forest, metrics, learning curve, ROC  
- `pandas`, `numpy` – data handling  
- `matplotlib`, `seaborn` – visualizations  
- `yellowbrick` – classification report visualization  
- `joblib` – model serialization  

---

## ⚙️ How to Run

1. Open the notebook in **Google Colab**.
2. Set runtime: **Runtime → Change runtime type → T4 GPU**.
3. Execute cells in order.
4. When prompted, **authorize Google Drive** – necessary for saving the model and figures.
5. The notebook will automatically:
   - Load the Iris dataset
   - Perform EDA
   - Train the Random Forest model
   - Evaluate performance and generate all plots
   - Save model + figures to Drive
   - Predict a sample flower and show confidence

---

## 📈 Results Summary

| Metric               | Score       |
|----------------------|-------------|
| **Test Accuracy**    | ≈ 90%       |
| **AUC (setosa)**     | 1.00        |
| **AUC (versicolor)** | ≈ 0.94      |
| **AUC (virginica)**  | ≈ 0.92      |

> *Exact numbers may vary slightly due to random splits.*

**Feature Importance** (top):  
- Petal length and petal width dominate, sepal measurements contribute less.

**Confusion Matrix**: Minor misclassifications between *versicolor* and *virginica*.

---

## 🧪 Inference Example

The notebook predicts a **new flower** with measurements:
## AUTHOR: **ADNAN JAMALI**
