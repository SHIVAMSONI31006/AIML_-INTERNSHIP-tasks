# Binary Classification using Logistic Regression

This repository contains a Python implementation of a binary classifier built using **Logistic Regression** to analyze the Breast Cancer Wisconsin dataset.

## 📌 Project Overview
The objective of this project is to build, evaluate, and fine-tune a Logistic Regression pipeline using industry-standard tools like `Scikit-learn`, `Pandas`, and `Matplotlib`.

---

## 🛠️ Step-by-Step Implementation

### 1. Dataset Selection
We utilized the **Breast Cancer Wisconsin Dataset** available natively in `sklearn`. It contains 30 numerical features computed from digitized images of fine needle aspirates (FNA) of breast masses, predicting whether a tumor is **Malignant (0)** or **Benign (1)**.

### 2. Preprocessing & Data Splitting
* **Train/Test Split:** Split the dataset into 80% training data and 20% test data, ensuring stratification to maintain class balance.
* **Feature Standardization:** Applied `StandardScaler` to normalize features ($\mu=0, \sigma=1$). This step ensures optimal and stable gradient descent convergence during model training.

### 3. Model Training
A standard `LogisticRegression` model was fitted on the scaled training subset. The algorithm uses a linear combination of inputs wrapped inside the **Sigmoid function** to output continuous probability scores between 0 and 1.

### 4. Mathematical Explanation: The Sigmoid Function
Logistic Regression models probability using the **Sigmoid (or Logistic) function**:

$$S(z) = \frac{1}{1 + e^{-z}}$$

Where $z$ is the linear combination of weights ($w$) and features ($x$):

$$z = w_0 + w_1x_1 + w_2x_2 + \dots + w_nx_n$$

* If $z \to \infty$, $S(z) \to 1$
* If $z \to -\infty$, $S(z) \to 0$
* By default, if $S(z) \ge 0.5$, the model predicts class `1`; otherwise, it predicts `0`.

---

## 📊 Evaluation & Threshold Tuning

### Metrics Covered:
* **Confusion Matrix:** Evaluates True Positives (TP), True Negatives (TN), False Positives (FP), and False Negatives (FN).
* **Precision:** Measures quality of positive predictions ($\frac{TP}{TP + FP}$).
* **Recall (Sensitivity):** Measures ability to catch all actual positive instances ($\frac{TP}{TP + FN}$).
* **ROC-AUC:** Measures the model's ability to distinguish between classes across all possible thresholds.

### Decision Threshold Tuning
Depending on the business case (e.g., medical diagnosis), minimizing False Negatives is often crucial. The code showcases how to shift the classification threshold from the default `0.5` to a custom value (e.g., `0.4`) to artificially boost **Recall** at the cost of a small trade-off in **Precision**.

---

## 🚀 How to Run the Project in VS Code

1. Open **VS Code** and open the folder containing these files.
2. Ensure you have the necessary libraries installed via your terminal:
   ```bash
   pip install numpy pandas scikit-learn matplotlib seaborn