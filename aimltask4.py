import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix, 
    classification_report, 
    roc_auc_score, 
    roc_curve, 
    precision_recall_curve
)

# ==========================================
# 1. LOAD AND PREPARE DATASET
# ==========================================
print("Loading Breast Cancer Dataset...")
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target  # 0: Malignant, 1: Benign

print(f"Dataset shape: {X.shape}")
print(f"Class distribution: {np.bincount(y)}\n")

# ==========================================
# 2. TRAIN/TEST SPLIT AND STANDARDIZATION
# ==========================================
# 80% Train, 20% Test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Standardize features (Crucial for Logistic Regression convergence & interpretation)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ==========================================
# 3. FIT LOGISTIC REGRESSION MODEL
# ==========================================
model = LogisticRegression(random_state=42)
model.fit(X_train_scaled, y_train)

# Get predicted classes and predicted probabilities
y_pred = model.predict(X_test_scaled)
y_prob = model.predict_proba(X_test_scaled)[:, 1] 

# ==========================================
# 4. EVALUATE MODEL PERFORMANCE
# ==========================================
print("--- MODEL EVALUATION ---")
# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)

# Classification Report (Precision, Recall, F1-Score)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# ROC-AUC Score
roc_auc = roc_auc_score(y_test, y_prob)
print(f"ROC-AUC Score: {roc_auc:.4f}\n")

# ==========================================
# 5. THRESHOLD TUNING & PLOTS
# ==========================================
# Let's see how changing the threshold affects precision and recall
precisions, recalls, thresholds = precision_recall_curve(y_test, y_prob)

# Example: Custom threshold tuning (e.g., lower threshold to catch more true positives)
custom_threshold = 0.4
y_pred_custom = (y_prob >= custom_threshold).astype(int)

print(f"--- Evaluation at Custom Threshold ({custom_threshold}) ---")
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_custom))
print(classification_report(y_test, y_pred_custom))

# --- Plotting Section ---
plt.figure(figsize=(12, 5))

# Plot 1: ROC Curve
plt.subplot(1, 2, 1)
fpr, tpr, _ = roc_curve(y_test, y_prob)
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC)')
plt.legend(loc="lower right")

# Plot 2: Precision-Recall vs Threshold
plt.subplot(1, 2, 2)
plt.plot(thresholds, precisions[:-1], 'b--', label='Precision')
plt.plot(thresholds, recalls[:-1], 'g-', label='Recall')
plt.axvline(x=custom_threshold, color='red', linestyle=':', label=f'Tuned Threshold ({custom_threshold})')
plt.xlabel('Threshold')
plt.ylabel('Score')
plt.title('Precision-Recall vs. Decision Threshold')
plt.legend(loc='best')

plt.tight_layout()
plt.show()