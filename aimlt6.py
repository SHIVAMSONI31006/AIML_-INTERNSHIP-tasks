import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from matplotlib.colors import ListedColormap

# ==========================================
# 1. LOAD AND PREPARE DATASET
# ==========================================
# Loading the classic Iris dataset
iris = load_iris()
X = iris.data[:, :2]  # Taking only the first two features (Sepal Length & Width) for 2D visualization
y = iris.target

# Splitting dataset into Training and Testing sets (80% Train, 20% Test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Feature Normalization (Scaling)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ==========================================
# 2. EXPERIMENT WITH DIFFERENT K VALUES
# ==========================================
k_values = [1, 3, 5, 7, 11]
print("--- Evaluation for Different K Values ---")

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train_scaled, y_train)
    y_pred = knn.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    print(f"K = {k} -> Accuracy: {acc * 100:.2f}%")

# ==========================================
# 3. EVALUATE FINAL MODEL (Choosing K=5)
# ==========================================
best_k = 5
final_model = KNeighborsClassifier(n_neighbors=best_k)
final_model.fit(X_train_scaled, y_train)
y_pred_final = final_model.predict(X_test_scaled)

print("\n--- Final Model Evaluation (K=5) ---")
print("Accuracy Score:", accuracy_score(y_test, y_pred_final))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred_final))
print("\nClassification Report:\n", classification_report(y_test, y_pred_final, target_names=iris.target_names))

# ==========================================
# 4. VISUALIZE DECISION BOUNDARIES
# ==========================================
# Create a mesh grid for plotting
x_min, x_max = X_train_scaled[:, 0].min() - 1, X_train_scaled[:, 0].max() + 1
y_min, y_max = X_train_scaled[:, 1].min() - 1, X_train_scaled[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                     np.arange(y_min, y_max, 0.02))

# Predict classifications for each point in the mesh grid
Z = final_model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Create color maps
cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

plt.figure(figsize=(8, 6))
plt.pcolormesh(xx, yy, Z, cmap=cmap_light, shading='auto')

# Plot training points
scatter = plt.scatter(X_train_scaled[:, 0], X_train_scaled[:, 1], c=y_train, cmap=cmap_bold, edgecolor='k', s=40)
plt.legend(handles=scatter.legend_elements()[0], labels=list(iris.target_names), loc="lower right")

plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.title(f"KNN Classification Decision Boundary (K = {best_k})")
plt.xlabel("Sepal Length (Standardized)")
plt.ylabel("Sepal Width (Standardized)")
plt.grid(True, linestyle='--', alpha=0.5)

# Save the plot visualization
plt.savefig('decision_boundary.png', dpi=300)
plt.show()