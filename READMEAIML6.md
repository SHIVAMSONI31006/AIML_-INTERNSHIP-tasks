# Task 6: K-Nearest Neighbors (KNN) Classification

## Objective
The primary goal of this task is to understand, implement, and analyze the **K-Nearest Neighbors (KNN)** algorithm for resolving multiclass classification problems. Using feature engineering, parameter tuning, and evaluation metrics, this implementation demonstrates how the model segregates structured target data.

## Tools & Libraries Used
* **Python 3**
* **Scikit-learn**: For model building, dataset sourcing, data scaling, and performance metrics evaluation.
* **Pandas & NumPy**: For efficient numerical operations and multi-dimensional matrix array slicing.
* **Matplotlib**: For drawing and saving the spatial decision boundary maps.

## Dataset
The project utilizes the benchmark **Iris Dataset**. For visualization purposes and testing feature optimization, the script utilizes the spatial characteristics of the flower specimens:
1.  Sepal Length
2.  Sepal Width

## Implementation Roadmap

### 1. Data Preprocessing
* The features are separated and split into an **80/20 train-test ratio** to ensure proper validation checks.
* **Standardization** (`StandardScaler`) is applied to bring feature variables to a standard scale where $\mu = 0$ and $\sigma = 1$. This step prevents larger magnitude scales from falsely dominating distance metric computations.

### 2. Hyperparameter Tuning
The script systematically evaluates the accuracy margins across multiple values of $K$ ($K = 1, 3, 5, 7, 11$) to identify over-fitting and under-fitting thresholds.

### 3. Model Evaluation
The final configuration (optimized at $K = 5$) is validated using standard confusion matrices and comprehensive metrics (Precision, Recall, and F1-Score).

### 4. Visualization
A dense mesh grid maps out coordinate predictions across a 2D space, explicitly rendering the decision boundaries between different flower species.

---

## How to Run the Project

1. Clone this repository or download the files.
2. Install the necessary system dependencies:
   ```bash
   pip install numpy pandas matplotlib scikit-learn