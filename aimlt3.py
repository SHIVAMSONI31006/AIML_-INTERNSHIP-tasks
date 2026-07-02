import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# --- Step 1: Import and Preprocess the Dataset ---
# Creating a dummy dataset (Years of Experience vs Salary)
data = {
    'Years_Experience': [1.1, 1.3, 1.5, 2.0, 2.2, 2.9, 3.0, 3.2, 3.2, 3.7, 3.9, 4.0, 4.0, 4.1, 4.5],
    'Education_Level': [12, 12, 14, 16, 14, 16, 16, 16, 18, 16, 18, 16, 16, 18, 18], # Included for potential multiple regression
    'Salary': [39343, 46205, 37731, 43525, 39891, 56642, 60150, 54445, 64445, 57189, 63218, 55794, 56957, 57081, 61111]
}
df = pd.DataFrame(data)
print("--- Dataset Preview ---")
print(df.head(), "\n")

# Setting up Features (X) and Target (y) for Simple Linear Regression
X = df[['Years_Experience']] # Independent Variable
y = df['Salary']            # Dependent Variable

# --- Step 2: Split data into train-test sets ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Step 3: Fit a Linear Regression model ---
model = LinearRegression()
model.fit(X_train, y_train)

# Generating Predictions
y_pred = model.predict(X_test)

# --- Step 4: Evaluate model using MAE, MSE, R² ---
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("--- Model Evaluation ---")
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"R-squared (R² Score): {r2:.2f}\n")

# Understanding Intercept and Coefficients
print("--- Interpretation of Coefficients ---")
print(f"Intercept (w0): {model.intercept_:.2f}")
print(f"Coefficient (w1 - Slope): {model.coef_[0]:.2f}")
print(f"Equation: Salary = {model.intercept_:.2f} + ({model.coef_[0]:.2f} * Years_Experience)\n")

# --- Step 5: Plot regression line ---
plt.figure(figsize=(8, 5))
plt.scatter(X_train, y_train, color='blue', label='Training Data')
plt.scatter(X_test, y_test, color='green', label='Testing Data')
plt.plot(X_train, model.predict(X_train), color='red', linewidth=2, label='Regression Line')
plt.title('Salary vs Experience (Linear Regression)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.legend()
plt.grid(True)
plt.show()