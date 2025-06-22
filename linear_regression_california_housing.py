
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


print("Starting California Housing Dataset Analysis")

california_housing = fetch_california_housing(as_frame=True)
X = california_housing.data         
y = california_housing.target       

print("\nCalifornia Housing Dataset Loaded Successfully!")
print(f"Shape of Features (X): {X.shape}")
print(f"Shape of Target (y): {y.shape}\n")


print("A quick look at the Features (X.head()):")
print(X.head())
print("\nAnd the Target values (y.head()):")
print(y.head())

print("\nInitiating Linear Regression Modeling Process")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"\nTraining features shape: {X_train.shape}")
print(f"Test features shape: {X_test.shape}")
print(f"Training target shape: {y_train.shape}")
print(f"Test target shape: {y_test.shape}\n")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train) 
X_test_scaled = scaler.transform(X_test)    
print("Features successfully scaled using StandardScaler.\n")
linear_model = LinearRegression()
linear_model.fit(X_train_scaled, y_train)
print("Linear Regression model has been successfully trained.\n")
y_pred = linear_model.predict(X_test_scaled)
print("Evaluating the Linear Regression Model's Performance")
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)
print(f"Mean Squared Error (MSE): {mse:.3f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.3f}")
print(f"R-squared (R2) Score: {r2:.3f}")
print("\nLinear Regression Coefficients (indicating feature importance):")
coefficients_df = pd.DataFrame({'Feature': X.columns,'Coefficient': linear_model.coef_})
print(coefficients_df.sort_values(by='Coefficient', ascending=False))
print(f"\nModel Intercept: {linear_model.intercept_:.3f}")
print("\n--- Generating Visualizations for Model Insights ---")
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.3, color='skyblue', label='Predicted vs. Actual')
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2, label='Ideal Prediction Line')
plt.xlabel("Actual Median House Value ($100k)")
plt.ylabel("Predicted Median House Value ($100k)")
plt.title("Linear Regression: Actual vs. Predicted House Values")
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()
plt.tight_layout()
plt.show()
residuals = y_test - y_pred
plt.figure(figsize=(10, 6))
sns.histplot(residuals, bins=50, kde=True, color='lightcoral')
plt.title("Distribution of Residuals (Actual - Predicted)")
plt.xlabel("Residuals")
plt.ylabel("Frequency")
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()
