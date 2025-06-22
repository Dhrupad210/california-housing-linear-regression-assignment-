# California Housing Linear Regression Analysis

## Project Overview
This repository contains a Python script that performs a Linear Regression analysis on the famous California Housing dataset. The goal is to predict the median house value in California districts based on various socio-economic and geographical features.

## Dataset
The California Housing dataset is a classic regression dataset available through scikit-learn. It contains 20,640 samples and 8 numerical features, with the target being the median house value (in hundreds of thousands of dollars).

## Libraries Used
- `numpy`
- `pandas`
- `scikit-learn`
- `matplotlib`
- `seaborn`

## How to Run the Code
1.  **Clone the repository (if you're pulling it from GitHub):**
 ```bash
    git clone [https://github.com/Dhrupad210/california-housing-linear-regression-assignment-.git](https://github.com/Dhrupad210/california-housing-linear-regression-assignment-.git)
    cd california-housing-linear-regression-assignment-
 ```
    (If you uploaded directly, you already have the file locally, just navigate to its folder)

2.  **Install the required libraries:**
    ```bash
    pip install numpy pandas scikit-learn matplotlib seaborn
    ```

3.  **Execute the Python script:**
    ```bash
    python3 linear_regression_california_housing.py
    ```

## Analysis and Results

### Model Used
A `LinearRegression` model from `sklearn.linear_model` was used. Features were scaled using `StandardScaler` to ensure optimal model performance. The data was split into 80% training and 20% testing sets.

### Evaluation Metrics
Upon running the script, the following evaluation metrics for the Linear Regression model will be displayed:
- **Mean Squared Error (MSE):** [Insert your RMSE value here, e.g., 0.523]
- **Root Mean Squared Error (RMSE):** [Insert your RMSE value here, e.g., 0.723]
- **R-squared (R2) Score:** [Insert your R2 score here, e.g., 0.590]

*(Note: Replace the bracketed values with the actual output you get when you run the script.)*

### Key Coefficients
The script also prints the coefficients for each feature, indicating their learned importance. For example, 'MedInc' (median income) is typically a strong positive predictor, while 'Longitude' might have a negative coefficient (as you move west in California, house prices might generally decrease depending on location).

### Visualizations
Two plots are generated:
1.  **Actual vs. Predicted House Values:** A scatter plot showing how well the model's predictions align with the actual median house values. Ideally, points should cluster along the diagonal line.
2.  **Distribution of Residuals:** A histogram showing the distribution of errors (actual - predicted values). For a good linear model, residuals should be approximately normally distributed around zero.

## Author
[Dhrupad]
