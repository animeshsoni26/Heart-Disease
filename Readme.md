# Heart Disease Analysis & Predictive Modeling

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![Jupyter Notebook](https://img.shields.io/badge/jupyter-%23FA0F00.svg?style=for-the-badge&logo=jupyter&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)

An end-to-end data science project utilizing the Heart Disease dataset to predict cardiovascular risk using supervised learning techniques. This project covers the full pipeline: from exploratory data analysis (EDA) to hyperparameter optimization.

## 📊 Project Architecture
The project follows a structured machine learning workflow:
1. **Exploratory Data Analysis (EDA):** Visualizing feature distributions and correlations using `Seaborn` and `Matplotlib`.
2. **Feature Engineering:** Automated preprocessing using Scikit-Learn `Pipelines` and `ColumnTransformer`.
3. **Model Selection:** Comparing classification and regression approaches.
4. **Optimization:** Tuning hyperparameters via `GridSearchCV`.
5. **Evaluation:** Analyzing residuals and classification metrics.

## 📂 Dataset Specification
The model is trained on medical clinical records (`heart_dataset.csv`) consisting of **1,888 observations**.
- **Target Variable:** `target` (Presence of heart disease).
- **Key Features:**
    - `age`, `sex`, `cp` (Chest pain type).
    - `trestbps` (Resting blood pressure).
    - `chol` (Serum cholesterol).
    - `thalach` (Maximum heart rate achieved).
    - `oldpeak` (ST depression induced by exercise).

## 🤖 Model Implementation
We implemented multiple models to compare baseline performance against optimized versions:

### Classification Models:
* **Logistic Regression:** Used as a baseline classifier.
* **K-Nearest Neighbors (KNN):** Optimized for distance-based classification.
* **Decision Tree Classifier:** Used for capturing non-linear relationships.

### Regression Analysis:
* **Decision Tree Regressor:** Implemented to predict continuous clinical outcomes and evaluate error metrics.



## 📈 Evaluation Metrics
The models are evaluated based on the following professional standards:
- **Accuracy Score:** Overall correctness of the model.
- **Confusion Matrix:** To analyze True Positives vs. False Positives.
- **Residual Analysis:** Scatter plots of `y_test - y_pred` to check for heteroscedasticity in regression.
- **Regression Errors:** Mean Absolute Error (MAE), Mean Squared Error (MSE), and $R^2$ Score.

