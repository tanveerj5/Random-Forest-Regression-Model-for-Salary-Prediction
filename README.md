# 🎯 Random Forest Regression Model for Salary Prediction

## 📊 Project Overview
This project demonstrates the implementation of a **Random Forest Regression Model** to predict salaries based on employee levels. The model leverages Python's powerful libraries like `pandas`, `numpy`, `matplotlib`, and `sklearn` to deliver reliable predictions with high resolution.

---

## 🔍 Objective
The goal is to predict the salary for a specific employee level using a machine learning model trained on historical salary data.

---

## 🛠️ Tools & Libraries Used
- 🐼 **Pandas**: Data manipulation and analysis
- 🔢 **NumPy**: Numerical computations
- 📈 **Matplotlib**: Data visualization
- 🤖 **Scikit-Learn**: Machine learning model implementation

---

## 🗂️ Dataset Information
The dataset contains information about different employee levels and corresponding salaries. The key features include:
- **Position Level**: Categorical and numerical representation of job levels
- **Salary**: The target variable to predict

---

## ⚙️ Model Training
The **Random Forest Regression Model** was trained on the entire dataset using 10 decision trees with a fixed random state for reproducibility.

### Model Parameters:
- `n_estimators=10` – Number of decision trees in the forest
- `random_state=0` – Ensures consistent results across runs

### 🧠 Calculating `n_estimators`:
The `n_estimators` parameter can be determined through hyperparameter tuning techniques such as:
- **Grid Search Cross-Validation**: Systematically tests different values
- **Random Search Cross-Validation**: Randomly samples a range of values
- **Learning Curves**: Observes model performance with varying tree counts
- **Empirical Testing**: Start with a baseline (e.g., 100) and adjust based on performance and computational resources

### 🧮 Formula-based Calculation:
- The `n_estimators` value can also be determined using a heuristic approach:
  - Start with a smaller number and incrementally increase it until performance plateaus.
  - Use `cross_val_score` from `sklearn.model_selection` to assess performance improvements.

---

## 🔢 Prediction
The model predicts the salary for an employee with a level of **6.5**.

### 🎯 Single Value Prediction:
- The model predicts salary for a single input level.

### 🔍 Multiple Value Prediction:
- The model can predict salaries for the entire dataset.

---

## 📊 Visualization Insights
### Initial Visualization:
- **Scatter Plot**: Displays the actual salary data points
- **Line Plot**: Shows the model's predictions across different levels

### High-Resolution Visualization:
- A finer granularity of predictions using a **0.01 step size** ensures a smoother and more accurate curve representation.

---

## 📈 Model Evaluation
### Metrics Used:
- **Mean Squared Error (MSE)**: Measures average squared difference between predicted and actual values.
- **Root Mean Squared Error (RMSE)**: Square root of the MSE.
- **Mean Absolute Error (MAE)**: Average absolute difference between predicted and actual values.
- **R² Score**: Proportion of variance explained by the model.

### 📋 Evaluation Outcome:
The evaluation metrics provide a comprehensive view of the model's accuracy and reliability.

### 🏆 Best Metric Selection:
- **RMSE (Root Mean Squared Error)** is often considered the best metric for regression tasks as it penalizes larger errors more than MAE, providing a clearer picture of the model's performance.
- **R² Score** is useful for understanding the proportion of variance explained but does not convey error magnitude directly.

---

## 🚀 Key Takeaways
- **Random Forest** delivers robust predictions by combining multiple decision trees.
- Fine-grained visualization helps in understanding model behavior.
- MSE, RMSE, MAE, and R² offer a well-rounded performance assessment.
- **RMSE** is generally the most informative metric when comparing model performance.

🔗 **Project Highlight:** Successfully applied machine learning techniques to predict salaries accurately based on position levels.

---

🌟 _This project showcases my skills in machine learning, data visualization, and Python programming._

