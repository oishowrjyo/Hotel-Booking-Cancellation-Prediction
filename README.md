# Hotel Booking Cancellation Prediction

## 📌 Overview
***Hotel Booking Cancellation Prediction*** is a Machine Learning (ML) project for the course **Artificial Intelligence Lab (CSE422L)** offered by **BRAC University**, using Jupyter Notebook completed by ***Oishowrjyo Dasgupta***. This project aims to predict whether a hotel booking will be canceled based on various reservation details. By analyzing historical booking data, we can build machine learning and deep learning models to identify patterns that lead to cancellations. This helps hoteliers optimize their inventory, anticipate revenue losses, and improve overbooking strategies.

## 📊 Dataset Description
The dataset used in this project contains hotel booking information with **119,390 records (rows)** and **32 features (columns)**. 

* **Target Variable:** `is_canceled` (0 = Not Canceled, 1 = Canceled)
* **Features:** Includes a mix of 20 numerical and 12 categorical features such as:
  * Booking details: `lead_time`, `arrival_date`, `stays_in_weekend_nights`, `stays_in_week_nights`
  * Guest details: `adults`, `children`, `babies`, `is_repeated_guest`
  * Financial/Logistical details: `deposit_type`, `adr`, `required_car_parking_spaces`
  * Distribution channels and market segments.

*Note: Initial exploratory data analysis (EDA) highlights that certain columns (like `company`, `agent`, `country`, and `children`) contain missing values that require preprocessing.*

## 🛠️ Technologies & Libraries
The project utilizes a robust Python data science stack, including traditional machine learning algorithms and artificial neural networks.

* **Data Manipulation & Analysis:** `pandas`, `numpy`
* **Data Visualization:** `seaborn`, `matplotlib`
* **Machine Learning (Scikit-Learn):** 
  * Classifiers: `KNeighborsClassifier`, `DecisionTreeClassifier`, `LogisticRegression`
  * Clustering: `KMeans`
  * Preprocessing: `StandardScaler`, `VarianceThreshold`, `train_test_split`
  * Metrics: `accuracy_score`, `precision_score`, `recall_score`, `f1_score`, `confusion_matrix`, `roc_curve`, `auc`
* **Deep Learning (TensorFlow/Keras):** `Sequential`, `Dense`, `Dropout`, `Adam` optimizer

## 🚀 Project Pipeline
Based on the project structure, the workflow includes the following phases:

1. **Exploratory Data Analysis (EDA):**
   * Reviewing dataset shape, feature types (numerical vs. categorical), and basic information.
   * Generating summary statistics (mean, standard deviation, min/max).
   * Checking feature variance and evaluating skewness (symmetrical vs. highly skewed data) to inform normalization strategies.
2. **Data Preprocessing (Next Steps):**
   * Handling missing/null values.
   * Encoding categorical variables.
   * Scaling numerical features using `StandardScaler`.
   * Feature selection/reduction (e.g., using `VarianceThreshold`).
3. **Model Building & Training:**
   * Implementing baseline classification models (Logistic Regression, Decision Trees, KNN).
   * Building a Deep Neural Network using TensorFlow/Keras for complex pattern recognition.
4. **Model Evaluation:**
   * Evaluating models using Accuracy, Precision, Recall, and F1-Scores.
   * Visualizing performance using Confusion Matrices and ROC-AUC curves.

## 💻 Guideline to Run
1. Clone this repository to your local machine.
2. Ensure you have Python installed.
3. Install the required dependencies:
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn tensorflow
   ```
4. Place the `hotel_info_dataset.csv` file in the same directory as the Jupyter Notebook.
5. Open the Jupyter Notebook (`.ipynb` file) and run the cells sequentially to reproduce the data analysis and model training steps.
