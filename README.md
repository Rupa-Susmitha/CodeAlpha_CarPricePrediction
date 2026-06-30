# 🚗 Car Price Prediction using Machine Learning

This project predicts the selling price of used cars using Machine Learning techniques.

---

## 📌 Project Overview
The goal of this project is to build a regression model that can predict car prices based on features such as:
- Year
- Present Price
- Driven Kms
- Fuel Type
- Transmission
- Owner

---

## 🧠 Steps Performed

1. Data Loading
2. Data Cleaning (removed null values & duplicates)
3. Feature Engineering (created Car Age from Year)
4. Encoding categorical variables
5. Model Training using Machine Learning
6. Model Evaluation using R² Score
7. Visualization of results

---

## 📊 Model Used
- Random Forest Regressor / Gradient Boosting Regressor

---

## 📈 Results

- Model Accuracy (R² Score): ~0.73 (approx)

---

## 🖼️ Visualization

### Actual vs Predicted Car Prices

![Graph](images/actual_vs_predicted.png)

---

## 📁 Project Structure
CodeAlpha_CarPricePrediction/
│
├── car_data.csv
├── car_price_prediction.py
├── car_price_model.pkl
├── README.md
└── images/
└── actual_vs_predicted.png


---

## ⚙️ Technologies Used

- Python 🐍
- Pandas
- NumPy
- Scikit-learn
- Matplotlib

---

## 🚀 How to Run

1. Clone the repository
2. Install required libraries:

pip install pandas numpy matplotlib scikit-learn

3. Run the script:

python car_price_prediction.py


---

## 🎯 Outcome

This project helped in understanding:
- Regression models
- Data preprocessing
- Feature engineering
- Model evaluation



