import pandas as pd

df = pd.read_csv("car_data.csv")

print(df.head())
print(df.shape)
print(df.info())
print(df.isnull().sum())

df = df.dropna()
df = df.drop_duplicates()

df = df.drop('Car_Name', axis=1)

df['Car_Age'] = 2026 - df['Year']
df = df.drop('Year', axis=1)

df = pd.get_dummies(df, drop_first=True)

X = df.drop('Selling_Price', axis=1)
y = df['Selling_Price']

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =========================
# BEST MODEL FOR THIS DATA
# =========================
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score

model = GradientBoostingRegressor(
    n_estimators=200,
    learning_rate=0.1,
    max_depth=3,
    random_state=42
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Predictions:", y_pred[:5])
print("Accuracy:", r2_score(y_test, y_pred))
import matplotlib.pyplot as plt

plt.scatter(y_test, y_pred)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted Car Price")
plt.show()
import joblib

joblib.dump(model, "car_price_model.pkl")