import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

 
data = {
    "Bedrooms": [2, 3, 4, 3, 5, 2, 4, 3, 5, 4,
                 2, 3, 4, 5, 3, 4, 2, 5, 4, 3],

    "Bathrooms": [1, 2, 3, 2, 4, 1, 3, 2, 4, 3,
                  1, 2, 3, 4, 2, 3, 1, 4, 3, 2],

    "Size_sqft": [900, 1200, 1800, 1400, 2500,
                  850, 2000, 1300, 2700, 2100,
                  800, 1250, 1900, 2800, 1500,
                  2200, 950, 2600, 1850, 1350],

    "Season": [
        "Winter", "Summer", "Summer", "Winter", "Summer",
        "Winter", "Spring", "Summer", "Summer", "Spring",
        "Winter", "Spring", "Summer", "Summer", "Winter",
        "Spring", "Winter", "Summer", "Spring", "Winter"
    ],

    "City": [
        "Vijayawada", "Hyderabad", "Hyderabad", "Vijayawada",
        "Bangalore", "Vijayawada", "Bangalore", "Hyderabad",
        "Bangalore", "Hyderabad", "Vijayawada", "Bangalore",
        "Hyderabad", "Bangalore", "Vijayawada", "Hyderabad",
        "Vijayawada", "Bangalore", "Hyderabad", "Vijayawada"
    ],

    # House price in lakhs
    "Price_Lakhs": [
        35, 55, 85, 60, 120,
        32, 100, 62, 135, 105,
        30, 58, 90, 145, 65,
        110, 38, 130, 88, 63
    ]
}


df = pd.DataFrame(data)
 

print("HOUSE PRICE DATASET")
print(df)

print("\nDataset Information:")
print(df.info())

print("\nStatistical Analysis:")
print(df.describe())

 

X = df.drop("Price_Lakhs", axis=1)
y = df["Price_Lakhs"]

 

categorical_columns = ["Season", "City"]

numeric_columns = [
    "Bedrooms",
    "Bathrooms",
    "Size_sqft"
]


 

preprocessor = ColumnTransformer(
    transformers=[
        ("categorical",
         OneHotEncoder(handle_unknown="ignore"),
         categorical_columns)
    ],
    remainder="passthrough"
)


 

model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("regressor", LinearRegression())
    ]
)


 

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


 

model.fit(X_train, y_train)

 

y_pred = model.predict(X_test)

 
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nMODEL PERFORMANCE")
print("Mean Absolute Error:", mae)
print("R2 Score:", r2)


 
new_house = pd.DataFrame({
    "Bedrooms": [3],
    "Bathrooms": [2],
    "Size_sqft": [1500],
    "Season": ["Summer"],
    "City": ["Hyderabad"]
})

predicted_price = model.predict(new_house)

print("\nNEW HOUSE DETAILS")
print(new_house)

print(
    "\nEstimated House Price:",
    round(predicted_price[0], 2),
    "Lakhs"
)


plt.scatter(y_test, y_pred)

plt.xlabel("Actual Price (Lakhs)")
plt.ylabel("Predicted Price (Lakhs)")
plt.title("Actual vs Predicted House Prices")

plt.show()
