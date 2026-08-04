import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

np.random.seed(42)

cities = ['Mumbai', 'Delhi', 'Bangalore', 'Pune', 'Chennai']
seasons = ['Winter', 'Summer', 'Monsoon', 'Spring']

n_samples = 500

city_base_price = {
    'Mumbai': 9000000,
    'Delhi': 7500000,
    'Bangalore': 6500000,
    'Pune': 5000000,
    'Chennai': 4500000
}

season_multiplier = {
    'Winter': 1.05,   # peak buying season
    'Summer': 0.97,
    'Monsoon': 0.93,
    'Spring': 1.02
}

data = []
for _ in range(n_samples):
    city = np.random.choice(cities)
    season = np.random.choice(seasons)
    rooms = np.random.randint(1, 6)  # 1 to 5 rooms

    base = city_base_price[city]
    price = base + (rooms - 1) * 800000
    price *= season_multiplier[season]
    price += np.random.normal(0, 300000)  # noise
    price = max(price, 500000)  # floor

    data.append([rooms, season, city, round(price, 2)])

df = pd.DataFrame(data, columns=['rooms', 'season', 'city', 'price'])

# Save the generated dataset so you can inspect/replace it
df.to_csv('/mnt/user-data/outputs/house_price_data.csv', index=False)

print("=" * 60)
print("DATA OVERVIEW")
print("=" * 60)
print(df.head())
print("\nShape:", df.shape)
print("\nSummary statistics:")
print(df.describe())
print("\nMissing values:\n", df.isnull().sum())


#  DATA ANALYSIS / VISUALIZATION


sns.set_style("whitegrid")
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Price by number of rooms
sns.boxplot(x='rooms', y='price', data=df, ax=axes[0, 0])
axes[0, 0].set_title('Price Distribution by Number of Rooms')

# Price by city
sns.boxplot(x='city', y='price', data=df, ax=axes[0, 1])
axes[0, 1].set_title('Price Distribution by City')
axes[0, 1].tick_params(axis='x', rotation=30)

# Price by season
sns.boxplot(x='season', y='price', data=df, ax=axes[1, 0])
axes[1, 0].set_title('Price Distribution by Season')

# Correlation heatmap (numeric only)
sns.heatmap(df[['rooms', 'price']].corr(), annot=True, cmap='coolwarm', ax=axes[1, 1])
axes[1, 1].set_title('Correlation Heatmap')

plt.tight_layout()
plt.savefig('/mnt/user-data/outputs/data_analysis.png', dpi=150)
print("\n[Saved] data_analysis.png")

 
# MODEL BUILDING
 
X = df[['rooms', 'season', 'city']]
y = df['price']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Preprocessing: one-hot encode categorical columns, pass rooms through
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(drop='first'), ['season', 'city'])
    ],
    remainder='passthrough'  # keeps 'rooms' as-is
)

model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])

model.fit(X_train, y_train)


# EVALUATION
 
y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("\n" + "=" * 60)
print("MODEL EVALUATION")
print("=" * 60)
print(f"MAE  : {mae:,.2f}")
print(f"RMSE : {rmse:,.2f}")
print(f"R^2  : {r2:.4f}")

# Actual vs Predicted plot
plt.figure(figsize=(7, 6))
plt.scatter(y_test, y_pred, alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()],
         'r--', label='Perfect prediction')
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')
plt.title('Actual vs Predicted House Prices')
plt.legend()
plt.tight_layout()
plt.savefig('/mnt/user-data/outputs/actual_vs_predicted.png', dpi=150)
print("[Saved] actual_vs_predicted.png")

# PREDICT NEW HOUSE PRICE (example usage)

def predict_price(rooms, season, city):
    """Predict the price of a house given rooms, season, and city."""
    input_df = pd.DataFrame([[rooms, season, city]],
                             columns=['rooms', 'season', 'city'])
    return model.predict(input_df)[0]


example = predict_price(rooms=3, season='Winter', city='Bangalore')
print("\n" + "=" * 60)
print("EXAMPLE PREDICTION")
print("=" * 60)
print(f"3-room house, Winter, Bangalore -> Estimated price: {example:,.2f}")

# Try a few more examples
examples = [
    (2, 'Summer', 'Pune'),
    (4, 'Monsoon', 'Mumbai'),
    (1, 'Spring', 'Chennai'),
]
for rooms, season, city in examples:
    price = predict_price(rooms, season, city)
    print(f"{rooms}-room house, {season}, {city} -> Estimated price: {price:,.2f}")