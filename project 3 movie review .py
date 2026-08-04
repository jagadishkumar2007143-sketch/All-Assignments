import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# 1. Sample data: review text + rating (1-10)
data = {
    "review": [
        "Amazing movie, loved the acting and story",
        "Best film I have seen this year",
        "Brilliant direction and great soundtrack",
        "Terrible plot, waste of time",
        "Boring and way too predictable",
        "Poor acting ruined the whole film",
        "Fantastic visuals and a moving story",
        "Awful script, fell asleep halfway",
        "Wonderful characters and a great ending",
        "Disappointing, would not recommend"
    ],
    "rating": [9, 10, 8, 2, 3, 2, 9, 1, 8, 3]
}
df = pd.DataFrame(data)

print(df)

print("\n")
# 2. Label: rating >= 6 -> Good, else Bad
df["label"] = df["rating"].apply(lambda r: "Good" if r >= 6 else "Bad")

# 3. Convert text to features
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df["review"])
y = df["label"]

print("Features shape:", X.shape)
print("Labels shape:", y.shape)

print("\n type of X:", type(X))
print(" type of y:", type(y))   


# 4. Train/test split + model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
model = MultinomialNB()
model.fit(X_train, y_train)

# 5. Evaluate
preds = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, preds))
print(classification_report(y_test, preds))

# 6. Predict a new review
new_review = ["The movie was absolutely fantastic"]
new_features = vectorizer.transform(new_review)
print("Prediction:", model.predict(new_features))