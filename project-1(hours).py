import pandas as pd
import matplotlib.pyplot as plt


data = {
    "Study_Hours" : [1,1.5,2,2.5,3,3.5,4,4.5,5,5.5],
    "Exam_Score" : [5,10,15,20,25,30,35,40,45,50]
}

df = pd.DataFrame(data)

print(df)


print(df.info())
print(df.isnull().sum())

# Remove duplicate rows if any
df = df.drop_duplicates()

print(df.describe())

# graph

plt.figure(figsize=(6,4))
plt.plot(df["Study_Hours"], df["Exam_Score"], marker='o')
plt.title("Study Hours vs Exam Score")
plt.xlabel("Study Hours")
plt.ylabel("Exam Score")
plt.grid(True)
plt.show()

#Bar

plt.figure(figsize=(8,5))
plt.bar(df["Study_Hours"].astype(str), df["Exam_Score"])
plt.title("Study Hours and Exam Score")
plt.xlabel("Study Hours")
plt.ylabel("Exam Score")
plt.grid(axis='y')
plt.show()

#Scatter Plot

plt.figure(figsize=(6,4))
plt.scatter(df["Study_Hours"], df["Exam_Score"])
plt.title("Scatter Plot")
plt.xlabel("Study Hours")
plt.ylabel("Exam Score")
plt.grid(axis='y')
plt.show()

#Box
plt.figure(figsize=(5,4))
plt.boxplot(df["Exam_Score"])
plt.title("Box Plot of Exam Scores")
plt.show()

Ō
