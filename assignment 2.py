import pandas as pd

 
# Data Collection

data = {
    "Name": [" VJK ", " BSH ", " VNK ", "VJP"],
    "Age": [20, 21, 23, 22 ],
    "Marks": [85, 90, 78, 85, ]
}

df = pd.DataFrame(data)

print("Original Dataset:")
print(df)
 
#  Data Preprocessinghaa
 
print("\nDataset Information:")
print(df.info())


print("\nDataset Shape:")
print(df.shape)
 
# Data Cleaning
 


print("\nMissing Values:")
print(df.isnull())

print("\nNumber of Missing Values:")
print(df.isnull().sum())


df = df.dropna()

print("\nDataset After Removing Missing Values:")
print(df)
 
# Find Duplicate Rows
 
print("\nDuplicate Rows:")
print(df.duplicated())

print("\nNumber of Duplicate Rows:")
print(df.duplicated().sum())

# Remove Duplicate Rows
df = df.drop_duplicates()

print("\nDataset After Removing Duplicate Rows:")
print(df)

 
# Final Clean Dataset
 
print("\nFinal Clean Dataset:")
print(df)
