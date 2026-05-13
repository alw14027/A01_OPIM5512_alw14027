from sklearn.datasets import fetch_california_housing
import pandas as pd

# Load California Housing dataset
housing = fetch_california_housing(as_frame=True)

# Features + target as a single DataFrame
df = housing.frame

# Quick check
print(df.head())
print(df.shape)

#Assistance from Gemini below
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
df.boxplot(column=['MedInc'])
plt.title('Distribution of Median Income in California')

# Save figure in "figs" folder
plt.savefig('figs/california_boxplot.png')

plt.show()
