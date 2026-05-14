import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing

# --- PART 1: DATA PREPARATION ---

# Load California Housing dataset
housing = fetch_california_housing(as_frame=True)

# Combine features and target into a single DataFrame
df = housing.frame

# Quick check of the data
print("Data Head:")
print(df.head())
print(f"\nDataFrame Shape: {df.shape}")

# --- PART 2: DIRECTORY MANAGEMENT ---

# Define the directory name
output_dir = 'figs'

# Check if the directory exists; if not, create it.
# This prevents FileNotFoundError when calling savefig.
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    print(f"Directory '{output_dir}' created.")
else:
    print(f"Directory '{output_dir}' already exists.")

# --- PART 3: PLOTTING ---

plt.figure(figsize=(10, 6))

# Create the boxplot for Median Income
df.boxplot(column=['MedInc'])

plt.title('Distribution of Median Income in California')
plt.ylabel('Income (in tens of thousands of $)')

# --- PART 4: SAVING AND SHOWING ---

# Use a relative path to save into the project folder
file_path = os.path.join(output_dir, 'california_boxplot.png')

# Save the figure
plt.savefig(file_path)
print(f"Figure saved successfully to: {file_path}")

# Display the plot
plt.show()