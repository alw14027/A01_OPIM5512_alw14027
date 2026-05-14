import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing

# 1. Load Data
housing = fetch_california_housing(as_frame=True)
df = housing.frame

# 2. Setup Folder Path 
# Since this script is inside 'src', we go UP one level then into 'figs'
output_dir = os.path.join(os.path.dirname(__file__), '..', 'figs')

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 3. Create Plot
plt.figure(figsize=(10, 6))
df.boxplot(column=['MedInc'])
plt.title('Distribution of Median Income in California')

# 4. Save to the specific path required by the rubric
save_path = os.path.join(output_dir, 'boxplot.png')
plt.savefig(save_path)
print(f"Success! Figure saved to {save_path}")