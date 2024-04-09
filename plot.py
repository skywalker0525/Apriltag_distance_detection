import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

file_path = 'Large Angle(40~5 inches).xlsx'
df = pd.read_excel(file_path)


df['Error Percentage'] = (df['Calculated Distance (cm)'] - df['Real Distance (cm)']) / df['Real Distance (cm)'] * 100

# 绘图
plt.figure(figsize=(10, 6))
unique_sizes = df['Real Size (cm)'].unique()
colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_sizes)))

for size, color in zip(unique_sizes, colors):
    subset = df[df['Real Size (cm)'] == size]
    plt.scatter(subset['Real Distance (cm)'], subset['Error Percentage'], color=color, label=f'Size: {size}cm')

plt.title("Large Angle-Error Percentages by Real Distance for Different Tag Sizes")
plt.xlabel("Real Distance (cm)")
plt.ylabel("Error Percentage (%)")
plt.legend()
plt.grid(True)
plt.axhline(0, color='grey', lw=0.8)
plt.show()
