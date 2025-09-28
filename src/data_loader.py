from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
from scipy.spatial import distance

# Define the path to the CSV file
csv_path = Path(__file__).parent.parent / 'data' / 'cones_data.csv'

# Load the CSV file into a DataFrame
df = pd.read_csv(csv_path, skiprows=1, header=None, names=['X', 'Y', 'color'])

# Filter rows where color is 'blue'
blue_cones = df[df['color'] == 'blue']

# Filter rows where color is 'yellow'
yellow_cones = df[df['color'] == 'yellow']

# Print the first 11 rows of each filtered DataFrame for verification

print(f"Blue cones:\n{blue_cones.head(11)}")

print(f"Yellow cones:\n{yellow_cones.head(11)}")

# Convert the filtered DataFrames to NumPy arrays
blue_coords = blue_cones[['X', 'Y']].to_numpy()
yellow_coords = yellow_cones[['X', 'Y']].to_numpy()

# Print the shapes of the resulting arrays
print(f"Blue coordinates shape: {blue_coords.shape}")
print(f"Yellow coordinates shape: {yellow_coords.shape}")

# Calculate the pairwise Euclidean distance between blue and yellow cones
dist_matrix = distance.cdist(blue_coords, yellow_coords)

# Print the shape of the distance matrix
print(f"Distance matrix shape: {dist_matrix.shape}")

# Find the index of the nearest yellow cone for each blue cone
nearest_indices = np.argmin(dist_matrix, axis=1)

print(f"Nearest yellow cone indices for each blue cone: {nearest_indices}")

# Get the coordinates of the nearest yellow cones
matched_yellow = yellow_coords[nearest_indices]

print(f"Matched yellow cone coordinates:\n{matched_yellow}")

# Calculate the midpoints between each blue cone and its nearest yellow cone
midpoints = (blue_coords + matched_yellow) / 2

print(f"Midpoints between blue and nearest yellow cones:\n{midpoints}")


plt.style.use('seaborn-v0_8-deep') # Use a predefined style for better aesthetics
#print(plt.style.available) # Print available styles to choose from

# Plot blue cones
plt.scatter(blue_coords[:, 0], blue_coords[:, 1], 
c='blue', marker='x', label='Blue Cones', edgecolor='black', linewidth=1, s=50)

# Plot yellow cones
plt.scatter(yellow_coords[:, 0], yellow_coords[:, 1], 
c='gold', marker='x', label='Yellow Cones', edgecolor='black', linewidth=1, s=50)

# Plot midpoints
plt.scatter(midpoints[:, 0], midpoints[:, 1], 
c='red', marker='o', label="Midpoints", edgecolor='black', linewidth=2)

plt.xlabel('X Coordinate') # Add a label to the x-axis
plt.ylabel('Y Coordinate') # Add a label to the y-axis
plt.title('Midpoint Calculation') # Add a title to the plot
plt.legend() # Add a legend to the plot
plt.axis('equal') # Equal scaling for both axes
plt.grid(True) # Add a grid for better readability
plt.show() # Display the plot

