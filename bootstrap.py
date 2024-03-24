import numpy as np
import matplotlib.pyplot as plt

# Sample data for slope and x intercept
slopes = np.random.uniform(-2, 2, 200)
x_intercepts = np.random.uniform(-10, 10, 200)

# Create a grid of parameter combinations
slope_grid, intercept_grid = np.meshgrid(slopes, x_intercepts)

# Generate x values for the lines
x_values = np.linspace(-10, 10, 100)

# Plotting the scatter plot
plt.figure(figsize=(10, 6))
plt.scatter(slope_grid, intercept_grid, color='b', marker='o', label='Parameter Space')
plt.xlabel('Slope')
plt.ylabel('X Intercept')
plt.title('Scatter Plot of Parameter Space')
plt.legend()

# Plotting the lines defined by the parameter space
plt.figure(figsize=(10, 6))
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Lines Defined by Parameter Space')

for slope, intercept in zip(slope_grid.flatten(), intercept_grid.flatten()):
    y_values = slope * x_values + intercept
    plt.plot(x_values, y_values, alpha=0.5)

plt.show()

