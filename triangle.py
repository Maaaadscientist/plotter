import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Create a plot
plt.figure()

# Define the vertices of the triangle
triangle_vertices = [(0, 0), (1, 0), (0.5, 0.75)]

# Create a Polygon patch representing the triangle
triangle_patch = patches.Polygon(triangle_vertices, closed=True, edgecolor='none', facecolor='gray', alpha=0.3)

# Add the triangle patch to the plot
plt.gca().add_patch(triangle_patch)

# Set plot limits and labels
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Shaded Triangle Range without Lines in Matplotlib')
plt.grid(True)
plt.show()

