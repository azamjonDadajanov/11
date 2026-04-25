import matplotlib.pyplot as plt

# Data for plotting
x = [1, 2, 3, 4, 5]
y = [2, 3, 5, 7, 11]

# Create a figure and an axes
fig, ax = plt.subplots()

# Plot the data
ax.plot(x, y, marker='o', linestyle='-', color='b', label='Primes')

# Add labels and a title
ax.set_xlabel('X-axis')
ax.set_ylabel('Y-axis')
ax.set_title('Simple Matplotlib Plot')
ax.legend()

# Save the plot as an image
plt.savefig('plot.png')
print("Plot saved as plot.png")
