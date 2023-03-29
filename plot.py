import matplotlib.pyplot as plt

with open('og_balanced_accuracy.txt') as f1, \
     open('n_balanced_accuracy_0.1.txt') as f2:
    og_balanced_accuracy = f1.read()
    og_balanced_accuracy = og_balanced_accuracy.replace('[', '')
    og_balanced_accuracy = og_balanced_accuracy.replace(']', '')
    n_balanced_accuracy = f2.read()
    n_balanced_accuracy = n_balanced_accuracy.replace('[', '')
    n_balanced_accuracy = n_balanced_accuracy.replace(']', '')

print(og_balanced_accuracy)
print(n_balanced_accuracy)

og_balanced_accuracy = og_balanced_accuracy.split(", ")
og_balanced_accuracy = [float(i) for i in og_balanced_accuracy]

n_balanced_accuracy = n_balanced_accuracy.split(", ")
n_balanced_accuracy = [float(i) for i in n_balanced_accuracy]

print(n_balanced_accuracy)
print(og_balanced_accuracy)

# Define two arrays of data to plot
# TODO Conductance kommer ikke i stigende rækkefølge pga. den måde vi kører eksperimenterne på. Derfor fake dataset...
x_values = [0, 0.1666666667, 0.298245614, 0.3939393939, 0.4666666667, 0.5238095238, 0.5652173913, 0.603960396, 0.6363636364]
y1_values = og_balanced_accuracy
y2_values = n_balanced_accuracy

# Create a new figure and axis object
fig, ax = plt.subplots()

# Plot the first set of data as a blue line
ax.plot(x_values, y1_values, color='blue')

# Plot the second set of data as a red line
ax.plot(x_values, y2_values, color='red')

# Add a legend to the plot
ax.legend(['Data 1', 'Data 2'])

# Set the x and y axis labels
ax.set_xlabel('X-axis Label')
ax.set_ylabel('Y-axis Label')

# Set the title of the plot
ax.set_title('Data Plot')

# Display the plot
plt.show()