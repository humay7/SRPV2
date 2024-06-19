import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import simps

# Example data (replace with your actual data)
iterations = np.arange(200, 1200,100)  # X axis from 1 to 10
baseline_accuracy = np.array([46.14999942779541, 63.63699722290039, 73.03099784851074, 71.99499816894532, 78.79799880981446, 86.86699752807617, 86.75099868774414, 82.65899829864502, 90.37799835205078, 91.2709976196289])  # Replace with actual baseline accuracies
my_code_accuracy = np.array([32.073, 56.822, 73.034, 79.112, 82.816, 86.736, 90.675, 86.821, 85.196, 92.098])


# Calculate AUC using NumPy's trapezoidal rule
baseline_auc = np.trapz(baseline_accuracy, iterations)
my_code_auc = np.trapz(my_code_accuracy, iterations)

# Alternatively, you can use SciPy's Simpson's rule for potentially more accurate results
baseline_auc_simps = simps(baseline_accuracy, iterations)
my_code_auc_simps = simps(my_code_accuracy, iterations)

# Print AUC values
print(f'Baseline AUC (trapezoidal): {baseline_auc}')
print(f'SRP Code AUC (trapezoidal): {my_code_auc}')
print(f'Baseline AUC (Simpson): {baseline_auc_simps}')
print(f'SRP Code AUC (Simpson): {my_code_auc_simps}')

# Plot the graphs
plt.plot(iterations, baseline_accuracy, label='Baseline Method', color='blue')
plt.plot(iterations, my_code_accuracy, label='SRP Code', color='green')
plt.fill_between(iterations, baseline_accuracy, alpha=0.1, color='blue')
plt.fill_between(iterations, my_code_accuracy, alpha=0.1, color='green')
plt.title('Accuracy Comparison Brightness30Contrast50 Baseline')
plt.xticks(iterations)
plt.xlabel('Labelled Points')
plt.ylabel('Accuracy')
plt.ylim(0, 100)  # Y axis from 0 to 100
plt.legend()
plt.show()
