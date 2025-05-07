import numpy as np
import matplotlib.pyplot as plt

# Define the line and noise model
np.random.seed(42)
x_vals = np.linspace(0, 5, 10)
y_ideal_vals = 1 + 2 * x_vals
y_noisy_vals = y_ideal_vals + np.random.normal(0, 1, size=x_vals.shape)

_x_vals = np.array([1,4])
_y_ideal_vals = 1 + 2 * _x_vals

# Fitted line for display
x_line = np.linspace(0, 5, 100)
y_line = 1 + 2 * x_line

# Plot
fig, ax = plt.subplots(figsize=(6, 4), tight_layout=True)
ax.plot(x_line, y_line, label=r'$y = 1 + 2x$', color='blue')
ax.plot(_x_vals, _y_ideal_vals, 'o', color='black', label='Ideal')
ax.plot(x_vals, y_noisy_vals, 'o', color='red', label='Noisy')

# Draw vertical lines (residuals) from noisy to predicted
for x, y in zip(x_vals, y_noisy_vals):
    y_pred = 1 + 2 * x
    ax.plot([x, x], [y, y_pred], color='gray', linewidth=0.8)

# Labels and legend
ax.set_xlim(0, 5)
ax.set_ylim(0, 12)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.legend()

plt.savefig('noisy.pdf', bbox_inches='tight')
