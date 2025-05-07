import matplotlib.pyplot as plt
import numpy as np

# Define the function and its derivatives
x = np.linspace(-3, 3, 400)
f = x**2
f_prime = 2 * x
f_double_prime = np.full_like(x, 2)

# Re-plot with updated labels using L instead of f
fig, axs = plt.subplots(1, 3, figsize=(12, 4), tight_layout=True)

axs[0].plot(x, f, label='$L(x) = x^2$')
axs[0].set_title('Function: $L(x)$')
axs[0].axhline(0, color='gray', linestyle='--', linewidth=0.5)
axs[0].axvline(0, color='gray', linestyle='--', linewidth=0.5)
axs[0].legend()

axs[1].plot(x, f_prime, label="$L'(x)$", color='orange')
axs[1].set_title('First Derivative: $L\'(x)$')
axs[1].axhline(0, color='gray', linestyle='--', linewidth=0.5)
axs[1].axvline(0, color='gray', linestyle='--', linewidth=0.5)
axs[1].legend()

axs[2].plot(x, f_double_prime, label="$L''(x)$", color='green')
axs[2].set_title('Second Derivative: $L\'\'(x)$')
axs[2].axhline(0, color='gray', linestyle='--', linewidth=0.5)
axs[2].axvline(0, color='gray', linestyle='--', linewidth=0.5)
axs[2].legend()

# Save the updated figure
pdf_path_updated = "convex-1d.pdf"
fig.savefig(pdf_path_updated)