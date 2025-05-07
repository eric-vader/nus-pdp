
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Gradient and function definitions
def L(x):
    return x**2

def grad_L(x):
    return 2 * x

# Learning rates and initializations
learning_rates = [10.0, 1.0, 0.1, 0.01]
theta_0 = 5
T = 10

# Store trajectories
trajectories = {gamma: [theta_0] for gamma in learning_rates}

# Run gradient descent and record theta values
for gamma in learning_rates:
    theta = theta_0
    for _ in range(1, T):
        theta = theta - gamma * grad_L(theta)
        trajectories[gamma].append(theta)

# x range for plotting
x_vals = np.linspace(-6, 6, 400)
y_vals = L(x_vals)

# Recreate side-by-side plots
fig, axs = plt.subplots(1, 4, figsize=(16, 4), sharey=True, tight_layout=True)
lines = {}
dots = {}

for i, gamma in enumerate(learning_rates):
    axs[i].plot(x_vals, y_vals, 'k--', alpha=0.5)
    axs[i].set_xlim(-6, 6)
    axs[i].set_ylim(0, 36)
    axs[i].set_title(f'$\gamma = {gamma}$')
    lines[gamma], = axs[i].plot([], [], 'o-', color='blue', label='Past Points')
    dots[gamma], = axs[i].plot([], [], 'o', color='orange', markersize=8, label='Current Point')

def init():
    for gamma in learning_rates:
        lines[gamma].set_data([], [])
        dots[gamma].set_data([], [])
    return list(lines.values()) + list(dots.values())

def animate(i):
    for gamma in learning_rates:
        x_data = trajectories[gamma][:i+1]
        y_data = [L(x) for x in x_data]
        
        if i > 0:
            lines[gamma].set_data(x_data, y_data)  # past points (blue)
        else:
            lines[gamma].set_data([], [])

        # current point (orange)
        dots[gamma].set_data([x_data[-1]], [y_data[-1]])  # wrap in list
    return list(lines.values()) + list(dots.values())


ani = animation.FuncAnimation(fig, animate, frames=T, init_func=init, blit=True, repeat=False)


# Save animation as GIF
ani.save("grad-descent-animation.gif", writer="pillow", fps=1)

fig.savefig("grad-descent-final-frame.pdf")
