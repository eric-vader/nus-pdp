---
author: Eric Han
title: Introduction to Gradient Descent
subtitle: PDP -- Micro-teaching Component
institute: Computer Science
date: L01 -- 8 May 2025
output: beamer_presentation
toc: False
toc-title: Content
toc-depth: 2
bibliography: bibliography.bib
colorlinks: true
linkcolor: nus-orange
urlcolor: nus-blue
toccolor: black
---

## Content

By the end of this activity, you should be able to:

- **Recognize** why/how gradient descent is essential in ML training  
- **Follow** and apply the gradient descent manually on simple functions  
- **Adjust** the learning rate to observe and achieve convergence  
- **Implement** gradient descent in code and verify it works correctly

### Recap

1. Compute Complexity is the measure of the computational time needed to execute an algorithm as a function of the input size. 
1. Matrix multiplication: if $A$ is $m \times k$ and $B$ is $k \times m$, then $AB$ is $m \times m$.  
2. Matrix inversion: inverting a $m \times m$ matrix takes $O(m^3)$ time.  
1. Residual Sum of Squares: $\RSS(X)=\sum_{i=1}^{n} \left(e_i \right)^2=\sum_{i=1}^{n} \left(y_i - {\hat f \left(x_i\right)}\right)^2$

## Why study Gradient Descent?

Gradient descent and its extensions/variants are used across Machine Learning:

- **Line search**: Adaptive step size based on function values  
- **Momentum**: Combines past updates to smooth oscillations  
- **Adaptive methods**: Algorithms like Adagrad, Adam improve learning in practice  
- **Distributed algorithms**: Enable training at scale across multiple machines  
- **Second-order methods**: Use curvature information (e.g., Newton's method), though expensive for large $d$  
- **Zero-th order methods**: Optimize without gradients -- useful for black-box or simulation-based problems  (e.g., Bayesian optimization)

**Takeaway:** Gradient descent builds the foundation for understanding modern optimization.

## Linear Models

Model between a single 1D input $x$ and output $y$ by learning parameters $\theta = \begin{bmatrix} \beta_0 \\ \beta_1 \end{bmatrix}$:

$$y \sim \beta_0 + \beta_1 x$$ {#eq:linear-eq-1d}

We define $x = \begin{bmatrix} 1 & x \end{bmatrix}$ and express the function $\hat f_\theta(x)$ as:

$$\hat f_\theta(x) = \beta_0 + \beta_1 x = 
\begin{bmatrix} 1 & x \end{bmatrix}
\begin{bmatrix} \beta_0 \\ \beta_1 \end{bmatrix}
= x \theta$$ {#eq:linear-setup-1d}

Here, $\beta_1$ is the slope and $\beta_0$ is the intercept; Here generalized to $m$-dimensions:
$$\hat f_\theta(x) = \beta_0 + \beta_1 x^{(1)} + \dots + \beta_m x^{(m)} = \begin{bmatrix}
1 & x^{(1)} & \cdots & x^{(m)}
\end{bmatrix}
\Biggl[\begin{smallmatrix}
\beta_0 \\
\beta_1 \\
\cdots \\
\beta_m
\end{smallmatrix}\Biggr] = x \theta$$ {#eq:linear-setup}

---

The simplest way is to consider perfect conditions, then we can use $\hat\theta = X^{-1}Y$.

### Matrix Inversion Example (1D)

Given $n=2$ points $(1,\ 3), (4,\ 9)$, we convert to matrix form:

$$
X = \begin{bmatrix}
1 & 1 \\
1 & 4
\end{bmatrix},\quad
y = \begin{bmatrix}
3 \\
9
\end{bmatrix}
$$

Solve using: $\theta = X^{-1} y$, via calculation of inverse:

$$
X^{-1} = \frac{1}{(1)(4)-(1)(1)} \begin{bmatrix}
4 & -1 \\
-1 & 1
\end{bmatrix}
= \frac{1}{3}
\begin{bmatrix}
4 & -1 \\
-1 & 1
\end{bmatrix}
$$

Then, we apply the inverse:

$$
\theta = \frac{1}{3}
\begin{bmatrix}
4 & -1 \\
-1 & 1
\end{bmatrix}
\begin{bmatrix}
3 \\
9
\end{bmatrix}
= \frac{1}{3}
\begin{bmatrix}
12 - 9 \\
-3 + 9
\end{bmatrix}
= \frac{1}{3}
\begin{bmatrix}
3 \\
6
\end{bmatrix}
= \begin{bmatrix}
1 \\
2
\end{bmatrix}
$$

Resulting line: $\hat y = 1 + 2x$.

---

### Problems with Matrix Inversion

Noise exists, causing the $X^{-1}$ to be non-invertible.

![Estimated/Actual line (blue); Error $e_i$ (vertical line).](python/noisy.pdf){width=55%}

Estimate $\hat\theta$ by minimizing the loss function $L$: $\RSS(X)=\underbrace{\RSS(\theta)}_\text{fix $X$, find $\theta$}=L_\text{RSS}(\theta) =L(\theta)$

## Normal Equation (RSS)

We want to find the estimate $\hat\theta = \arg\min_{\theta\in\Theta} L(\theta)$ where it minimizes the loss.
$$\begin{aligned}
& L(\theta) &&= L_{\text{RSS}}(\theta) = \sum_{i=1}^n (y_i - x_i \theta)^2 
  &&&\text{\small (expand the loss)} \\
& &&=  \lVert Y - X\theta \rVert^2 = Y^\top Y - 2 Y^\top X\theta + \theta^\top X^\top X\theta
   &&&\text{\small (convert to matrices)} \\
& \nabla L(\theta) &&= \frac{\partial L_\text{RSS}}{\partial \theta} = -\left(2 X^\top \left(Y - X\theta\right)\right)
   &&&\text{\small (compute gradient)} \\
& \nabla L(\theta)=0 &&\implies \hat\theta = \underbrace{(X^\top X)^{-1} X^\top}_\text{Pseudo-inverse $X^\dag$} Y
   &&&\text{\small (Solve for $\hat\theta$)} \\
\end{aligned}$$

### Problems with Normal Equation

* **Require** closed-form gradient $\nabla L$
* **Compute Complexity** of $X^\dag$ --- takes very long to compute $(X^\top X)^{-1}$: $O(m^3)$
* **Accuracy** --- Invertiblity of $(X^\top X)^{-1}$
* **Optimality** of $\hat\theta$ --- $L(\theta)$ is convex.

## Convex Function (1D)

![A function is **convex** if it curves upwards — like a bowl — and has no dips.](python/convex-1d.pdf){width=90%}

**Definition:** A twice-differentiable function $L: \mathbb{R} \to \mathbb{R}$ is convex if: $L''(\theta) \ge 0 \quad \forall \theta \in \mathbb{R}$

**Implication:** If $L''(\theta) > 0$, $L$ is **strictly convex**, and any local minimizer is the unique global minimizer.

## Gradient Descent

Given a convex loss function $L : \mathbb{R}^m \to \mathbb{R}$ with a minimizer, an initial point $\theta^{(0)} \in \mathbb{R}^m$, step size $\gamma > 0$, and number of iterations $T > 0$, we iteratively update:
\begin{equation}
\theta^{(t+1)} = \theta^{(t)} - \gamma \nabla L(\theta^{(t)}).
\label{eq:grad-descent}
\end{equation}
Each step moves in the direction of steepest descent scaled by $\gamma$;

### Algorithm

1. Start with initial $\theta_0$, step size $\gamma$, and total steps $T$  
2. Run for each step:
   - Update $\theta^{(t+1)} = \theta^{(t)} - \gamma \nabla L(\theta^{(t)})$ 
3. Return at step T: $\theta^{(T)}$

## 5-minute Activity [L00.1]

Work in groups of 2s or 3s (split the work):

1. Given a simple function $L(x)=x^2$, 
    1. What is its first-order derivative $L'$?
    1. Solve for the minimum $x$
    1. Compute step-by-step over 3 iterations of $x$-values for $\gamma\in\{10, 1, 0.1, 0.01\}$.
    1. What did you notice?
1. [Extra] What is the Compute Complexity of Gradient Descent for $\RSS$?

### Gradient Descent Algorithm

1. Start with initial $\theta_0$, step size $\gamma$, and total steps $T$  
2. Run for each step:
   - Update $\theta^{(t+1)} = \theta^{(t)} - \gamma \nabla L(\theta^{(t)})$ 
3. Return at step T: $\theta^{(T)}$

We say Gradient Descent converges if it finds the global minimizer.

---

#### Answer

|  t |             10.0 |   1.0 |    0.1 |    0.01 |
|---:|-----------------:|------:|-------:|--------:|
|  0 |      5           |     5 | 5      | 5       |
|  1 |    -95           |    -5 | 4      | 4.9     |
|  2 |   1805           |     5 | 3.2    | 4.802   |
|  3 | -34295           |    -5 | 2.56   | 4.70596 |

1. Given a simple function $L(x)=x^2$, 
    1. First-order derivative $L'(x)=2x$
    1. Minimum is $x=0$
    1. See above, step-by-step over 3 iterations of x-values.
    1. What did we notice:
        1. Gradient Descent does not find $0$ within $T=3$, requiring more iterations.
        1. For bad values of $\gamma$, Gradient Descent does not converge.
        1. Speed of convergence depends on $\gamma$.
1. Compute Complexity is $O(nmT)$.

---

#### Illustration

See the [animation](https://eric-han.com/teaching/demo/grad-descent-animation.gif) for how gradient descent evolves over time.

![$x$-values after 10 steps of gradient descent with various learning rates on $L(x) = x^2$.](python/grad-descent-final-frame.pdf)

## Summary

**Takeaway**: Gradient descent is the foundation of modern optimization in machine learning.

- **Linear models** can be solved via matrix inversion only under ideal conditions.
- **Normal equation** avoids direct inversion of $X$, but still incurs $O(m^3)$ compute cost.
- **Convex functions** ensure any local min is global min, enabling efficient optimization.
- **Gradient descent** is a scalable, general-purpose method:
  - Works even when inversion fails.
  - Faster in most situations to compute $O(nmT)$.
  - Depends critically on the learning rate $\gamma$.
  - Converges if convex and $\gamma$ is not bad.
- Key trade-off: **simplicity vs. scalability vs. convergence speed**.

### Assignment 1 [Due in 7 days]

Implement the function `gradient_descent(grad_L, theta_0, gamma, T)` to minimize $L(\theta)$ using $T$ steps of gradient descent; Submit on Coursemology.
