import numpy as np
import matplotlib.pyplot as plt
from sympy import symbols, exp, abs, fourier_transform, ifourier_transform, simplify

# Symbolic computation with SymPy
x, omega = symbols("x omega", real=True)

# Define f(x)
f_x = exp(-abs(x))

# Fourier transform of f(x)
F_omega = fourier_transform(f_x, x, omega)

# Inverse Fourier transform of F(omega)
f_x_reconstructed = simplify(ifourier_transform(F_omega, omega, x))

# Numerical computation for visualization
omega_vals = np.linspace(-10, 10, 1000)
F_omega_numeric = 2 / (1 + omega_vals**2)

# Visualization
plt.figure(figsize=(12, 6))

# Plot original function in time domain
plt.subplot(1, 2, 1)
x_vals = np.linspace(-10, 10, 1000)
f_x_numeric = np.exp(-np.abs(x_vals))
plt.plot(x_vals, f_x_numeric, label=r"$f(x) = e^{-|x|}$")
plt.title("Original Function $f(x)$")
plt.xlabel("$x$")
plt.ylabel("$f(x)$")
plt.grid(True)
plt.legend()

# Plot Fourier transform in frequency domain
plt.subplot(1, 2, 2)
plt.plot(
    omega_vals,
    F_omega_numeric,
    label=r"$\mathcal{F}[f(x)] = \frac{2}{1 + \omega^2}$",
    color="orange",
)
plt.title("Fourier Transform $\mathcal{F}[f(x)]$")
plt.xlabel("$\omega$")
plt.ylabel("$\mathcal{F}[f(x)]$")
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()
