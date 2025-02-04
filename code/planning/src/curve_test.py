import numpy as np
from scipy.interpolate import splprep, splev
import matplotlib.pyplot as plt

# sinus trajectorie
t = np.linspace(0, 2*np.pi, 10)
trajectory = np.array([t, np.sin(t)]).T

# Trajektorie glätten mit einem Spline
tck, u = splprep([trajectory[:, 0], trajectory[:, 1]], s=0.0009)  # s=Glättungsfaktor
u_fine = np.linspace(0, 1, 1000)  # feinere Abtastung
x_smooth, y_smooth = splev(u_fine, tck)

# Numerische Ableitungen der geglätteten Trajektorie
dx, dy = splev(u_fine, tck, der=1)
ddx, ddy = splev(u_fine, tck, der=2)

# Krümmung berechnen
curvature = (dx * ddy - dy * ddx) / (dx**2 + dy**2)**1.5

# Abstand entlang der geglätteten Trajektorie berechnen
distance = np.cumsum(np.sqrt(np.diff(x_smooth)**2 + np.diff(y_smooth)**2))
distance = np.insert(distance, 0, 0)  # Start bei 0 m

# Krümmung im Bereich der ersten 10 m
curvature_in_range = curvature[distance <= 10]

# Mittlere Krümmung in den ersten 10 m
mean_curvature = np.mean(curvature_in_range)
print(f"Mittlere Krümmung in den ersten 10 m: {mean_curvature:.4f}")

# Plot der geglätteten Trajektorie und Krümmung
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(trajectory[:, 0], trajectory[:, 1], 'o-', label='Original Trajektorie')
plt.plot(x_smooth, y_smooth, '-', label='Geglättete Trajektorie')
plt.xlabel('x [m]')
plt.ylabel('y [m]')
plt.legend()
plt.title('Trajektorie')

plt.subplot(1, 2, 2)
plt.plot(distance, curvature, label='Krümmung')
plt.axvline(10, color='r', linestyle='--', label='10 m Bereich')
plt.xlabel('Abstand entlang der Trajektorie [m]')
plt.ylabel('Krümmung [1/m]')
plt.legend()
plt.title('Krümmung entlang der Trajektorie')

plt.tight_layout()
plt.savefig('curve_test.png')
