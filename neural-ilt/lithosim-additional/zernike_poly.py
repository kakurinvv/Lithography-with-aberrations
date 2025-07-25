import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import math

def zernike_polynomial(n, m, size=256):
    x = np.linspace(-1, 1, size)
    y = np.linspace(-1, 1, size)
    X, Y = np.meshgrid(x, y)

    r = np.sqrt(X**2 + Y**2)
    theta = np.arctan2(Y, X)
    mask = (r <= 1).astype(float)

    R = np.zeros_like(r)
    for k in range((n - abs(m)) // 2 + 1):
        coef = ((-1)**k * math.factorial(n - k) /
               (math.factorial(k) *
                math.factorial((n + abs(m))//2 - k) *
                math.factorial((n - abs(m))//2 - k)))
        R += coef * r**(n - 2*k)

    if m > 0:
        Z = R * np.cos(m * theta)
    elif m < 0:
        Z = R * np.sin(abs(m) * theta)
    else:
        Z = R

    return Z * mask

def plot_combined_zernike(coeffs, size=512):
    n_m_list = [
        (0,0),    # Z0
        (1,-1),   # Z1
        (1,1),    # Z2
        (2,-2),   # Z3
        (2,0),    # Z4
        (2,2),    # Z5
        (3,-3),   # Z6
        (3,-1),   # Z7
        (3,1),    # Z8
        (3,3),    # Z9
        (4,0),    # Z10
        (4,2),    # Z11
        (4,-2),   # Z12
        (4,4),    # Z13
        (4,-4)    # Z14
    ]

    combined = np.zeros((size, size))

    for i, coeff in enumerate(coeffs):
        if i >= len(n_m_list):
            break
        n, m = n_m_list[i]
        combined += coeff * zernike_polynomial(n, m, size)

    max_val = np.max(np.abs(combined))
    v_range = max_val if max_val > 0 else 1

    plt.figure(figsize=(10, 8))
    im = plt.imshow(combined,
                   cmap='jet',
                   extent=[-1, 1, -1, 1],
                   norm=Normalize(vmin=-v_range, vmax=v_range))

    cbar = plt.colorbar(im, fraction=0.046, pad=0.04)
    cbar.set_label('Значение', fontsize=12)

    circle = plt.Circle((0, 0), 1, fill=False,
                       color='white', linestyle='--', linewidth=1)
    plt.gca().add_patch(circle)
    plt.grid(True, alpha=0.3)

    title = "Комбинация полиномов Цернике\n"
    title += " + ".join([f"{coeff:.2f}·Z{i}" for i, coeff in enumerate(coeffs) if coeff != 0])
    plt.title(title, fontsize=14, pad=20)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()

#coeffs = [1, 0.9, -0.2, 0.8, 1.0, 0.9, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
coeffs = [0.0, 0.7, 0.7, 2.0, 0.9, 0.0, 0.0, 1.2, 1.1, 5.0, 0.0]
plot_combined_zernike(coeffs)
