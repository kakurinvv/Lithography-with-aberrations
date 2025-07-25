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

def plot_zernike_grid(polynomials, titles, rows, cols, figsize=(15, 12)):
    """Создает сетку графиков полиномов Цернике"""
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    axes = axes.ravel()
    
    z_min, z_max = -1, 1
    norm = Normalize(vmin=z_min, vmax=z_max)

    for i, ((n, m), title) in enumerate(zip(polynomials, titles)):
        ax = axes[i]
        Z = zernike_polynomial(n, m)
        im = ax.imshow(Z, cmap='jet', extent=[-1, 1, -1, 1], norm=norm)
        ax.set_title(title, fontsize=10)
        ax.axis('off')
        
        circle = plt.Circle((0, 0), 1, fill=False, 
                          color='white', linestyle='--', linewidth=0.5)
        ax.add_patch(circle)

    # Скрываем пустые ячейки
    for i in range(len(polynomials), rows*cols):
        axes[i].axis('off')

    
    return fig

# Первые 15 полиномов Цернике (n, m) и их названия
polynomials = [
    (0, 0), (1, -1), (1, 1), 
    (2, -2), (2, 0), (2, 2),
    (3, -3), (3, -1), (3, 1), (3, 3),
    (4, -4), (4, -2), (4, 0), (4, 2), (4, 4)
]

titles = [
    'Z(0,0)\nPistон', 'Z(1,-1)\nНаклон X', 'Z(1,1)\nНаклон Y',
    'Z(2,-2)\nАстигматизм 45°', 'Z(2,0)\nДефокус', 'Z(2,2)\nАстигматизм 0°',
    'Z(3,-3)\nТрифой 30°', 'Z(3,-1)\nКома X', 'Z(3,1)\nКома Y', 'Z(3,3)\nТрифой 0°',
    'Z(4,-4)\nКвадрофой 22.5°', 'Z(4,-2)\nВторичный астигматизм 45°',
    'Z(4,0)\nСферич. абер.', 'Z(4,2)\nВторичный астигматизм 0°', 'Z(4,4)\nКвадрофой 0°'
]

fig = plot_zernike_grid(polynomials, titles, rows=3, cols=5, figsize=(16, 14))
fig.suptitle('Первые 15 полиномов Цернике', fontsize=16, y=0.95)
plt.tight_layout()
plt.show()
