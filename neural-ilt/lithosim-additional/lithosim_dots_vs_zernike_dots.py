import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

img1_path = '/content/lithosim/t1_0_mask_ress.png'
img2_path = '/content/lithosim/output/refine_litho_out/t1_0_mask.png'

if not os.path.exists(img1_path):
    raise FileNotFoundError(f"Файл {img1_path} не найден!")
if not os.path.exists(img2_path):
    raise FileNotFoundError(f"Файл {img2_path} не найден!")
    
img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)

if img1 is None or img2 is None:
    raise ValueError("Ошибка загрузки изображений!")

diff = cv2.absdiff(img1, img2)

plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.imshow(img1, cmap='gray', vmin=0, vmax=255)
plt.title('Маска Lithosim')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(img2, cmap='gray', vmin=0, vmax=255)
plt.title('Маска с Цернике')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(diff, cmap='gray', vmin=0, vmax=255)
plt.title('Разность масок')
plt.axis('off')

plt.tight_layout()
output_path = '/content/diff_result_zernike.png'
plt.savefig(output_path, bbox_inches='tight', pad_inches=0.1, dpi=300)
plt.close()  # Закрываем фигуру, чтобы не показывалась в Colab

print(f"Результат сохранён в {output_path}")
