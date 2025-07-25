import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

print("Проверка файлов:")
img1_path = '/content/lithosim/output/refine_litho_out/t1_0_mask.png'
img2_path = '/content/lithosim/output/refine_net_output/t1_0_mask.png'
print("img1 exists:", os.path.exists(img1_path))
print("img2 exists:", os.path.exists(img2_path))

img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)

if img1 is None or img2 is None:
    print("Ошибка загрузки изображений!")
else:
    diff = cv2.absdiff(img1, img2)
    
    plt.figure(figsize=(15, 5))
    plt.subplot(131), plt.imshow(img1, cmap='gray'), plt.title('Маска 1 (Test)')
    plt.subplot(132), plt.imshow(img2, cmap='gray'), plt.title('Маска 2 (Lithosim)')
    plt.subplot(133), plt.imshow(diff, cmap='gray'), plt.title('Разность')

    plt.savefig('/content/diff_result.png')
    print("Результат сохранён в /content/diff_result.png")
    plt.show()
