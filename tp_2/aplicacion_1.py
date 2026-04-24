
import cv2
import numpy as np
import matplotlib.pyplot as plt 


def main():
    img_path='../Imagenes_cursado/earth.bmp'
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    print(img)

    #yo en lo personal haria mas blanca para ver los curepos celestes del fondo

    r = np.arange(256)
    c = 255 / np.log(1 + 255)
    lut = (c * np.log(1 + r)).astype(np.uint8)

    img_realzada = cv2.LUT(img, lut)

    #dibujo
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    axs[0].imshow(img, cmap='gray', vmin=0, vmax=255)
    axs[0].set_title("Imagen Original")
    axs[1].imshow(img_realzada, cmap='gray', vmin=0, vmax=255)
    axs[1].set_title("Imagen Realzada")

    plt.show()

if __name__ == "__main__":
    main()
