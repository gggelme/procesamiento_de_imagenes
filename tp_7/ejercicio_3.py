import numpy as np
import matplotlib.pyplot as plt
import cv2
import SimpleITK as sitk

import sys
sys.setrecursionlimit(100000)

seed_point = []
T = 15



def onclick(event):
    global seed_point

    # verificar que el click fue dentro de la imagen
    if event.xdata is not None and event.ydata is not None:

        x = int(event.xdata)
        y = int(event.ydata)

        seed_point.append((x,y))

        print(f"Semilla seleccionada: x={x}, y={y}")

def grow_region(img,region,visited,x,y,seed_value):

    h, w = img.shape
    if x < 0 or x >= w or y < 0 or y >= h:
        return #tiene que cortar para que no salga error

    if visited[y, x]:
        return #si ya lo visito que se vuelva nomas

    visited[y, x] = True #sino que lo ponga que ya lo visitó
    pixel_value = img[y, x]

    if abs(int(pixel_value) - int(seed_value)) > T:
        return
    region[y, x] = 255

    grow_region(img, region, visited, x+1, y, seed_value)
    grow_region(img, region, visited, x-1, y, seed_value)
    grow_region(img, region, visited, x, y+1, seed_value)
    grow_region(img, region, visited, x, y-1, seed_value)




def main():
    img = cv2.imread("Imagenes_cursado/PostSpinalRodsAP.jpg", cv2.IMREAD_GRAYSCALE)
    fig, ax = plt.subplots(figsize=(8,8))

    ax.imshow(img, cmap="gray")
    ax.set_title("Click para seleccionar semilla")

    
    fig.canvas.mpl_connect('button_press_event', onclick)
    plt.show()

    print("Semilla final:", seed_point)

    #expansion

    region = np.zeros_like(img, dtype=np.uint8)
    visited = np.zeros_like(img, dtype=bool)

    for (x_seed, y_seed) in seed_point: #para todo los puntos
        seed_value = img[y_seed, x_seed]
        grow_region(img,region,visited,x_seed,y_seed,seed_value)


    colored = cv2.applyColorMap(region,cv2.COLORMAP_JET)

    plt.figure(figsize=(14,6))

    plt.subplot(1,2,1)
    plt.imshow(img, cmap='gray')
    plt.title("Imagen original")
    plt.axis("off")

    plt.subplot(1,2,2)
    plt.imshow(cv2.cvtColor(colored, cv2.COLOR_BGR2RGB))
    plt.title(f"Region growing | T={T}")
    plt.axis("off")

    plt.show()

    # ahora utilizando simpleitk
    img_sitk = sitk.GetImageFromArray(img)
    
    #vamos a acumular las regiones
    seg_total = sitk.Image(img_sitk.GetSize(), sitk.sitkUInt8)
    seg_total.CopyInformation(img_sitk)

    for (x, y) in seed_point:

        seed_value = int(img[y, x])

        lower = seed_value - T
        upper = seed_value + T

        seg = sitk.ConnectedThreshold(
            img_sitk,
            seedList=[(x, y)],
            lower=lower,
            upper=upper
        )

        seg_total = sitk.Maximum(seg_total, seg)

    seg_np = sitk.GetArrayFromImage(seg_total)


    plt.figure(figsize=(6,6))
    plt.imshow(seg_np, cmap="gray")
    plt.title("SimpleITK Region Growing")
    plt.axis("off")
    plt.show()

if __name__ == "__main__":
    main()






