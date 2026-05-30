import numpy as np
import matplotlib.pyplot as plt
import cv2
import SimpleITK as sitk

import sys
sys.setrecursionlimit(100000)

sample_points = []
h_values = []



def onclick(event):
    global sample_points, hsv

    if event.xdata is not None and event.ydata is not None:

        x = int(event.xdata)
        y = int(event.ydata)

        sample_points.append((x, y))

        pixel_hsv = hsv[y, x]

        h = pixel_hsv[0]  # canal H

        h_values.append(h)

        print(f"Semilla: x={x}, y={y}, H={h}")


def grow_region(binary, visited, x, y):
    h, w = binary.shape

    if x < 0 or x >= w or y < 0 or y >= h:
        return

    if visited[y, x]:
        return

    if binary[y, x] == 0:
        return

    visited[y, x] = True

    grow_region(binary, visited, x+1, y)
    grow_region(binary, visited, x-1, y)
    grow_region(binary, visited, x, y+1)
    grow_region(binary, visited, x, y-1)


def count_rosas(binary):
    h, w = binary.shape
    visited = np.zeros_like(binary, dtype=bool)

    count = 0

    for y in range(h):
        for x in range(w):

            if binary[y, x] == 1 and not visited[y, x]:
                grow_region(binary, visited, x, y)
                count += 1

    return count


def main():
    global hsv,h_values
    img = cv2.imread("Imagenes_cursado/rosas.jpg",) #BGR
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    #ahora seleccionamos la muestra
    fig, ax = plt.subplots()
    ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    cid = fig.canvas.mpl_connect('button_press_event', onclick)

    plt.title("Click para muestrear H en HSV")
    plt.show()

    print("Valores H seleccionados:", h_values)

    h_values = np.array(h_values, dtype=np.float32)
    media = float(np.mean(h_values))
    desvio = float(np.std(h_values))

    H = hsv[:,:,0].copy()

    #umbralizamos con el valor de media +- 1.5 veces el desvio

    lower_h = media -1.5*desvio
    upper_h = media + 1.5*desvio

    mask = cv2.inRange(H, lower_h, upper_h) #usamos esta mascara para cambiar los valroes
    H[mask == 0] = 0
    
    plt.figure()
    plt.title("Canal H (Hue)")
    plt.imshow(H, cmap='gray')
    plt.colorbar()
    plt.show()

    #aplicamos un filtro de mediana para limpiar puntos que estan inconexos

    H_mediana = cv2.medianBlur(H, 5)
    plt.figure()
    plt.title("H limipiada con mediana")
    plt.imshow(H_mediana, cmap='gray')
    plt.colorbar()
    plt.show()

    #sobre esta imagen ya limpia, expandiremos para contar las rosas recorriendo la matriz de datos

    binary = (H_mediana > 0).astype(np.uint8)
    num_rosas = count_rosas(binary)
    print("Cantidad de rosas:", num_rosas)

    




    




if __name__ == "__main__":
    main()






