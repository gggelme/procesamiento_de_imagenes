import numpy as np
import cv2
import matplotlib.pyplot as plt

def calcular_fft(imagen):
    """
    Retorna la fft normal y la desplazada
    """
    F = np.fft.fft2(imagen)
    Fshift = np.fft.fftshift(F)
    magnitud = np.log(1 + np.abs(Fshift))

    return Fshift, magnitud

def mostrar_resultados(img, magnitud):

    fig, ax = plt.subplots(1,2,figsize=(14,6))

    ax[0].imshow(img,cmap='gray')
    ax[0].set_title("Imagen degradada")
    ax[0].axis('off')

    ax[1].imshow(magnitud,cmap='gray')
    ax[1].set_title("Espectro FFT")
    ax[1].axis('off')

    plt.tight_layout()

    plt.show()

def seleccionar_picos(magnitud):
    fig, ax = plt.subplots(figsize=(8,8))

    ax.imshow(magnitud, cmap='gray')
    ax.set_title("Seleccione picos del ruido")

    print("\nHaga click sobre los picos.")
    print("Cerrar la ventana para terminar.\n")

    puntos = plt.ginput(
        n=-1,
        timeout=0
    )

    plt.close()
    return puntos

def get_distancias(shape):
    P, Q = shape
    u = np.arange(P)
    v = np.arange(Q)

    U,V = np.meshgrid(u,v, indexing = "ij")
    D = np.sqrt((U-P/2)**2 + (V-Q/2)**2)
    return D #distancias de todos los puntos al centro

def filtro_rechazabanda_ideal(shape, parametros):
    D = get_distancias(shape)
    H = np.ones(shape)
    for D0, W in parametros:
        mask = (D >= (D0 - W/2)) & (D <= (D0 + W/2))
        H[mask] = 0
    return H

def filtro_rechazabanda_butterworth(shape, parametros, n=2):
    D = get_distancias(shape)
    H_final = np.ones(shape)
    for D0, W in parametros:
 
        H_i = 1 / (1 + ((D * W) / (D**2 - D0**2 + 1e-8))**(2 * n))
        H_final *= H_i 
    return H_final

def filtro_notch_ideal(shape, picos, D0):
    P, Q = shape
    U, V = np.meshgrid(np.arange(P), np.arange(Q), indexing='ij')
    centro_u, centro_v = P // 2, Q // 2
    H = np.ones(shape)

    for (px, py) in picos:
        u0 = py - centro_u
        v0 = px - centro_v
        
        D1 = np.sqrt((U - centro_u - u0)**2 + (V - centro_v - v0)**2)
        D2 = np.sqrt((U - centro_u + u0)**2 + (V - centro_v + v0)**2)
        
        H[D1 <= D0] = 0
        H[D2 <= D0] = 0
    return H

def filtro_notch_gaussiano(shape, picos, D0):
    P, Q = shape
    U, V = np.meshgrid(np.arange(P), np.arange(Q), indexing='ij')
    centro_u, centro_v = P // 2, Q // 2
    H = np.ones(shape)

    for (px, py) in picos:
        u0 = py - centro_u
        v0 = px - centro_v
        
        D1 = np.sqrt((U - centro_u - u0)**2 + (V - centro_v - v0)**2)
        D2 = np.sqrt((U - centro_u + u0)**2 + (V - centro_v + v0)**2)
        
        H_i = (1 - np.exp(-0.5 * (D1**2 / (D0**2 + 1e-8)))) * \
              (1 - np.exp(-0.5 * (D2**2 / (D0**2 + 1e-8))))
        H *= H_i
    return H

def aplicar_filtro(fshift, filtro):
    f_filtrada = fshift * filtro
    
    #deshacemos el shift
    f_ishift = np.fft.ifftshift(f_filtrada)
    return f_ishift

def plot_resultados(img, img_reconstruida, F_filtrada_shift, H):
    plt.figure(figsize=(16, 8))
    plt.subplot(2, 2, 1)
    plt.imshow(img, cmap='gray')
    plt.title('Imagen Original Degradada')
    plt.axis('off')

    plt.subplot(2, 2, 2)
    magnitud_filtrada = 20 * np.log(np.abs(F_filtrada_shift) + 1)
    plt.imshow(magnitud_filtrada, cmap='jet')
    plt.title('Espectro Filtrado (Magnitud)')
    plt.axis('off')

    plt.subplot(2, 2, 3)
    plt.imshow(H, cmap='gray')
    plt.title('Máscara del Filtro Notch')
    plt.axis('off')

    plt.subplot(2, 2, 4)
    plt.imshow(img_reconstruida, cmap='gray')
    plt.title('Imagen Reconstruida (Sin Ruido)')
    plt.axis('off')

    plt.tight_layout()
    plt.show()
    
def calcular_ecm(original, procesada):
    return np.mean((original.astype(float) - procesada.astype(float))**2)

def optimizar_filtro(img_degradada, img_original, picos):
    radios = np.arange(1, 51, 2)  # Probamos D0 desde 1 hasta 50
    errores = []
    
    fshift, _ = calcular_fft(img_degradada)
    
    for d in radios:

        H = filtro_notch_gaussiano(img_degradada.shape, picos, D0=d)
        f_filtrada = aplicar_filtro(fshift, H)
        img_back = np.abs(np.fft.ifft2(np.fft.ifftshift(f_filtrada)))
        
        res = cv2.normalize(img_back, None, 0, 255, cv2.NORM_MINMAX)
        
        error = calcular_ecm(img_original, res)
        errores.append(error)
    
    plt.figure(figsize=(8, 5))
    plt.plot(radios, errores, 'b-o')
    plt.title('Error Cuadrático Medio (ECM) vs Radio D0')
    plt.xlabel('Radio D0')
    plt.ylabel('ECM')
    plt.grid(True)
    plt.show()
    
    return radios[np.argmin(errores)]


def main():
    img = cv2.imread("Imagenes_cursado/img_degradada.tif",cv2.IMREAD_GRAYSCALE)
    Fshift, magnitud = calcular_fft(img)
    mostrar_resultados(img, magnitud)

    picos = []

    puntos = seleccionar_picos(magnitud)
    print("\nPicos seleccionados: (considerar par conjugado)\n")

    for i, p in enumerate(puntos):

        x, y = p

        print(
            f"Pico {i+1}: "
            f"x={int(x)}, y={int(y)}"
        )
        picos.append((x,y))

    
    # cambiar entre filtros
    H = filtro_notch_ideal(img.shape, picos, D0=10)
    F_filtrada_shift = aplicar_filtro(Fshift, H)

    img_back = np.fft.ifft2(np.fft.ifftshift(F_filtrada_shift))
    img_reconstruida = np.abs(img_back)
    img_reconstruida = cv2.normalize(img_reconstruida, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    plot_resultados(img, img_reconstruida, F_filtrada_shift, H)
    img_real = cv2.imread("Imagenes_cursado/img.tif", cv2.IMREAD_GRAYSCALE)
    d_opt = optimizar_filtro(img, img_real, picos)
    

    #obtencion del ruido optimo

    H_limpieza = filtro_notch_gaussiano(img.shape, picos, d_opt)
    H_solo_ruido=1 - H_limpieza
    F_ruido_shift = Fshift * H_solo_ruido

    img_solo_ruido = np.abs(np.fft.ifft2(np.fft.ifftshift(F_ruido_shift)))
    img_solo_ruido_vis = cv2.normalize(img_solo_ruido, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(H_solo_ruido, cmap='gray')
    plt.title('Filtro Notch Pasante (Máscara de Ruido)')

    plt.subplot(1, 2, 2)
    plt.imshow(img_solo_ruido_vis, cmap='gray')
    plt.title('Patrón de Ruido Extraído')
    plt.show()




    
    



if __name__ == "__main__":
    main()