import numpy as np
import cv2
import matplotlib.pyplot as plt


class SegmentImage():
    #clase segmentar una imagen delimitando una region de interes sobre ella

    def __init__(self, ruta, color_type):
        
        self.original_bgr = cv2.imread(ruta) #leemos la imagen
        self.color_type = color_type

        if self.color_type == "RGB":
            self.img = cv2.cvtColor(self.original_bgr, cv2.COLOR_BGR2RGB)  #traducir a rgb la imagen
            self.preprocess_function = self.preprocess_RGB
            self.segment_function = self.sphere_RGB
        elif self.color_type == "HSV":
            self.img = cv2.cvtColor(self.original_bgr, cv2.COLOR_BGR2HSV) #traducir  a HSV
            self.preprocess_function = self.preprocess_HSV
            self.segment_function = self.hyperplane_HSV
        

        self.roi = None
        self.mask = None


    # preprocesamiento pasabajos de la imagen 
    def preprocess_RGB(self, ksize = 5):
        #aplicamos median blur para todas las componentes
        self.img = cv2.medianBlur(self.img, ksize)
    
    def preprocess_HSV(self, ksize =5):
        #limpiamos solo H
        H = self.img[:, :, 0]
        S = self.img[:, :, 1]
        V = self.img[:, :, 2]

        H_blur = cv2.medianBlur(H, ksize)

        self.img = cv2.merge([H_blur,S,V])

    # sleccion del roi
    def select_roi(self):

        rect = cv2.selectROI(
            "Seleccion ROI",
            self.original_bgr,
            fromCenter=False,
            showCrosshair=True
        )
        cv2.destroyWindow("Seleccion ROI")
        x, y, w, h = rect

        self.roi = self.img[y:y+h, x:x+w]

    def sphere_RGB(self, k=2):

        roi = self.roi.astype(np.float32) 

        mean = np.mean(roi.reshape(-1, 3),axis=0)
        std = np.std(roi.reshape(-1, 3),axis=0)

        sigma = np.mean(std) #media por componente promedio

        img = self.img.astype(np.float32)
        dist = np.linalg.norm(img - mean,axis=2)

        self.mask = dist < (k * sigma)
        return self.mask.astype(np.uint8) * 255

    def hyperplane_HSV(self, k=2):

        roi = self.roi.astype(np.float32)

        H_roi = roi[:, :, 0]
        S_roi = roi[:, :, 1]

        h_mean = np.mean(H_roi)
        s_mean = np.mean(S_roi)

        h_std = np.std(H_roi)
        s_std = np.std(S_roi)

        H = self.img[:, :, 0].astype(np.float32)
        S = self.img[:, :, 1].astype(np.float32)

        en_H = ((H > h_mean - k * h_std) &(H < h_mean + k * h_std))
        en_S = ((S > s_mean - k * s_std) &(S < s_mean + k * s_std))

        self.mask = en_H & en_S

        return self.mask.astype(np.uint8) * 255

    def apply_mask(self):

        result = self.original_bgr.copy()
        result[~self.mask] = 0
        return result
    
    def run(self, k=2, blur_size=5):

        self.preprocess_function(blur_size)
        self.select_roi()
        mask = self.segment_function(k)
        result = self.apply_mask()

        cv2.imshow("Mascara", mask)
        cv2.imshow("Resultado", result)

        cv2.waitKey(0)
        cv2.destroyAllWindows()


# probamos

def main():

    ruta = "Imagenes_cursado/flores02.jpg"

    print("Segmentación RGB")
    seg_rgb = SegmentImage(ruta=ruta,color_type="RGB")
    seg_rgb.run(k=2,blur_size=5)

    print("Segmentación HSV")
    seg_hsv = SegmentImage(ruta=ruta,color_type="HSV")
    seg_hsv.run(k=2,blur_size=5)


if __name__ == "__main__":
    main()