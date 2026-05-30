import numpy as np
import cv2


class SegmentVideoHSV:

    def __init__(self, ruta_video):
        self.cap = cv2.VideoCapture(ruta_video)

        if not self.cap.isOpened():
            raise ValueError(
                f"No se pudo abrir el video: {ruta_video}"
            )

        self.h_mean = None
        self.s_mean = None

        self.h_std = None
        self.s_std = None

    def select_roi(self):
        ret, frame = self.cap.read()
        if not ret:
            raise ValueError("No se pudo leer el primer frame")

        self.first_frame = frame.copy() #nos quedamos con el primer frame
        hsv = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)

        rect = cv2.selectROI("Seleccion ROI",frame,fromCenter=False,showCrosshair=True)
        cv2.destroyWindow("Seleccion ROI")

        x, y, w, h = rect

        roi = hsv[y:y+h, x:x+w]
        H = roi[:, :, 0].astype(np.float32)
        S = roi[:, :, 1].astype(np.float32)

        self.h_mean = np.mean(H)
        self.s_mean = np.mean(S)

        self.h_std = np.std(H)
        self.s_std = np.std(S)

    #prprocesamiento del frame 
    def preprocess_frame(self, hsv, ksize=51):

        H = hsv[:, :, 0]
        S = hsv[:, :, 1]
        V = hsv[:, :, 2]

        H_blur = cv2.medianBlur(H, ksize)
        H_blur = cv2.medianBlur(H, ksize)
        H_blur = cv2.medianBlur(H, ksize)

        hsv_filtered = cv2.merge([H_blur,S,V])
        return hsv_filtered
    
    #segmentamos
    def segment_frame(self, hsv, k=6):
        H = hsv[:, :, 0].astype(np.float32)
        S = hsv[:, :, 1].astype(np.float32)
        en_H = ((H > self.h_mean - k * self.h_std) &(H < self.h_mean + k * self.h_std))

        en_S = ((S > self.s_mean - k * self.s_std) &(S < self.s_mean + k * self.s_std))
        mask = en_H & en_S
        return mask.astype(np.uint8) * 255

    def run(self, k=2, blur_size=5):
        self.select_roi()
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            hsv = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)

            hsv = self.preprocess_frame(hsv,blur_size)
            mask = self.segment_frame(hsv,k)

            result = frame.copy()
            result[mask == 0] = 0

            cv2.imshow("Original", frame)
            cv2.imshow("Mascara", mask)
            cv2.imshow("Segmentacion", result)

            key = cv2.waitKey(30) & 0xFF

            # ESC para salir
            if key == 27:
                break

        self.cap.release()
        cv2.destroyAllWindows()


def main():

    ruta = "Imagenes_cursado/pedestrians.mp4"
    seg = SegmentVideoHSV(ruta)

    seg.run(
        k=2,
        blur_size=5
    )


if __name__ == "__main__":
    main()