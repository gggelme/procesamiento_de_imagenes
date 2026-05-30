import cv2
import numpy as np


class SegmentWebcamHSV:

    def __init__(self):

        self.cap = cv2.VideoCapture(0)

        if not self.cap.isOpened():
            raise ValueError("No se pudo abrir webcam")

        cv2.namedWindow("Original | Segmentacion HSV")

        cv2.createTrackbar("H min","Original | Segmentacion HSV",0,179,self.nothing)
        cv2.createTrackbar("H max","Original | Segmentacion HSV",179,179,self.nothing)
        cv2.createTrackbar("S min","Original | Segmentacion HSV",0,255,self.nothing)
        cv2.createTrackbar("S max","Original | Segmentacion HSV",255,255,self.nothing)
        cv2.createTrackbar("V min","Original | Segmentacion HSV",0,255,self.nothing)
        cv2.createTrackbar("V max","Original | Segmentacion HSV",255,255,self.nothing)
        cv2.createTrackbar("Blur","Original | Segmentacion HSV",5,15,self.nothing)

    def nothing(self, x):
        pass

    def get_values(self):

        h_min = cv2.getTrackbarPos("H min","Original | Segmentacion HSV")
        h_max = cv2.getTrackbarPos("H max","Original | Segmentacion HSV")
        s_min = cv2.getTrackbarPos("S min","Original | Segmentacion HSV")
        s_max = cv2.getTrackbarPos("S max","Original | Segmentacion HSV")
        v_min = cv2.getTrackbarPos("V min","Original | Segmentacion HSV")
        v_max = cv2.getTrackbarPos("V max","Original | Segmentacion HSV")
        blur = cv2.getTrackbarPos("Blur","Original | Segmentacion HSV")

        if blur % 2 == 0:
            blur += 1

        if blur < 1:
            blur = 1

        return (h_min, h_max,s_min, s_max,v_min, v_max,blur)

    def preprocess_HSV(self, hsv, blur):

        H = hsv[:, :, 0]
        S = hsv[:, :, 1]
        V = hsv[:, :, 2]

        H_blur = cv2.medianBlur(H, blur)

        hsv_filtered = cv2.merge([H_blur,S,V])
        return hsv_filtered

    def segment_HSV(self,hsv,h_min, h_max,s_min, s_max,v_min, v_max):

        lower = np.array([h_min,s_min,v_min])
        upper = np.array([h_max,s_max,v_max])
        mask = cv2.inRange(hsv,lower,upper)

        return mask

    def run(self):

        while True:

            ret, frame = self.cap.read()
            if not ret:
                break

            (h_min, h_max,s_min, s_max,v_min, v_max,blur) = self.get_values()
            hsv = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)

            hsv = self.preprocess_HSV(hsv,blur)
            mask = self.segment_HSV(hsv,h_min, h_max,s_min, s_max,v_min, v_max)
            result = cv2.bitwise_and(frame,frame,mask=mask)

            combined = np.hstack([frame,result])

            cv2.imshow("Original | Segmentacion HSV",combined)
            key = cv2.waitKey(1) & 0xFF

            if key == 27:
                break

        self.cap.release()
        cv2.destroyAllWindows()

def main():

    seg = SegmentWebcamHSV()

    try:
        seg.run()

    finally:
        seg.cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()