import cv2
import numpy as np


class WebcamRectDetector:

    def __init__(self, cam_index=0):

        self.cap = cv2.VideoCapture(cam_index)
        if not self.cap.isOpened():
            raise ValueError("No se pudo abrir la webcam")

        self.lines = None
        self.rect = None

    def preprocess(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        return blur
    
    def edges(self, frame):
        return cv2.Canny(frame, 50, 150)
    
    def detect_lines(self, edges):

        lines = cv2.HoughLinesP(edges,rho=1,theta=np.pi / 180,threshold=80,minLineLength=60,maxLineGap=10)

        self.lines = lines
        return lines
    
    def split_lines(self, angle_tol=10):
        # de todas los segmentos identificados separamos aquellos que sean horizontales o verticales

        horizontal = []
        vertical = []

        if self.lines is None: #sino ecntonro
            return horizontal, vertical

        for line in self.lines:
            x1, y1, x2, y2 = line[0]

            #recta que los une
            angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))

            #lienas horizontales
            if abs(angle) < angle_tol:
                horizontal.append((x1, y1, x2, y2))

            #lineas verticales
            elif abs(abs(angle) - 90) < angle_tol:
                vertical.append((x1, y1, x2, y2))
            
            #else esta inclinada

        return horizontal, vertical
    
    def build_rectangle(self, horizontal, vertical):

        if len(horizontal) < 2 or len(vertical) < 2:
            self.rect = None
            return None

        ys = []
        xs = []

        for x1, y1, x2, y2 in horizontal:
            ys.extend([y1, y2])

        for x1, y1, x2, y2 in vertical:
            xs.extend([x1, x2])

        x1, x2 = int(min(xs)), int(max(xs))
        y1, y2 = int(min(ys)), int(max(ys))

        self.rect = (x1, y1, x2, y2)

        return self.rect
    
    def draw(self, frame):

        out = frame.copy()
        if self.rect is not None:
            x1, y1, x2, y2 = self.rect

            cv2.rectangle(out,(x1, y1),(x2, y2),(0, 255, 0),3)

            cv2.putText(out,"RECTANGULO DETECTADO",(x1, y1 - 10),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0, 255, 0),2)

        return out
    
    def run(self):

        while True:

            ret, frame = self.cap.read()

            if not ret:
                break

            # pipeline
            pre = self.preprocess(frame)
            edg = self.edges(pre)
            self.detect_lines(edg)

            h, v = self.split_lines()
            self.build_rectangle(h, v)

            result = self.draw(frame)

            # mostrar
            cv2.imshow("Edges", edg)
            cv2.imshow("Rect Detector", result)

            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                break

        self.cap.release()
        cv2.destroyAllWindows()

def main():

    detector = WebcamRectDetector(0)
    detector.run()


if __name__ == "__main__":
    main()