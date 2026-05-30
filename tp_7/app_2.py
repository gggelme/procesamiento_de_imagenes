import cv2
import numpy as np


class LataDetector:

    def __init__(self, path):

        self.img = cv2.imread(path)

        if self.img is None:
            raise ValueError("No se pudo cargar la imagen")

        self.gray = None
        self.blurred = None
        self.circles = None

        self.grandes = 0
        self.chicas = 0

    def preprocess(self):

        self.gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        self.blurred = cv2.medianBlur(self.gray, 5)

        return self.blurred
    
    def detect(self):

        self.circles = cv2.HoughCircles(
            self.blurred,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=100,
            param1=70,
            param2=50,
            minRadius=50,
            maxRadius=200
        )

        return self.circles

    def classify(self):

        self.grandes = 0
        self.chicas = 0

        if self.circles is None:
            return 0, 0, 0

        circles = np.uint16(np.around(self.circles))

        for x, y, r in circles[0]:

            if r > 120:
                self.grandes += 1
                color = (0, 255, 0)
            else:
                self.chicas += 1
                color = (255, 0, 0)

            cv2.circle(self.img, (x, y), r, color, 2)
            cv2.circle(self.img, (x, y), 2, (0, 0, 255), 3)

        total = len(circles[0])

        return total, self.grandes, self.chicas
    
    def run(self):

        self.preprocess()
        self.detect()

        total, grandes, chicas = self.classify()

        print("===================================")
        print(f"Total de latas: {total}")
        print(f"Grandes: {grandes}")
        print(f"Chicas: {chicas}")
        print("===================================")

        cv2.imshow("Latas detectadas", self.img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def main():

    detector = LataDetector("Imagenes_cursado/latas.png")
    detector.run()


if __name__ == "__main__":
    main()