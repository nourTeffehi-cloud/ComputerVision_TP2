import sys
import os
import random
import numpy as np
import cv2
from PyQt5 import QtWidgets, uic
from PyQt5.QtWidgets import QFileDialog
from PyQt5.QtGui import QPixmap, QImage
import matplotlib.pyplot as plt


qtcreator_file = "design2.ui"
Ui_MainWindow, QtBaseClass = uic.loadUiType(qtcreator_file)


def _np_gray_to_qpixmap(img: np.ndarray) -> QPixmap:
    h, w = img.shape
    qimg = QImage(img.data, w, h, w, QImage.Format_Grayscale8)
    return QPixmap.fromImage(qimg)


class DesignWindow(QtWidgets.QMainWindow, Ui_MainWindow):
    def __init__(self):
        super().__init__()
        self.resize(1400, 900)
        self.setupUi(self)
        self.label_4.setMinimumSize(300, 180)
        self.label_8.setMinimumSize(300, 180)


        # Connexions boutons
        self.pushButton_2.clicked.connect(self.get_image)
        self.pushButton.clicked.connect(self.show_HistOriginal)
        self.pushButton_3.clicked.connect(self.show_ImgHistEqualized)
        self.pushButton_4.clicked.connect(self.show_ImgThresholding)
        self.pushButton_5.clicked.connect(self.show_ImgFiltered)
        self.pushButton_6.clicked.connect(self.show_ImgAugmented)

        self.img_gray = None

    
    def makeFigure(self, target_label, image):
        if isinstance(image, str):
            img = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
        else:
            img = image

        pix = _np_gray_to_qpixmap(img)
        target_label.setPixmap(pix)
        target_label.setScaledContents(True)


    def get_image(self):
        file, _ = QFileDialog.getOpenFileName(self, "Choisir image", "", "Images (*.jpg *.png *.jpeg)")
        if file == "":
            return

        self.img_gray = cv2.imread(file, cv2.IMREAD_GRAYSCALE)

        self.makeFigure(self.label_2, self.img_gray)


    def show_HistOriginal(self):
        if self.img_gray is None:
            return

        hist = cv2.calcHist([self.img_gray], [0], None, [256], [0, 256])

        plt.figure()
        plt.plot(hist)
        plt.title("Histogramme original")
        plt.savefig("Original_Histogram.png")
        plt.close()

        self.makeFigure(self.label_4, "Original_Histogram.png")


    def show_ImgHistEqualized(self):
        if self.img_gray is None:
            return

        equal = cv2.equalizeHist(self.img_gray)
        cv2.imwrite("Equalized_Image.png", equal)

        self.makeFigure(self.label_6, equal)

        hist = cv2.calcHist([equal], [0], None, [256], [0, 256])
        plt.figure()
        plt.plot(hist)
        plt.title("Histogramme égalisé")
        plt.savefig("Equalized_Histogram.png")
        plt.close()

        self.makeFigure(self.label_8, "Equalized_Histogram.png")


    def show_ImgThresholding(self):
        if self.img_gray is None:
            return

        if self.radioButton.isChecked():  # Binaire
            _, th = cv2.threshold(self.img_gray, 120, 255, cv2.THRESH_BINARY)
        else:  # Otsu
            _, th = cv2.threshold(self.img_gray, 0, 255, cv2.THRESH_OTSU)

        cv2.imwrite("Thresholding_Image.png", th)
        self.makeFigure(self.label_10, th)


    def show_ImgFiltered(self):
        if self.img_gray is None:
            return

        if self.radioButton_4.isChecked():
            filt = cv2.blur(self.img_gray, (11, 11))

        elif self.radioButton_3.isChecked():
            filt = cv2.GaussianBlur(self.img_gray, (15, 15), 10)

        elif self.radioButton_5.isChecked():
            filt = cv2.medianBlur(self.img_gray, 13)

        else:
            return

        cv2.imwrite("Filtered_Image.png", filt)
        self.makeFigure(self.label_12, filt)

    
    def show_ImgAugmented(self):
        if self.img_gray is None:
            return

        h, w = self.img_gray.shape

        # Rotation
        if self.radioButton_6.isChecked():
            M = cv2.getRotationMatrix2D((w//2, h//2), 45, 1)
            aug = cv2.warpAffine(self.img_gray, M, (w, h))

        # Extraction quart supérieur gauche
        elif self.radioButton_7.isChecked():
            aug = self.img_gray[:h//2, :w//2]
            aug = cv2.resize(aug, (w, h), interpolation=cv2.INTER_NEAREST)

        # Agrandissement (zoom)
        elif self.radioButton_8.isChecked():
            s = random.uniform(1.5, 4.0)
            new_w, new_h = int(w*s), int(h*s)
            big = cv2.resize(self.img_gray, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

            x = max((new_w - w) // 2,0)
            y = max((new_h - h) // 2,0)
            aug = big[y:y+h, x:x+w]
            if aug.size == 0:
                aug = cv2.resize(big, (w, h), interpolation=cv2.INTER_CUBIC)
            aug = np.ascontiguousarray(aug)

        else:
            return

        cv2.imwrite("Augmented_Image.png", aug)
        self.makeFigure(self.label_14, aug)



if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = DesignWindow()
    window.show()
    sys.exit(app.exec_())
