# Python module import
import sys

# PyQt5 import
from PyQt5.QtWidgets import QApplication, QMainWindow
from PyQt5.uic import loadUi

# Tensorflow import

# Project module imoprt
from Resnet50 import resnet50


class cvdl_hw01_dl_window(QMainWindow):

    def __init__(self):
        super(cvdl_hw01_dl_window, self).__init__()
        loadUi('cvdl_2020_hw02.ui', self)

        # Add callback to each button
        self.pushButton.clicked.connect(self.__butt_cb_1_1_bg_subtraction)
        self.pushButton_2.clicked.connect(self.__butt_cb_2_1_optical_flow_preprocessing)
        self.pushButton_3.clicked.connect(self.__butt_cb_2_2_optical_flow_video_tracking)
        self.pushButton_4.clicked.connect(self.__butt_cb_3_1_perspective_transform)
        self.pushButton_5.clicked.connect(self.__butt_cb_4_1_PCA_image_reconstruction)
        self.pushButton_6.clicked.connect(self.__butt_cb_4_2_PCA_reconstruction_error_compute)
        self.pushButton_7.clicked.connect(self.__butt_cb_5_1_Resnet50_show_accuracy)
        self.pushButton_8.clicked.connect(self.__butt_cb_5_2_Resnet50_show_tensorboard_training)
        self.pushButton_9.clicked.connect(self.__butt_cb_5_3_Resnet50_random_select_testing_image)
        self.pushButton_10.clicked.connect(self.__butt_cb_5_4_Resnet50_show_tensorboard_training)

    def __butt_cb_1_1_bg_subtraction(self):
        print("1.1")

    def __butt_cb_2_1_optical_flow_preprocessing(self):
        print("2.1")

    def __butt_cb_2_2_optical_flow_video_tracking(self):
        print("2.2")

    def __butt_cb_3_1_perspective_transform(self):
        print("3.1")

    def __butt_cb_4_1_PCA_image_reconstruction(self):
        print("4.1")

    def __butt_cb_4_2_PCA_reconstruction_error_compute(self):
        print("4.2")

    def __butt_cb_5_1_Resnet50_show_accuracy(self):
        print("5.1")

    def __butt_cb_5_2_Resnet50_show_tensorboard_training(self):
        print("5.2")

    def __butt_cb_5_3_Resnet50_random_select_testing_image(self):
        print("5.3")

    def __butt_cb_5_4_Resnet50_show_tensorboard_training(self):
        print("5.4")

if __name__=='__main__':
    app = QApplication(sys.argv)
    w = cvdl_hw01_dl_window()
    w.show()
    sys.exit(app.exec())