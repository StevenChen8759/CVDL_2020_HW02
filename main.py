# Python module import
import sys

# PyQt5 import
from PyQt5.QtWidgets import QApplication, QMainWindow
from PyQt5.uic import loadUi

# Tensorflow import

# Project module imoprt
from Background_Subtraction import bgsub
from Optical_Flow import opticalFlow
from Perspective_Transform import PerspectiveTransform
from PCA import pca
from Resnet50 import resnet50


class cvdl_hw02_window(QMainWindow):

    def __init__(self):
        super(cvdl_hw02_window, self).__init__()
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
        print("=====================================================================")
        print("---------------------1.1 Background Subtraction----------------------")
        bgsub.bgsub()

    def __butt_cb_2_1_optical_flow_preprocessing(self):
        print("=====================================================================")
        print("------------------2.1 Optical Flow - Preprocessing-------------------")
        self.kpts = opticalFlow.preprocessing()
        self.pushButton_3.setEnabled(True)

    def __butt_cb_2_2_optical_flow_video_tracking(self):
        print("=====================================================================")
        print("-----------------2.2 Optical Flow - Video Tracking-------------------")
        opticalFlow.videoTracking(self.kpts)

    def __butt_cb_3_1_perspective_transform(self):
        print("=====================================================================")
        print("---------3.1 Perspective Transform - Perspective Transform-----------")
        PerspectiveTransform.transform()

    def __butt_cb_4_1_PCA_image_reconstruction(self):
        print("=====================================================================")
        print("-------------------4.1 PCA - Image Reconstruction--------------------")
        self.reconstruction_error = pca.ImageReconstruction()
        self.pushButton_6.setEnabled(True)

    def __butt_cb_4_2_PCA_reconstruction_error_compute(self):
        print("=====================================================================")
        print("-----------------4.2 PCA - Show Reconstruction Error-----------------")
        print(self.reconstruction_error)

    def __butt_cb_5_1_Resnet50_show_accuracy(self):
        print("=====================================================================")
        print("--------------------5.1 Resnet50 - show accuracy---------------------")
        resnet50.show_train_acc("Resnet50/srcfile/resnet50_train_history_log.csv")

    def __butt_cb_5_2_Resnet50_show_tensorboard_training(self):
        resnet50.show_tensorboard_training()


    def __butt_cb_5_3_Resnet50_random_select_testing_image(self):
        print("=====================================================================")
        print("-------------5.3 Resnet50 - random select testing image--------------")
        resnet50.random_select_testdata()

    def __butt_cb_5_4_Resnet50_show_tensorboard_training(self):
        print("5.4")

if __name__=='__main__':
    app = QApplication(sys.argv)
    w = cvdl_hw02_window()
    w.show()
    sys.exit(app.exec())