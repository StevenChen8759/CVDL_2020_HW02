# Python module import
import sys

# Project module imoprt
from utils import dataset_op
from utils.constants import *
from utils import model_op

# PyQt5 import
from PyQt5.QtWidgets import QApplication, QMainWindow
from PyQt5.uic import loadUi

# Tensorflow import

class cvdl_hw01_dl_window(QMainWindow):

    def __init__(self):
        super(cvdl_hw01_dl_window, self).__init__()
        loadUi('dl_select.ui', self)

        # Add callback to each button
        self.pushButton.clicked.connect(self.__butt_cb_show_train_img)
        self.pushButton_2.clicked.connect(self.__butt_cb_show_hyper_param)
        self.pushButton_3.clicked.connect(self.__butt_cb_show_model_struct)
        self.pushButton_4.clicked.connect(self.__butt_cb_show_inference_acc)
        self.pushButton_5.clicked.connect(self.__butt_cb_inference)

    def __butt_cb_show_train_img(self):
        print("Randomly pick 10 images in training set and show its label...")
        dataset_op.cifar10_img_show()

    def __butt_cb_show_hyper_param(self):
        print("Hyperparameter:")
        print("Batch size: %d" % TRAIN_BATCH_SIZE)
        print("Learning Rate: %f" % LEARNING_RATE)
        print("Optimizer: %s" % OPTIMIZER)

    def __butt_cb_show_model_struct(self):
        model_op.show_struct()

    def __butt_cb_show_inference_acc(self):
        model_op.show_acc()

    def __butt_cb_inference(self):
        model_op.do_inference(self.spinBox.value())

if __name__=='__main__':
    app = QApplication(sys.argv)
    w = cvdl_hw01_dl_window()
    w.show()
    sys.exit(app.exec())