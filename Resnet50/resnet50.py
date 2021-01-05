import csv
import random

import matplotlib.pyplot as plt

# Show Resnet50 Training Accuracy
def show_train_acc(history_file_path):

    with open(history_file_path, 'r') as history_file:
        reader = csv.DictReader(history_file, delimiter=',')
        title = reader.fieldnames
        for titleItem in title:
            print(titleItem, end="  ")
        print("")

        for row in reader:
            print("%s      %s      %s  %s           %s" % (
                    row["epoch"], row["accuracy"][0:4],
                    row["loss"][0:4], row["val_accuracy"][0:4],
                    row["val_loss"][0:4]))

def show_tensorboard_training():
    tb_acc_loss = plt.imread("./Resnet50/srcfile/tensorboard_acc.jpg")

    plt.title("Tensorboard Training Process")
    plt.imshow(tb_acc_loss)
    plt.show()

def random_select_testdata():
    idx_all = random.randint(0, 24999)

    plt.figure(figsize=(1,1))
    if idx_all >= 12500:
        new_idx = idx_all // 2 # Integer division

        # Select picture(s) from Dog set
        img = plt.imread("./Resnet50/srcfile/dataset_ASIRRA/PetImages/Dog/%d.jpg" % new_idx)
        plt.title("Dog(1)")
    else:
        new_idx = idx_all

        # Select picture(s) from Cat set
        img = plt.imread("./resnet50/srcfile/dataset_ASIRRA/PetImages/Cat/%d.jpg" % new_idx)
        plt.title("Cat(0)")

    figManager = plt.get_current_fig_manager()
    figManager.window.showMaximized()

    plt.imshow(img)
    plt.show()

def random_erasing_show_code():
    pass

def random_erasing_show_comparison():
    pass