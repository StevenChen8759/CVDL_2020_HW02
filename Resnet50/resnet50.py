import csv

from PIL import Image

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
    pass

def random_select_testdata():
    pass

def random_erasing_show_code():
    pass

def random_erasing_show_comparison():
    pass