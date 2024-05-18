import os
import subprocess
from converter import DatasetConverter
import yaml_creator
import random
import cald_train
import app as application


class MainApp:
    def __init__(self, classes):
        self.classes = classes
        self.train_path = os.path.join(os.getcwd(), "dataset", "train")
        self.validation_path = os.path.join(os.getcwd(), "dataset", "validation")
        self.labelme_anno_path = os.path.join(os.getcwd(), "dataset", "labelme")
        self.percentage = 0

    def mapping_classes(self):
        temp = {}
        for i, cl in enumerate(self.classes):
            temp[i] = cl

        self.classes = temp

    def create_yaml(self):
        yaml_creator.create_yaml_file(self.classes)

    def app_run(self):
        application.main(self.train_path, self.classes)


def main():
    app.create_yaml()
    app.app_run()


if __name__ == "__main__":
    print("Please enter the classes you want to detect:\n")
    classes = input().split()

    temp = {}
    for i, cl in enumerate(classes):
        temp[i] = cl

    classes = temp

    app = MainApp(classes)
    main()
