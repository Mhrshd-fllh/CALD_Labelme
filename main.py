import os
import subprocess
from converter import DatasetConverter
import yaml_creator
import random
import cald_train


class MainApp:
    def __init__(self, classes):
        self.classes = classes
        self.train_path = os.path.join(os.getcwd(), "dataset", "train")
        self.validation_path = os.path.join(os.getcwd(), "dataset", "validation")
        self.labelme_anno_path = os.path.join(os.getcwd(), "dataset", "labelme")
        self.percentage = 0
        self.zeroth_cycle = True
        self.model = cald_train.ModelConsistency(self.train_path, self.classes)

    def mapping_classes(self):
        temp = {}
        for i, cl in enumerate(self.classes):
            temp[i] = cl

        self.classes = temp

    def create_yaml(self):
        yaml_creator.create_yaml_file(self.classes)

    def app_run(self):
        path_to_execute = os.path.join(os.getcwd(), "labelme", "app.py")
        command = ["python", f"{path_to_execute}"]
        subprocess.run(command)

    def train_model(self):
        self.model.train_model()

    def non_zero_run(self):
        images = self.model.select_images()
        self.resorting(images)

    def convert_labels(self):
        conv = DatasetConverter(
            self.labelme_anno_path, self.classes, self.train_path, self.validation_path
        )
        conv.process_labelme_annotations()

    def resorting(self, train_images):
        pass

    def saving_model(self):
        pass

    def detect(self):
        pass

    def zero_cycle_run(self):
        train_images = os.listdir(os.path.join(self.train_path, "images"))
        random.shuffle(train_images)
        self.resorting(train_images)


def main():
    app.create_yaml()
    app.app_run()
    app.convert_labels()


if __name__ == "__main__":
    print("Please enter the classes you want to detect:\n")
    classes = input().split()

    temp = {}
    for i, cl in enumerate(classes):
        temp[i] = cl

    classes = temp

    app = MainApp(classes)
    main()

    def select_images():
        if app.zeroth_cycle:
            app.zero_cycle_run()
            app.zeroth_cycle = False
        else:
            app.non_zero_run()

    def train_model():
        app.train_model()
