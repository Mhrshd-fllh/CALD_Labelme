import os
import subprocess
from converter import DatasetConverter
import yaml_creator


class MainApp:
    def __init__(self, classes):
        self.classes = classes
        self.train_path = os.path.join(os.getcwd(), "dataset", "train")
        self.validation_path = os.path.join(os.getcwd(), "dataset", "validation")
        self.labelme_anno_path = os.path.join(os.getcwd(), "dataset", "labelme")
        self.percentage = 0
        self.zeroth_cycle = True

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
        pass

    def convert_labels(self):
        conv = DatasetConverter(
            self.labelme_anno_path, self.classes, self.train_path, self.validation_path
        )
        conv.process_labelme_annotations()

    def resorting(self):
        pass

    def saving_model(self):
        pass

    def detect(self):
        pass

    def zero_cycle_run(self):
        pass


def main():
    app.mapping_classes()
    app.create_yaml()
    app.app_run()
    app.convert_labels()


if __name__ == "__main__":
    print("Please enter the classes you want to detect:\n")
    classes = input().split()
    app = MainApp(classes)
    main()

    def train_request():
        if app.zeroth_cycle:
            app.zero_cycle_run()
        else:
            app.train_model()
