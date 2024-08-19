import os
from application import yaml_creator

import app as application


class MainApp:
    def __init__(self):
        self.train_path = os.path.join(os.getcwd(), "dataset", "train")
        self.validation_path = os.path.join(os.getcwd(), "dataset", "validation")
        self.labelme_anno_path = os.path.join(os.getcwd(), "dataset", "labelme")
        self.percentage = 0

    def app_run(self):
        application.main(self.train_path)


def main():
    os.mkdir("dataset")
    os.mkdir("dataset/labelme")
    os.mkdir("dataset/train")
    os.mkdir("dataset/validation")
    os.mkdir("dataset/train/images")
    os.mkdir("dataset/train/labels")
    os.mkdir("dataset/validation/images")
    os.mkdir("dataset/validation/labels")
    app.app_run()


if __name__ == "__main__":
    app = MainApp()
    main()
