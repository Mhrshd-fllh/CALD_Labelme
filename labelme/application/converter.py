import os
import json
import random
import shutil


class DatasetConverter:
    def __init__(
        self, input_dir, label_mapping, train_path, validation_path, unlabeled
    ):
        self.input_dir = input_dir
        self.label_mapping = label_mapping
        self.train_path = os.path.join(train_path, "labels")
        self.validation_path = os.path.join(validation_path, "labels")
        self.train_images = os.path.join(train_path, "images")
        self.validation_images = os.path.join(validation_path, "images")
        self.unlabeled = unlabeled

    def process_labelme_annotations(self, train_labeled_set, validation_labeled_set):
        # Process each JSON file in the input_directory
        labels = os.listdir(self.input_dir)
        random.shuffle(labels)
        n = len(labels)
        train_labels = labels[: int(0.8 * n)]
        validation_labels = labels[int(0.8 * n) :]
        train_labeled_set = train_labels
        validation_labeled_set = validation_labels
        for filename in train_labeled_set:
            if filename.endswith(".json"):
                self.labelme_json_path = os.path.join(self.input_dir, filename)
                self.yolo_txt_path = os.path.join(
                    self.train_path, f"{os.path.splitext(filename)[0]}.txt"
                )
                shutil.copy(
                    os.path.join(
                        self.unlabeled, f"{os.path.splitext(filename)[0]}.jpg"
                    ),
                    os.path.join(
                        self.train_images, f"{os.path.splitext(filename)[0]}.jpg"
                    ),
                )
                train_labeled_set.append(f"{os.path.splitext(filename)[0]}.jpg")
                self.labelme_to_yolo()

        # print("train done")
        for filename in validation_labeled_set:
            if filename.endswith(".json"):
                self.labelme_json_path = os.path.join(self.input_dir, filename)
                self.yolo_txt_path = os.path.join(
                    self.validation_path, f"{os.path.splitext(filename)[0]}.txt"
                )
                shutil.copy(
                    os.path.join(
                        self.unlabeled, f"{os.path.splitext(filename)[0]}.jpg"
                    ),
                    os.path.join(
                        self.validation_images, f"{os.path.splitext(filename)[0]}.jpg"
                    ),
                )
                validation_labeled_set.append(f"{os.path.splitext(filename)[0]}.jpg")
                self.labelme_to_yolo()
        return train_labeled_set, validation_labeled_set

    def labelme_to_yolo(self):
        with open(self.labelme_json_path, "r") as json_file:
            data = json.load(json_file)
        image_height = data["imageHeight"]
        image_width = data["imageWidth"]

        with open(self.yolo_txt_path, "w") as yolo_file:

            for shape in data["shapes"]:
                label = shape["label"]
                points = shape["points"]
                label = {
                    i for i in self.label_mapping if self.label_mapping[i] == label
                }
                x, y, w, h = self.get_yolo_coordinates(
                    points, image_height, image_width
                )
                yolo_file.write(f"{label.pop()} {x} {y} {w} {h}\n")

    def get_yolo_coordinates(self, points, image_height, image_width):
        x = (points[0][0] + points[1][0]) / (2 * image_width)
        y = (points[0][1] + points[1][1]) / (2 * image_height)
        w = abs(points[1][0] - points[0][0]) / image_width
        h = abs(points[1][1] - points[0][1]) / image_height

        return x, y, w, h
