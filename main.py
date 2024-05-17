# import os
# import subprocess
# import converter
# import random
# import shutil
# import yaml_creator
# import cald_train


# def run_labelme(input_dir, output_dir):
#     # Replace 'labelme' with the actual path to the labelme executable
#     labelme_executable = os.path.join(
#         os.getcwd(), "labelme_project", "labelme", "labelme", "app.py"
#     )

#     # Example command to run LabelMe
#     command = ["python", labelme_executable, input_dir, "-O", output_dir]

#     # Run LabelMe using subprocess
#     subprocess.run(command)


# def move_files(source_dir, destination_dir, num_files):
#     all_files = os.listdir(source_dir)

#     selected_files = random.sample(all_files, min(num_files, len(all_files)))

#     for filename in selected_files:
#         if filename.endswith(".md"):
#             continue
#         source_path = os.path.join(source_dir, filename)
#         destination_path = os.path.join(destination_dir, filename)
#         shutil.move(source_path, destination_path)


# def main():
#     # Saving main directories
#     input_dir = os.path.join(os.getcwd(), "labelme_project", "dataset", "input_files")

#     # Getting Info from user
#     print(
#         "Please move your dataset to 'labelme_project/dataset/input_files/all_images' to start training active learning object detection."
#     )
#     inp = input("Please enter the classes that you want to detect\n")
#     classes = [cl for cl in inp.split()]
#     label_mapping = {}
#     for i, cls in enumerate(classes):
#         label_mapping[i] = cls
#     yaml_creator.create_yaml_file(label_mapping)
#     cycle_number = int(input("Enter Cycle Number you want to train yolo:\n"))

#     # move_files(os.path.join(input_dir, 'all_images'), os.path.join(input_dir, 'train', 'images'), 200)
#     move_files(
#         os.path.join(input_dir, "all_images"),
#         os.path.join(input_dir, "validation", "images"),
#         100,
#     )
#     yolo = cald_train.ModelConsistency(
#         os.path.join(input_dir, "all_images"),
#         os.path.join(input_dir, "train"),
#         True,
#         os.path.join(input_dir, "train"),
#         label_mapping,
#         400,
#     )

#     yolo.select_images()
#     # Labeling our train files
#     run_labelme(
#         os.path.join(input_dir, "train", "images"),
#         os.path.join(input_dir, "train", "annotations", "labelme"),
#     )

#     # Labeling our validation files
#     run_labelme(
#         os.path.join(input_dir, "validation", "images"),
#         os.path.join(input_dir, "validation", "annotations", "labelme"),
#     )
#     train_dir = os.path.join(input_dir, "train")
#     validation_dir = os.path.join(input_dir, "validation")
#     convert = converter.DatasetConverter(
#         os.path.join(train_dir, "annotations", "labelme"),
#         label_mapping,
#         os.path.join(train_dir, "labels"),
#     )
#     convert.process_labelme_annotations()
#     convert = converter.DatasetConverter(
#         os.path.join(validation_dir, "annotations", "labelme"),
#         label_mapping,
#         os.path.join(validation_dir, "labels"),
#     )
#     convert.process_labelme_annotations()

#     yolo.train_model()

#     for i in range(1, cycle_number):
#         yolo = cald_train.ModelConsistency(
#             os.path.join(input_dir, "all_images"),
#             os.path.join(input_dir, "train"),
#             False,
#             os.path.join(input_dir, "active_learning", "sampeled_images"),
#             label_mapping,
#             100,
#         )
#         yolo.train_model()
#         run_labelme(
#             os.path.join(input_dir, "active_learning", "sampeled_images", f"{i}"),
#             os.path.join(
#                 input_dir, "active_learning", "sampeled_annotations", "labelme", f"{i}"
#             ),
#         )
#         converter.DatasetConverter(
#             os.path.join(
#                 input_dir, "active_learning", "sampeled_annotations", "labelme", f"{i}"
#             ),
#             label_mapping,
#             os.path.join(input_dir, "active_learning", "labels"),
#         )
#         files = os.listdir(
#             os.path.join(input_dir, "active_learning", "sampeled_images")
#         )
#         random_validation = random.sample(files, 20)
#         for image_file in random_validation:
#             image_path = os.path.join(
#                 input_dir, "active_learning", "sampeled_images", image_file
#             )
#             label_path = os.path.join(
#                 input_dir,
#                 "active_learning",
#                 "labels",
#                 f"{os.path.splitext(image_file)[0]}.txt",
#             )
#             shutil.move(image_path, os.path.join(input_dir, "validation", "images"))
#             shutil.move(label_path, os.path.join(input_dir, "validation", "labels"))
#         move_files(
#             os.path.join(input_dir, "active_learning", "sampeled_images"),
#             os.path.join(input_dir, "train", "images"),
#             80,
#         )
#         move_files(
#             os.path.join(input_dir, "active_learning", "labels"),
#             os.path.join(input_dir, "train", "labels"),
#             80,
#         )
#         labelme_annotations = os.path.join(
#             input_dir, "active_learning", "sampeled_annotaions", "labelme"
#         )
#         for file in os.listdir(labelme_annotations):
#             source_path = os.path.join(labelme_annotations, file)
#             os.remove(source_path)


# if __name__ == "__main__":
#     main()
import os
import subprocess

from yaml import YAMLError
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

    def resorting(self):
        pass

    def saving_model(self):
        pass

    def detect(self):
        pass


def main():
    print("Please enter the classes you want to detect:\n")
    classes = input().split()
    app = MainApp(classes)
    app.mapping_classes()
    app.create_yaml()
    app.app_run()
