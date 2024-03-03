import os
import json

class DatasetConverter:
    def __init__(self, input_dir,label_mapping, output_dir):
        self.input_dir = input_dir
        self.label_mapping = label_mapping
        self.output_dir = output_dir
        self.train_path = os.path.join(self.input_dir, 'train')
        self.validation_path = os.path.join(self.input_dir, 'validation')  

    def process_labelme_annotations(self):
        # Process each JSON file in the input_directory
        for filename in os.listdir(self.input_dir):
            if filename.endswith(".json"):
                self.labelme_json_path = os.path.join(self.input_dir, filename)
                self.yolo_txt_path = os.path.join(self.output_dir, f"{os.path.splitext(filename)[0]}.txt")
                self.labelme_to_yolo()

    def labelme_to_yolo(self):
        with open(self.labelme_json_path, 'r') as json_file:
            data = json.load(json_file)
        image_height = data['imageHeight']
        image_width = data['imageWidth']

        with open(self.yolo_txt_path, 'w') as yolo_file:

            for shape in data['shapes']:
                label = shape['label']
                points = shape['points']
                label = {i for i in self.label_mapping if self.label_mapping[i] == label}
                x, y, w, h = self.get_yolo_coordinates(points, image_height, image_width)
                yolo_file.write(f'{label} {x} {y} {w} {h}\n')
                yolo_file.write(f'{image_height} ')
                yolo_file.write(f'{image_width}')

    def get_yolo_coordinates(self, points, image_height, image_width):
        x = (points[0][0] + points[1][0]) / (2 * image_width)
        y = (points[0][1] + points[1][1]) / (2 * image_height)
        w = abs(points[1][0] - points[0][0]) / image_width
        h = abs(points[1][1] - points[0][1]) / image_height

        return x, y, w, h
