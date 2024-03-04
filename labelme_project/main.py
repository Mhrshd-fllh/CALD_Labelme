import os
import subprocess
from dataset import converter
import random
import shutil
import yaml_creator
import cald_train
#Global Variables 
cycle_number = 0


def run_labelme(input_dir, output_dir):
    # Replace 'labelme' with the actual path to the labelme executable
    labelme_executable = os.path.join(os.getcwd(), 'labelme_project','labelme', 'labelme', '__main__.py')

    # Example command to run LabelMe
    command = ['python', labelme_executable, input_dir, '-O', output_dir]

    # Run LabelMe using subprocess
    subprocess.run(command)

def move_files(source_dir, destination_dir, num_files):
    all_files = os.listdir(source_dir)

    selected_files = random.sample(all_files, min(num_files, len(all_files)))

    for filename in selected_files:
        source_path = os.path.join(source_dir, filename)
        destination_path = os.path.join(destination_dir, filename)
        shutil.move(source_path, destination_path)

    for filename in selected_files:
        source_path = os.path.join(source_dir, filename)
        os.remove(source_path)

def main():
    #Saving cycle number for folder names
    global cycle_number
    
    #Saving main directories
    input_dir = os.path.join(os.getcwd(), 'labelme_project', 'dataset', 'input_files')

    # Getting Info from user 
    print("Please move your dataset to 'labelme_project/dataset/input_files/all_images' to start training active learning object detection.")
    inp = input("Please enter the classes that you want to detect\n")
    classes = [cl for cl in inp.split()]
    label_mapping = {}
    for i, cls in enumerate(classes):
        label_mapping[i] = cls
    
    cycle_number = int(input(" Enter Cycle Number you want to train yolo:\n"))
    for i in range(cycle_number):
        os.mkdir(os.path.join(input_dir, 'active_learning', 'sample_images', f'{i}'))
        os.mkdir(os.path.join(input_dir, 'active_learning', 'sample_annotations', 'yolo', f'{i}'))
        os.mkdir(os.path.join(input_dir, 'active_learning', 'sample_annotations', 'labelme', f'{i}'))
    yolo = cald_train.ModelConsistency(os.path.join(input_dir, 'all_images'), os.path.join(input_dir, 'train', 'images'),
                                       os.path.join(input_dir, 'train', 'annotations', 'yolo'), True,
                                       os.path.join(input_dir, 'active_learning', 'sample_images', '0'),
                                       label_mapping, 200)


    # Moving Files to train and validation path by random choice
    # move_files(os.path.join(input_dir, 'all_images'), os.path.join(input_dir, 'train', 'images'), 200)
    move_files(os.path.join(input_dir, 'all_images'), os.path.join(input_dir, 'validation', 'images'), 800)
    
    #Labeling our train files
    run_labelme(os.path.join(input_dir, 'train', 'images'), os.path.join(input_dir, 'train', 'annotations', 'labelme'))
    
    #Labeling our validation files
    run_labelme(os.path.join(input_dir, 'validation', 'images'), os.path.join(input_dir, 'validation', 'annotations', 'labelme'))
    train_dir = os.path.join(input_dir, 'train')
    validation_dir = os.path.join(input_dir, 'validation')

    convert = converter.DatasetConverter(os.path.join(train_dir, 'annotations', 'labelme'), os.path.join(train_dir, 'annotations', 'yolo'), label_mapping)
    convert.process_labelme_annotations()
    convert = converter.DatasetConverter(os.path.join(validation_dir, 'annotations', 'labelme'), os.path.join(validation_dir, 'annotations', 'yolo'), label_mapping)
    convert.process_labelme_annotations()
    yaml_creator.create_yaml_file(label_mapping)
    


    for i in range(1, cycle_number):
        yolo = cald_train.ModelConsistency(os.path.join(input_dir, 'all_images'), os.path.join(input_dir, 'train', 'images'),
                                           os.path.join(input_dir, 'train', 'validation', 'yolo'), False,
                                           os.path.join(input_dir, 'active_learning', 'sampeled_images', f'{i}'),
                                           label_mapping, 500)
        run_labelme(os.path.join(input_dir, 'active_learning', 'sampeled_images', f'{i}'), os.path.join(input_dir, 'active_learning', 'sampeled_annotations', 'labelme', f'{i}'))
        converter.DatasetConverter(os.path.join(input_dir, 'active_learning', 'sampeled_annotations', 'labelme', f'{i}'), label_mapping, os.path.join(input_dir, 'active_learning', 'sampeled_annotations', 'yolo')) 
    


if __name__ == '__main__':
    main()
