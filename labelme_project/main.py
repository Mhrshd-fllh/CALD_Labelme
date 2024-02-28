import os
import subprocess
from dataset import converter
import random
import shutil


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
        destination_path = os.paht.join(destination_dir, filename)
        shutil.move(source_path, destination_path)

    for filename in selected_files:
        source_path = os.path.join(source_dir, filename)
        os.remove(source_path)

def main():
    #Saving cycle number for folder names
    global cycle_number
    
    #Saving main directories
    input_dir = os.path.join(os.getcwd(), 'labelme_project', 'dataset', 'input_files')
    output_dir = os.path.join(os.getcwd(), 'labelme_project', 'dataset', 'output_files')

    # Getting Info from user 
    print("Please move your dataset to 'labelme_project/dataset/input_files/all_images' to start training active learning object detection.")
    inp = input("Please enter the classes that you want to detect\n")
    classes = [cl for cl in inp.split()]

    #Moving Files to train and validation path by random choice
    move_files(os.path.join(input_dir, 'all_images'), os.path.join(input_dir, 'train', 'images'), 200)
    move_files(os.path.join(input_dir, 'all_images'), os.path.join(input_dir, 'validation', 'images'), 800)
    
    #Labeling our train files
    run_labelme(os.path.join(input_dir, 'train', 'images'), os.path.join(input_dir, 'train', 'annotations', 'labelme'))
    
    #Labeling our validation files
    run_labelme(os.path.join(input_dir, 'validation'), os.path.join(input_dir, 'validation', 'annotations', 'labelme'))


    #cald_train()
    cycle_number += 1

    


if __name__ == '__main__':
    main()
