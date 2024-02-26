import os
import subprocess
from dataset import converter


#Global Variables 
cycle_number = 0


def run_labelme(input_dir, output_dir):
    # Replace 'labelme' with the actual path to the labelme executable
    labelme_executable = os.path.join(os.getcwd(), 'labelme_project','labelme', 'labelme', 'app.py')

    # Example command to run LabelMe
    command = ['python', labelme_executable, input_dir, '-O', output_dir]

    # Run LabelMe using subprocess
    subprocess.run(command)


def main():
    global cycle_number
    input_dir = os.path.join(os.getcwd(), 'labelme_project', 'dataset', 'input_files')
    output_dir = os.path.join(os.getcwd(), 'labelme_project', 'dataset', 'output_files')
    yolo_input_dir = os.path.join(input_dir, 'labelme_project', 'active_learning', 'yolo')
    yolo_output_dir = os.path.join(output_dir, 'labelme_project', 'active_learning', 'yolo')
    labelme_input_dir = os.path.join(input_dir, 'labelme_project', 'active_learning', 'labelme')
    labelme_output_dir = os.path.join(output_dir, 'labelme_project', 'active_learning', 'labelme')

    #Labeling our train files

    run_labelme(os.path.join(input_dir, 'first_files'), os.path.join(input_dir, 'train'))
    
    #Labeling our validation files

    run_labelme(os.path.join(input_dir, 'first_files'), os.path.join(input_dir, 'validation'))
    
    #cald_train()
    cycle_number += 1

    


if __name__ == '__main__':
    import argparse
    main()
