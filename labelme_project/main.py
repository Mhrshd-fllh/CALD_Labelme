import os
import subprocess
from dataset import converter


#Global Variables 
cycle_number = 0


def run_labelme(input_dir, output_dir):
    # Replace 'labelme' with the actual path to the labelme executable
    labelme_executable = 'labelme'

    # Example command to run LabelMe
    command = [labelme_executable, input_dir, '-O', output_dir]

    # Run LabelMe using subprocess
    subprocess.run(command)

# Example usage
input_directory = 'path/to/images'
output_directory = 'path/to/annotations'

run_labelme(input_directory, output_directory)


def main(args):
    run_labelme()
    #cald_train()
    cycle_number += 1
    


if __name__ == '__main__':
    main(args)