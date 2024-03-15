import yaml
import os

def create_yaml_file(label_mapping):
    
    path_ = os.path.join(os.getcwd(), 'labelme_project', 'dataset')
    train = 'train/images'
    val = 'validation/images'

    with open('labelme_project/dataset.yaml', 'w') as dataset_file:
        dataset_file.write(f'path: {path_}\n')
        dataset_file.write(f'train: \n')
        dataset_file.write(f'  - {train}\n')
        dataset_file.write(f'val: \n')
        dataset_file.write(f'  - {val}\n')

        dataset_file.write(f'names: \n')
        for i , label in label_mapping.items():
            dataset_file.write(f'  {i}: {label}\n')

    dataset_file.close()

