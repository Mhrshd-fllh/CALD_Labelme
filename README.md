# CALD (Consistency-Based Active Learning for Object Detection) Labelme
## Description
Consistency-Based Active Learning for Object Detection combined with LabelMe. In this project, we modified the CALD code available at https://github.com/we1pingyu/CALD for YOLOv8 and integrated it with LabelMe for custom datasets.

- [installation](#installation)
- [Usage](#usage)
- [Contact](#Contact)

## Installation
First Create an environment:
```cmd
conda create -n "labelme" python=3.9
```
Then run command for installation of all packages:
```cmd
pip install -r requirements.txt
```

## Usage
First move your dataset to 'CALD_Labelme/dataset/train/images'.
Then run below command:
```cmd
python main.py
```
First press the Selection button and after that wait for dataset to shown up in the file list in the right. After labeling the files, you should press the train button and wait for training end and then you do it like a cycle and after all you have your results on runs folder on this directory that shows you the best and last epoch of each cycle.

## Contact 
- [Email] (fallah_mehrshad82@comp.iust.ac.ir)
- [[Telegram] (@Mehrshad_Fallah)](https://t.me/Mehrshad_Fallah)
