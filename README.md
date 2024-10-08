# CALD (Consistency-Based Active Learning for Object Detection) Labelme
![Build Status](https://img.shields.io/badge/build-coverage-brightgreen)
![License](https://img.shields.io/badge/license-MIT-blue)
![Version](https://img.shields.io/badge/version-2.0.0-orange)

## Description
Consistency-Based Active Learning for Object Detection combined with LabelMe. In this project, we modified the CALD code available at [CALD GitHub](https://github.com/we1pingyu/CALD) for YOLOv8 and integrated it with LabelMe for custom datasets. Check [Labelme](https://github.com/labelmeai/labelme) for labelme code.

- [installation](#installation)
- [Usage](#usage)
- [Contact](#Contact)
- [License](#License)

## Installation
First Create an environment:
```sh
conda create -n "labelme" python=3.9
```
Activate the envrionment:
```sh
conda activate labelme
```
Then install all required packages:
```sh
pip install -r requirements.txt
```
If you have GPU you can run this command too:
```sh
python -m pip install torch==1.13.1+cu117 torchvision>=0.13.1+cu117 torchaudio>=0.13.1+cu117 --extra-index-url https://download.pytorch.org/whl/cu117
```
You can also set the ```KMP_DUPLICATE_LIB``` to be True:
```sh
set KMP_DUPLICATE_LIB=TRUE
```

## Usage
1. Run the following command:
```sh
python labelme/main.py
```
2. Press the "Selection" button and wait for the dataset to appear in the file list on the right.
3. After labeling the files, press the "Train" button and wait for the training to finish.
4. Repeat this cycle as needed. The results will be saved in the `runs` folder, showing the best and last epoch of each cycle.

## Contact 
For any inquiries, please reach out via email or telegram:
- Email: (fallah_mehrshad82[at]comp[dot]iust[dot]ac[dot]ir)
- [[Telegram] (@Mehrshad_Fallah)](https://t.me/Mehrshad_Fallah)

## License
This project is licensed under the MIT License. See the [LICENSE](License) file for details.
