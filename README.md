Consistency Based Active Learning for Object Detection combining with label me. 
In this project we changed the https://github.com/we1pingyu/CALD code for YOLOv8 and connected it with label me for custom datasets that we could train model based on our custom classes and images.
This is how this project works. It gets dataset in 'dataset/input_files/all_images' directory and then randomly select images from this directory and move it to train and validation directories. 
There you should label this images by labelme and then our training starts from here and by the inconsistency metrics of yolo we choose another images and again labelling them and then training and this cycle goes on.
After each cycle you want program will be closed and you have your last_weights and then you could detect objects that you want in another dataset by that weights.
Please download labelme from https://github.com/labelmeai/labelme and move it to 'labelme_project/labelme' directory to connect labelme and cald.
