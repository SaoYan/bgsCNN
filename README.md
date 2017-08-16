# Background Subtraction Using Deep Learning Method 
You can find the details about my model in the following two reports:  
1. [Background Subtraction Using Deep Learning--Part I](https://saoyan.github.io/DL-background-subtraction-1/)
2. [Background Subtraction Using Deep Learning--Part II](https://saoyan.github.io/DL-background-subtraction-2/)

## Contents of this repository
* generate_bg.py  
  generating background images; very time consuming to run  
  You can get the result from [here](https://drive.google.com/open?id=0BxTycO36H3VAZ0hkenJKcVNCMlk). Extract this and you will get a directory containing the original dataset with the generated background images. You can directly use it and run prepare_data.py.
* prepare_data.py  
  constructing TFrecords files for preparation of training the model
* bgsCNN_v*.py  
  training the model  
  v1 ~ v3 respectively correspond to Model I ~ III mentioned in the second report; experiments on v4 have not finished yet

## How to run

### 1. Dependences
* [Tensorflow](https://github.com/tensorflow/tensorflow)
* [OpenCV](https://github.com/opencv/opencv) ***compiled with Python support*** (you can refer to [this repository](https://github.com/SaoYan/OpenCV_SimpleDemos) for compiling OpenCV)
* [bgslibrary](https://github.com/andrewssobral/bgslibrary) (needed only if you want to run generate_bg.py yourself)
* Downloaded Checkpoint file of ResNet_V2_50 from [Tensorflow Model Zoo](https://github.com/tensorflow/models/tree/master/slim), and put resnet_v2_50.ckpt at the same directory as Python script files.

### 2. Run the code
Suppose we use bgsCNN_v2
* If you want to run both generate_bg.py and prepare_data.py (trust me, you don't want to run generate_bg.py yourself!):
```
python train.py \
  --generate_bg True \
  --prepare_data True  \
  --dataset_dir dataset \
  --log_dir logs \
  --model_version 2 \ ten
  --train_batch_size 40 \
  --test_batch_size 200 \
  --max_iteration 10000 \
```
* If you've downloaded the dataset I provided and don't need to run generate_bg.py (suppose the downloaded data is stored in directory "dataset"):
```
python train.py \
  --prepare_data True  \
  --dataset_dir dataset \
  --log_dir logs \
  --model_version 2 \
  --train_batch_size 40 \
  --test_batch_size 200 \
  --max_iteration 10000 \
```
* If you've already had the TFrecords files and don't want to tun prepare_data.py (suppose the two TFrecords files are train.tfrecords & test.tfrecords):
```
python train.py \
  --prepare_data False  \
  --train_file train.tfrecords \
  --test_file test.tfrecords \
  --log_dir logs \
  --model_version 2 \
  --train_batch_size 40 \
  --test_batch_size 200 \
  --max_iteration 10000 \
```
