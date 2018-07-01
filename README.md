# Background Subtraction Using Deep Learning Method

[![DOI](https://zenodo.org/badge/96831496.svg)](https://zenodo.org/badge/latestdoi/96831496)
[![AUR](https://img.shields.io/aur/license/yaourt.svg?style=plastic)](LICENSE)

***
Most Recent Updates  
May 23, 2018: Some pre-trained models are released.  
***

You can find the details about my model in the following reports:  
1. [Background Subtraction Using Deep Learning--Part I](https://saoyan.github.io/posts/2017/07/27)
2. [Background Subtraction Using Deep Learning--Part II](https://saoyan.github.io/posts/2017/08/07)  
3. [Background Subtraction Using Deep Learning--Part III](https://saoyan.github.io/posts/2017/11/18)  

A poster is also available. (The poster is only based on experiment results of v1~v3)  
[JPG version](https://saoyan.github.io/files/Mitacs_Internship_Poster.jpg)  
[PDF version](https://saoyan.github.io/files/Mitacs_Internship_Poster.pdf)

## Pre-trained models  
Unfortunately, pre-trained models of v1 and v4 are missing :(  
[Version 2](https://drive.google.com/open?id=1p35wm7NV4EB6649AGOATan5EJrBakINM)  
[Version 3](https://drive.google.com/open?id=14iLu2VTxoZ71-FCxkr0MeIYwAj4cbNMa)  
[Version 5](https://drive.google.com/open?id=15xnbA86tfHA_y6wrA8qVpC_gQD8nvnwH)

## Contents of this repository
* generate_bg.py  
  generating background images; very time consuming to run  
  You can get the preprocessed dataset from [here](https://drive.google.com/open?id=0BxTycO36H3VARFdRQkR1WHJYM0E).(**If you have problem accessing Google Drive, please use this [alternative link](http://pan.baidu.com/s/1qYmcUC0)**) Extract this and you will get a directory containing the original dataset with the generated background images. You can directly use it and run prepare_data.py.  

* prepare_data.py  
  constructing TFrecords files for preparation of training the model
* bgsCNN_v*.py  
  training the model  
  v1 ~ v3 respectively correspond to Model I ~ III mentioned in the second report; ~~v4, v5 haven't been included in reports yet~~

## How to run

### 1. Dependences
* [Tensorflow](https://github.com/tensorflow/tensorflow)
* [OpenCV](https://github.com/opencv/opencv) ***compiled with Python support*** (you can refer to [this repository](https://github.com/SaoYan/OpenCV_SimpleDemos) for compiling OpenCV)
* [bgslibrary](https://github.com/andrewssobral/bgslibrary) (needed only if you want to run generate_bg.py yourself)
* Downloaded Checkpoint file of ResNet_V2_50 from [Tensorflow Model Zoo](https://github.com/tensorflow/models/tree/master/research/slim), and put resnet_v2_50.ckpt at the same directory as Python script files.
* Downloaded Checkpoint file of vgg_16 from [Tensorflow Model Zoo](https://github.com/tensorflow/models/tree/master/research/slim), and put vgg_16.ckpt at the same directory as Python script files.

### 2. Training
***
**NOTE**  
If you use bgsCNN_v1, v2 or v3, set the image_height & image_width as multiples of 32 plus 1, e.g. 321.  
If you use bgsCNN_v4 or v5, set the image_height & image_width as multiples of 32, e.g. 320.
***
In the following demos, suppose we use bgsCNN_v2.
* If you want to run both generate_bg.py and prepare_data.py (trust me, you don't want to run generate_bg.py yourself!):
```
python train.py \
  --generate_bg True \
  --prepare_data True  \
  --dataset_dir dataset \
  --log_dir logs \
  --model_version 2 \
  --image_height 321 \
  --image_width 321 \
  --train_batch_size 40 \
  --test_batch_size 200 \
  --max_iteration 10000
```
* If you've downloaded the dataset I provided and don't need to run generate_bg.py (suppose the downloaded data is stored in directory "dataset"):
```
python train.py \
  --prepare_data True  \
  --dataset_dir dataset \
  --log_dir logs \
  --model_version 2 \
  --image_height 321 \
  --image_width 321 \
  --train_batch_size 40 \
  --test_batch_size 200 \
  --max_iteration 10000
```
* If you've already had the TFrecords files and don't want to tun prepare_data.py (suppose the two TFrecords files are train.tfrecords & test.tfrecords):
```
python train.py \
  --prepare_data False  \
  --train_file train.tfrecords \
  --test_file test.tfrecords \
  --log_dir logs \
  --model_version 2 \
  --image_height 321 \
  --image_width 321 \
  --train_batch_size 40 \
  --test_batch_size 200 \
  --max_iteration 10000
```

### 3. Test on the test set
When you've finished the training, you can evaluate the model on test to see average test loss. The logs of this test procedure will be in sub-directory ***"model_test"*** under your identified logs directory.
```
python test.py \
  --test_file test.tfrecords \
  --log_dir logs \
  --model_version 2 \
  --image_height 321 \
  --image_width 321 \
  --optimal_step 9600
```

### 4. Test on video
You can also run the model on your own video.
```
python test_on_video.py \
  --log_dir logs \
  --model_version 2 \
  --image_height 321 \
  --image_width 321 \
  --video_file test.mp4
  --optimal_step 9600
```
