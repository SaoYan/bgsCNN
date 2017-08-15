# VehicleCounting_DL
In this project I try to perform background subtraction using deep learning method.  
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

### 2. Run the code
You can choose one from bgsCNN_v*.py and run it directly.  
***ATTENTION***  
If you've downloaded the data I provided and don't need to run generate_bg.py, comment out the two corresponding lines.
```
# from generate_bg import generate_bg
```
AND
```
# generate_bg()
```
