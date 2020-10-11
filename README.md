[//]: # (Image References)

[image1]: ./images/keypoint_example.png "Facial Keypoint Detection"

# Facial Keypoint Detection


## Project Overview

The aim of this project is to build a facial keypoint detection system. Facial keypoints include points around the eyes, nose, and mouth on a face and are used in many applications. These applications include: facial tracking, facial pose recognition, facial filters, and emotion recognition. The completed code predicts the locations of facial keypoints on each face; examples of these keypoints are displayed below.

Originally, inspired to the project required as part of the program: Udacity Computer Vision Nanodegree, it is here presented with multiple improvements such as: customize weights initialization - to improve training performances and GPU training support.

![Facial Keypoint Detection][image1]

The project is divided into three main Python notebooks that will perform: data visualization and initial exploration, model train and evaluation and finally model prediction

__1. Load and Visualize Data.ipynb__ : Loading and Visualizing the Facial Keypoint Data

__2. Define the Network Architecture.ipynb__ : Defining and Training a Convolutional Neural Network (CNN) to Predict Facial Keypoints

__3. Facial Keypoint Detection, Complete Pipeline.ipynb__ : Facial Keypoint Detection Using Haar Cascades and your Trained CNN

Additionally, to the notebooks above, there three files, two Python script file (.py) and a jupyter notebook:

__model.py__ :  Define the convolutional neural network architecture and the function forward.

__data_load.py__ :  Data loading and transform pipeline to be applied on a samples.

__2b. Define the Network Architecture-GPU_train.ipynb__ : This notebook uses the structure of the notebook 2. with the implementation of the GPU train to speed up the training time (by a factor x10) and improve the accuracy reached.


## Project Instructions

All of the starting code and resources you'll need to run this project are in this Github repository. Before you can get started coding, you'll have to make sure that you have all the libraries and dependencies required to support this project.

*Note that this project does not require the use of GPU. Nevertheless, the usage of the GPU is recommended to reach decent results. 2b. Define the Network Architecture-GPU_train.ipynb notebook can be run inside Google Colab and levering on the Nvidia K80 offered by Google.*


### Local Environment Instructions

1. Clone the repository, and navigate to the downloaded folder. This may take a minute or two to clone due to the included image data.
```
git clone https://github.com/FrancescoMrn/CNN_Facial_Keypoints
cd CNN_Facial_Keypoints
```

2. Create (and activate) a new environment, named `cnn_keypoints` with Python 3.6. If prompted to proceed with the install `(Proceed [y]/n)` type y. Similarly it is possible to use any environment will all the requirements reported in **requirements.txt**

	- __Linux__ or __Mac__:
	```
	conda create -n cnn_keypoints python=3.6
	source activate cnn_keypoints
	```
	- __Windows__:
	```
	conda create --name cnn_keypoints python=3.6
	activate cnn_keypoints
	```

	At this point your command line should look something like: `(cnn_keypoints) <User>:CNN_Facial_Keypoints <user>$`. The `(cnn_keypoints)` indicates that your environment has been activated, and you can proceed with further package installations.

3. Install PyTorch and torchvision; this should install the latest version of PyTorch.

	- __Linux__ or __Mac__:
	```
	conda install pytorch torchvision -c pytorch
	```
	- __Windows__:
	```
	conda install pytorch-cpu -c pytorch
	pip install torchvision
	```

6. Install a few required pip packages, which are specified in the requirements text file (including OpenCV).
```
pip install -r requirements.txt
```


## Data

The data you will need to train a neural network will be downloaded inside the subdirectory `data` by the running of the following bash code reported inside the notebooks:

```
!mkdir /data
!wget -P /data/ https://s3.amazonaws.com/video.udacity-data.com/topher/2018/May/5aea1b91_train-test-data/train-test-data.zip
!unzip -n /data/train-test-data.zip -d /data
```

In this folder are training and tests set of image/keypoint data, and their respective csv files. This will be further explored in Notebook 1.


## Possible improvements

 - Implementation of Tensorboard to explore train and evaluation loss.
 - Hyperparameters optimization.
 - Live face detection and keypoints regression.
 - Extend project to another dataset: https://www.kaggle.com/c/facial-keypoints-detection/overview
 - Facial Keypoints Detection with Inception model structure


LICENSE: This project is licensed under the terms of the MIT license.
