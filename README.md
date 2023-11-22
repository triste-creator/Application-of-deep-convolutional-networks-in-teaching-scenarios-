# bookish-waddle
Introduction
This is a system based on CNN deep learning network, which aims to analyze students' classroom concentration in teaching scenes. 
Face detection in images, real-time monitoring and videos in teaching scenes are carried out. 
First, CV2 cascade face detection model is used for face detection, and then the improved VGG model is used for face expression recognition and classification. 
Get the concentration analysis of class students.
Dataset
The dataset of this project is a grayscale image stored in the form of a two-dimensional matrix, and the data and expressions are stored in the csv table cnn_data.csv
链接：https://pan.baidu.com/s/1lMikoRshus9cP8nnh4iYAQ?pwd=8888 
提取码：8888

First, data_processing.py was run to preprocess the data set, and train_vgg.py was run to train and test the model and visualize the training data. The model_apply.py file completes the actual test application.

