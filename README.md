# Image Segmentation using fully convolutional neural net

This collection of codes includes an implementation of the fully convolution (FCN) version of Alexnet (see https://arxiv.org/abs/1411.4038), which can be used to detect objects in images. The code in this package has been written with one goal: keeping it simple. The model is written only for 1 type of object to be detected (in addition to background), and all functions and classes are explicit; nothing is pre-trained. The goal is not to provide a powerful state-of-the-art model, but to make the basics understandable.

This code was tested on the Penn-Fudan pedestrian dataset https://www.seas.upenn.edu/~jshi/ped_html/. Even though the dataset is small, the results are reasonable.
