# Image Segmentation with Fully Convolutional Networks

Image segmentation (i.e. the task of locating objects in images) is vital in a whole range of applications, ranging from self-driving cars to medical diagnostics. Computer algorithms known as *fully convolutional neural networks* have been shown to perform exceptionally well in this task. Networks are typicelly trained on large datasets of images and corresponding *segmentation masks*, which prescribe the location of the relevant objects:

![a photo of a cat](https://github.com/frank-roesler/Image_Segmentation/blob/main/cat_img15.png)
![a segmentation mask for a cat photo](https://github.com/frank-roesler/Image_Segmentation/blob/main/cat_mask15.png)

Understanding the basics of these models is key in order to build and evaluate well-performing them. This collection of codes includes a PyTorch implementation of some neural network models for image segmentation (see e.g. https://arxiv.org/abs/1411.4038), which can be used to detect the locations of objects in images. The code in this package has been written with one goal in mind: keeping it simple. The model is written only for 1 type of object category to be detected (in addition to background), and all functions and classes are explicit. The goal is not to provide a powerful state-of-the-art model, but to make the basics clear and understandable.

This code was tested on a subset of the COCO (*Common objects in context*) dataset: https://cocodataset.org. Even simple autoencoder models turn out to learn nontrivial information. An executable Colab Notebook, in which trained mocels can be tested hands-on is available here: [Colab Notebook.](https://colab.research.google.com/drive/12E-xU8nwK90xTCrgef3fBkxuz4lU6E3N?usp=sharing)

### Contents:
* `utils.py`: contains classes and functions needed during training and inference.
* `nets.py`: contains implementations of segmentation models.
* `train.py`: main training script.
* `prediction.py`: loads test images, applies a trained model and plots the results.

### Dependencies:
* PyTorch, TorchVision,
* PIL,
* Matplotlib.

### Results:
The following images show the results of two different trained models: (1) a simple convolutional autoencoder and (2) a FCN-ResNet model with transfer learning. Clearly, the simple model has learned nontrivial information, but exhibits [texture bias](https://arxiv.org/pdf/1811.12231.pdf). In Contrast, the more sophisticated FCN-ResNet architechture produces more specific, high quality segmentations.

**Autoencoder:**  
![segmented photo of a cat with autoencoder model](https://github.com/frank-roesler/Image_Segmentation/blob/main/cat_aue9.png)
![segmented photo of a cat with autoencoder model](https://github.com/frank-roesler/Image_Segmentation/blob/main/cat_aue12.png)

**FCN-ResNet:**  
![segmented photo of a cat with resnet model](https://github.com/frank-roesler/Image_Segmentation/blob/main/cat_resnet9.png)
![segmented photo of a cat with resnet model](https://github.com/frank-roesler/Image_Segmentation/blob/main/cat_resnet12.png)

Any comments or queries are welcome at https://frank-roesler.github.io/contact/
