# Image Segmentation with FCNs

This collection of codes includes a PyTorch implementation of the fully convolutional (FCN) version of Alexnet (see https://arxiv.org/abs/1411.4038), which can be used to detect objects in images. The code in this package has been written with one goal: keeping it simple. The model is written only for 1 type of object to be detected (in addition to background), and all functions and classes are explicit; nothing is pre-trained. The goal is not to provide a powerful state-of-the-art model, but to make the basics understandable.

This code was tested with the Penn-Fudan pedestrian dataset https://www.seas.upenn.edu/~jshi/ped_html/. Even though the dataset is small, the results are reasonable.

### Contents:
* `prepare_images`: Resizes all images to (512,512), shuffles them and saves them.
* `dataset.py`: dataset class that loads training images and segmentation masks. Assumes folder structure `path/images/` and `path/masks`, respectively.
* `functions.py`: contrins functions to train and predict from a model.
* `nets.py`: contains implementation of FCN-Alexnet.
* `train.py`: main training loop. Assumes images to be split into a training set `./train_data/` and a validation set `./val_data/`.
* `prediction.py`: loads test images from a folder `./test_data/` and plots the model's prediction.

Any comments or queries are welcome at https://frank-roesler.github.io/contact/
