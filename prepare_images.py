from PIL import Image
import glob
import random


# Training data can be downloaded from https://www.seas.upenn.edu/~jshi/ped_html/

# load images from PennFudanPed/PNGImages/, reshuffle, resize and save them:
images = sorted(list(glob.glob('PennFudanPed/PNGImages/*.png')))
random.seed(5)
indices = list(range(len(images)))
random.shuffle(indices)
for n,i in enumerate(indices):
    path=images[i]
    img = Image.open(path)
    img = img.resize((512,512))
    img.save('training_data/images/img{:03d}_512.png'.format(n))


# load masks from PennFudanPed/PedMasks/, reshuffle, resize and save them:
images = sorted(list(glob.glob('PennFudanPed/PedMasks/*.png')))
indices = list(range(len(images)))
random.shuffle(indices)
for n,i in enumerate(indices):
    path=images[i]
    img = Image.open(path)
    img = img.resize((512,512))
    img.save('training_data/masks/mask{:03d}_512.png'.format(n))

