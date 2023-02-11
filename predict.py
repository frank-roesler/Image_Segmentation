from PIL import Image
from utils import predict, load_model
from nets import AutoEncoder
import torch

# This is an example script that loads a pre-trained neural net from 'PATH_TO_CHECKPOINT'
# and applies it to a set of test images.

test_img_path = 'PATH_TO_TEST IMAGE'
model_path    = 'PATH_TO_CHECKPOINT'

device        = torch.device('cuda' if torch.cuda.is_available() else "cpu")

# Make a prediction with saved net:
model = AutoEncoder(nc=8).to(device)
checkpoint = torch.load(model_path, map_location=device)
model, train_img_size = load_model(model, checkpoint, plot_loss_curve=False)

img        = Image.open(test_img_path)
prediction = predict(model, img, trained_img_size=train_img_size, device=device)
overlay    = Image.blend(img.convert('RGBA'), prediction.convert('RGBA'), alpha=0.5)
overlay.show()
