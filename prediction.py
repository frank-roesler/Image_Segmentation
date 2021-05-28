from PIL import Image
from functions import *
from pathlib import Path
from nets import *


# Make 10 predictions with saved net:
output_size=512
model = MyFCN()
state_dict = torch.load('fcn_alexnet_50_epochs', map_location=torch.device('cpu'))
model.load_state_dict(state_dict)

# Assumes 10 test images in directory "./test_data/img0-img9"
for img_number in range(10):
    path = Path.cwd() / 'test_data'
    img = Image.open(path / f"img{img_number}.png")
    prediction = predict(model, img)
    overlay = Image.blend(img.convert('RGBA'), prediction.convert('RGBA'), alpha=0.6)
    overlay.show()
