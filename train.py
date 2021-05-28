import time
from nets import *
from dataset import *
from functions import *
import torch.optim as optim
from torch.utils import data

# This script trains an image segmentation model and saves the trained parameter state dict for the simplest
# case of two object categories (background and object).
# The training images must be in a folder "./training_data" and the validation images in "./val_data".
# The folder structure is assumed to be "training_data/images" for training images and "training_data/masks"
# for the corresponding targets.
# This code was tested on the Penn-Fudan pedestrian dataset https://www.seas.upenn.edu/~jshi/ped_html/.
# Even though the dataset is small, the results are reasonable.

output_size = 256
# If CUDA is available:
print('CUDA: '+str(torch.cuda.is_available()))
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

PATH_train = Path.cwd() / 'training_data'
PATH_val = Path.cwd() / 'val_data'
train_data = SegDataset(PATH_train)
val_data = SegDataset(PATH_val)

batch_size = 16
train_data_loader = data.DataLoader(train_data, batch_size=batch_size)
val_data_loader = data.DataLoader(val_data, batch_size=batch_size)

model = MyFCN()

learning_rate = 1e-4
optimizer = optim.Adam(model.parameters(), lr = learning_rate)
loss_fn = torch.nn.MSELoss()

N_epochs = 50

time_start = time.process_time()
train(model, optimizer, loss_fn, train_data_loader, val_data_loader, epochs=N_epochs)

torch.save(model.state_dict(), f"fcn_alexnet_{N_epochs}_epochs")
time_elapsed = (time.process_time()-time_start)/3600
print(time_elapsed,' hours')














