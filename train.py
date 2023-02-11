import torch.backends.mps
from nets import AutoEncoder, FCN_Resnet
from utils import train, load_model, SegDataset
import torch.optim as optim
from torch.utils import data

# This script trains an image segmentation model and saves the trained parameter state dict for the simplest
# case of two object categories (background and object).
# The folder structure is assumed to be 'TRAINING_DATA/images' for training images and 'TRAINING_DATA/masks'
# for the corresponding targets (analogously for validation data).

device_name = 'mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'
device      = torch.device(device_name)
print('Device: ', device)

AugmentData         = True
LoadPretrainedModel = False
ExportPath          = 'checkpoint'
PreTrainedPath      = 'checkpoint'
checkpoint          = torch.load(PreTrainedPath, map_location=device) if LoadPretrainedModel else None

batch_size    = 32
learning_rate = 1e-3
N_epochs      = 300

PATH_train = 'TRAINING_DATA'
PATH_val   = 'VALIDATION_DATA'
train_data = SegDataset(PATH_train, augment=AugmentData)
val_data   = SegDataset(PATH_val, augment=AugmentData)
train_data_loader = data.DataLoader(train_data, batch_size=batch_size)
val_data_loader   = data.DataLoader(val_data, batch_size=batch_size)

model = AutoEncoder(nc=8).to(device)
# model = FCN_Resnet().to(device)
# for name, param in model.named_parameters():
#     if 'resnet.classifier.4' not in name:
#         param.requires_grad = False
print("Number of model parameters:", sum(p.numel() for p in model.parameters()))

optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
loss_fn   = torch.nn.MSELoss()

train(model, optimizer, loss_fn, train_data_loader, val_data_loader,
      epochs=N_epochs, device=device, export_path=ExportPath, checkpoint=checkpoint)


