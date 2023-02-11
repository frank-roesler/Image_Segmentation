from torch.utils.data import Dataset
from pathlib import Path
from torchvision import transforms
import torchvision.transforms.functional as TF
from PIL import Image
import matplotlib.pyplot as plt
import torch
from numpy import inf
from numpy.random import rand, randn


class SegDataset(Dataset):
    """Dataset class for training image segmentation with 2 classes (object + background).
    __getitem__ method returns a pair of image and corresponding segmentation mask with optional data augmentation."""
    def __init__(self, path, img_transforms=None, augment=False):
        self.images = sorted(list(Path(path).glob('images/*.png')))
        self.masks = sorted(list(Path(path).glob('masks/*.png')))
        self.length = len(self.images)
        self.augment = augment
        if img_transforms == None:
            self.transforms = transforms.ToTensor()
        else:
            self.transforms = img_transforms

    def augment_data(self, img, mask):
        """applies random transformations to both images and masks (hard-coded for now...)"""
        q = rand()
        if 0 < q < 0.2:
            img = TF.hflip(img)
            mask = TF.hflip(mask)
        elif 0.2 < q < 0.4:
            angle = 360*rand()
            img = TF.rotate(img, angle)
            mask = TF.rotate(mask, angle)
        elif 0.4 < q < 0.6:
            xshear, yshear = 10*randn(2)
            img = TF.affine(img, angle=0, translate=[0, 0], scale=1, shear=[xshear,yshear])
            mask = TF.affine(mask, angle=0, translate=[0, 0], scale=1, shear=[xshear,yshear])
        elif 0.6 < q < 0.8:
            img = TF.vflip(img)
            mask = TF.vflip(mask)
        return img, mask

    def __getitem__(self, idx):
        img_name, mask_name = (self.images[idx], self.masks[idx])
        img = Image.open(img_name)
        mask = Image.open(mask_name)
        if self.augment:
            img, mask = self.augment_data(img, mask)
        img_tensor = self.transforms(img)
        mask_tensor = self.transforms(mask)
        mask_tensor[mask_tensor > 0] = 1
        mask_tensor = torch.max(mask_tensor, dim=0, keepdim=True)[0] # ensures only 2 classes (0=background, 1=object)
        return (img_tensor, mask_tensor)

    def __len__(self):
        return self.length


def train(model, optimizer, loss_fn, train_loader, val_loader, epochs=100, device=torch.device("cpu"), export_path="checkpoint", checkpoint=None):
    """trains a model for specified number of epochs. Automatically saves a checkpoint every time the validation loss
    reaches a new minimum. Optionally, training can be started from a pre-saved checkpoint."""
    print("training...")
    best_val_loss = inf  # threshold for saving new checkpoint
    train_losses  = []
    val_losses    = []
    current_epoch = 0
    if checkpoint:       # load model from checkpoint
        current_epoch = checkpoint['epoch']+1
        train_losses  = checkpoint['train_losses']
        val_losses    = checkpoint['val_losses']
        best_val_loss = checkpoint['best_val_loss']
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optim_state_dict'])

    fig, ax = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for epoch in range(current_epoch,epochs):
        training_loss = 0.0
        val_loss      = 0.0
        model.train()
        for batch in train_loader:      # compute training loss
            optimizer.zero_grad()
            input, target = batch
            input = input.to(device)
            target = target.to(device)
            output = model(input)
            loss = loss_fn(output, target)
            loss.backward()
            optimizer.step()
            training_loss += loss.data.item() * input.size(0)
        training_loss /= len(train_loader.dataset)

        model.eval()
        for batch in val_loader:        # compute validation loss
            input, target = batch
            input = input.to(device)
            target = target.to(device)
            output = model(input)
            loss = loss_fn(output, target)
            val_loss += loss.data.item() * input.size(0)
        val_loss /= len(val_loader.dataset)
        train_losses.append(training_loss)
        val_losses.append(val_loss)
        if val_loss < best_val_loss:    # save best model so far
            torch.save({"model_state_dict": model.state_dict(),
                        "optim_state_dict": optimizer.state_dict(),
                        "train_img_size":   input.shape[-1],
                        "epoch":            epoch,
                        "train_losses":     train_losses,
                        "val_losses":       val_losses,
                        "best_val_loss":    best_val_loss
                        }, export_path)
            best_val_loss = val_loss

        if epoch % 10 == 0:
            plot_losses(ax, train_losses, val_losses, epoch)


def predict(model,img, trained_img_size=128, device=torch.device("cpu")):
    """Given an input image and a trained model, this function computes and returns the model's predicted segmentation mask.
    Images are first resized to the size the model was trained on, then segmented, then reverted to original size."""
    img_size = (trained_img_size,trained_img_size)
    totensor = transforms.ToTensor()
    resize = transforms.Resize(img_size)
    img_tensor = totensor(img.convert('RGB'))
    orig_shape = img_tensor.shape[-2:]
    revert = transforms.Resize(orig_shape)
    img_tensor = resize(img_tensor)
    img_tensor = img_tensor.unsqueeze(0).to(device)
    model.eval()
    prediction = model(img_tensor).squeeze(0)
    prediction = revert(prediction)
    toimage = transforms.ToPILImage()
    q = 0.5
    prediction[prediction>=q] = 1
    prediction[prediction<q] = 0
    segmentation = toimage(prediction)
    return segmentation


def plot_losses(ax, train_losses, val_losses, epoch, block=False):
    """plots training and validation losses during training, alongside their running means."""
    window = 8
    train_means = [sum([element/(2*window) for element in train_losses[i-window:i+window]]) for i in range(window,len(train_losses)-window)]
    val_means = [sum([element/(2*window) for element in val_losses[i-window:i+window]]) for i in range(window,len(val_losses)-window)]
    ax[0].cla()
    ax[1].cla()
    ax[0].semilogy(range(epoch+1),train_losses, linewidth=0.5)
    ax[1].semilogy(range(epoch+1),val_losses, linewidth=0.5)
    ax[0].semilogy(range(window,epoch-window+1),train_means)
    ax[1].semilogy(range(window,epoch-window+1),val_means)
    ax[0].set_title("Training Loss")
    ax[1].set_title("Validation Loss")
    plt.show(block=block)
    plt.pause(0.001)
    print('Epoch: {}, Training Loss: {:.4f}, Validation Loss: {:.4f}'.format(epoch, train_losses[-1], val_losses[-1]))


def load_model(model, checkpoint, plot_loss_curve=False):
    """loads pre-trained parameters given in 'checkpoint' into an existing model. Optionally plots the loss curve
    logged during the training process. The size of the images which the model was trained on is also loaded and returned."""
    state_dict = checkpoint['model_state_dict']
    model.load_state_dict(state_dict)
    train_img_size = checkpoint['train_img_size']
    epoch = checkpoint['epoch']
    train_losses = checkpoint['train_losses']
    val_losses = checkpoint['val_losses']
    if plot_loss_curve:
        fig, ax = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
        plot_losses(ax, train_losses, val_losses, epoch, block=True)
    return model, train_img_size



