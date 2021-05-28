from torch.utils.data import Dataset
from pathlib import Path
from torchvision import transforms
from PIL import Image


class SegDataset(Dataset):
    """Dataset class for segmentation with 1 object class"""
    def __init__(self, path):
        self.images = sorted(list(Path(path).glob('images/*.png')))
        self.masks = sorted(list(Path(path).glob('masks/*.png')))
        self.length = len(self.images)
        self.transforms = transforms.ToTensor()

    def __getitem__(self, idx):
        img_name, mask_name = (self.images[idx], self.masks[idx])
        img  = Image.open(img_name)
        mask = Image.open(mask_name)
        img_tensor  = self.transforms(img)
        mask_tensor = self.transforms(mask)
        mask_tensor[mask_tensor>0]=1 # This line is only to ensure two classes (0=background, 1=object)
        return (img_tensor, mask_tensor)

    def __len__(self):
        return self.length
