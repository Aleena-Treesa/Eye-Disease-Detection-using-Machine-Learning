import torchvision.datasets as datasets
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import torch



batch_size = 32
data_dir = 'D:/PROJECT/Dataset/ODIR-5K/ODIR-5K'
dataset = datasets.ImageFolder(data_dir)

# Define your dataset
dataset = ImageFolder(root='D:/PROJECT/Dataset/ODIR-5K/ODIR-5K', transform=transforms.ToTensor())

# Create a DataLoader
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)




# Define a transformation to resize images
transform = transforms.Compose([
    transforms.Resize((200, 200)),
    transforms.ToTensor(),
])



def get_mean_std(loader):
    # Compute the mean and standard deviation of all pixels in the dataset
    num_pixels = 0
    mean = 0.0
    std = 0.0
    for images, _ in loader:
        batch_size, num_channels, height, width = images.shape
        num_pixels += batch_size * height * width
        mean += images.mean(axis=(0, 2, 3)).sum()
        std += images.std(axis=(0, 2, 3)).sum()

    mean /= num_pixels
    std /= num_pixels

    return mean, std



loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
mean, std = get_mean_std(loader)