import torch
from torchvision import datasets, transforms
from tqdm import tqdm
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

def get_mean_std(loader):
    channels_sum, channels_squared_sum, num_batches = 0, 0, 0

    for data, _ in tqdm(loader):
        channels_sum += torch.mean(data, dim=[0,2,3])
        channels_squared_sum += torch.mean(data**2, dim=[0,2,3])
        num_batches += 1

    mean = channels_sum/num_batches
    std = (channels_squared_sum/num_batches - mean**2)**0.5

    return mean, std

transform = transforms.Compose([
    transforms.Resize([64,64]),
    transforms.ToTensor()
])

train_d = datasets.ImageFolder("./datasets", transform=transform)

train_loader = torch.utils.data.DataLoader(train_d, batch_size=32)

mean, std = get_mean_std(train_loader)

print(mean)
print(std)