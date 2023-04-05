import torch
from torchvision import datasets, transforms
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

transform = transforms.Compose([
    transforms.Resize([64,64]),
    transforms.ToTensor()
])

if __name__ == "__main__":
    dataset = datasets.ImageFolder("./datasets/", transform=transform)
    loader = torch.utils.data.DataLoader(dataset, batch_size=len(dataset), num_workers=1)

    data = next(iter(loader))

    print(data[0].mean())
    print(data[0].std())