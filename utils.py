import torch
from torchvision import datasets, transforms

def save_checkpoint(state, filename="checkpoints/my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)

def load_checkpoint(checkpoint, model):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])

def get_loaders(train_path, eval_path, batch_size, transform, num_workers=1, pin_memory=True, shuffle=True):
    train_d = datasets.ImageFolder(train_path, transform=transform)
    val_d = datasets.ImageFolder(eval_path, transform=transform)
    # test_d = datasets.ImageFolder(test_path, transform=transform)
    
    train_loader = torch.utils.data.DataLoader(train_d, batch_size=batch_size, num_workers=num_workers,
        pin_memory=pin_memory, shuffle=shuffle)
    
    val_loader = torch.utils.data.DataLoader(val_d, batch_size=batch_size, num_workers=num_workers,
        pin_memory=pin_memory, shuffle=shuffle)
    
    # test_loader = torch.utils.data.DataLoader(test_d, batch_size=batch_size, num_workers=num_workers,
    #     pin_memory=pin_memory, shuffle=shuffle)

    return train_loader, val_loader

def get_test_loader(test_path, batch_size, transform, num_workers=1, pin_memory=True, shuffle=True):
    test_d = datasets.ImageFolder(test_path, transform=transform)
    
    test_loader = torch.utils.data.DataLoader(test_d, batch_size=batch_size, num_workers=num_workers,
        pin_memory=pin_memory, shuffle=shuffle)
    
    return test_loader

def get_transforms(img_size):
    train_tranform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(img_size),
        transforms.Normalize(mean=[0.3544, 0.3261, 0.3850], std=[0.2801, 0.2741, 0.3105]),
        # [0.3720, 0.3410, 0.4056] [0.2808, 0.2760, 0.3118]
    ])

    return train_tranform

def accuracy(outputs, labels):
    predicted = torch.argmax(torch.nn.functional.softmax(outputs, dim=1), dim=1)
    equals = predicted == labels

    acc = torch.mean(equals.float())

    return acc