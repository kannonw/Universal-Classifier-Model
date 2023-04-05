import torch
from tqdm import tqdm
from model import UniversalClassifier
from torch import nn, optim
from PIL import ImageFile
from utils import (
    save_checkpoint,
    load_checkpoint,
    get_loaders,
    get_transforms,
    accuracy
)

LEARNING_RATE = 1e-5
DEVICE = "cuda" if torch.cuda.is_available else "cpu"
BATCH_SIZE = 64
EPOCHS = 8
NUM_WORKS = 1
PIN_MODEL = True
SHUFFLE = True
LOAD_MODEL = True
IMAGE_SIZE = [64, 64]
TRAIN_IMG_DIR = "./data/train"
VAL_IMG_DIR = "./data/val"
LOW_LOSS = 1.

model = UniversalClassifier().to(DEVICE)
optimizer = optim.Adam(model.parameters(), LEARNING_RATE)
loss_fn = nn.CrossEntropyLoss()
ImageFile.LOAD_TRUNCATED_IMAGES = True

train_loader, val_loader = get_loaders(
    TRAIN_IMG_DIR,
    VAL_IMG_DIR,
    BATCH_SIZE,
    get_transforms(IMAGE_SIZE),
    NUM_WORKS,
    PIN_MODEL,
    SHUFFLE
)


def train_model(loader, mode):
    loop = tqdm(loader)
    running_loss, running_acc = 0, 0

    for (data, targets) in loop:
        data = data.to(DEVICE)
        targets = targets.to(DEVICE)

        optimizer.zero_grad()

        predictions = model(data)
        loss = loss_fn(predictions, targets)
        acc = accuracy(predictions, targets)
        # acc = utils.check_accuracy(predictions, targets)

        running_loss += loss.item()
        running_acc += acc.item()

        loss.backward()
        optimizer.step()

        loop.set_postfix(loss=running_loss/len(loader), acc=running_acc/len(loader), low_loss=LOW_LOSS, mode=mode) # Train

    return running_loss/len(loader)
    



def main():
    global LOW_LOSS

    if LOAD_MODEL:
        load_checkpoint(torch.load("checkpoints/proto_2.pth.tar"), model)

    for _ in range(EPOCHS):
        model.train()
        loss = train_model(train_loader, "train") # model.trainning()

        model.eval()
        train_model(val_loader, "eval") # model.trainning()

        if loss <= LOW_LOSS:
            checkpoint = {
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "low_loss": LOW_LOSS
            }
            save_checkpoint(checkpoint, "checkpoints/proto_3.pth.tar")
            LOW_LOSS = loss

if __name__ == "__main__":
    main()