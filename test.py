import torch
import matplotlib.pyplot as plt
from torch.nn.functional import softmax
from model import UniversalClassifier
from utils import (
    get_test_loader,
    get_transforms,
    load_checkpoint
)

def test_model(test_loader, model):
    data, labels = next(iter(test_loader))
    
    model.eval()
    predict = model(data)
    predict = torch.argmax(softmax(predict, dim=1), dim=1)
    predict, labels = predict.detach().cpu().numpy(), labels.detach().cpu().numpy()

    # labels = [idx_to_class[l] for l in labels]
    # predict = [idx_to_class[p] for p in predict]

    data = data.permute(0, 2, 3, 1).numpy()

    ncolumn, nrow = int(len(data)/16), 16
    _, ax = plt.subplots(ncolumn, nrow, figsize=(20, 20))
    
    i = 0
    for row in range(ncolumn):
        ax_row = ax[row]
        for column in range(nrow):
            ax_column = ax_row[column]
            ax_column.imshow(data[i], cmap='gray');
            ax_column.set_xticklabels([])
            ax_column.set_yticklabels([])
            col = 'blue' if labels[i] == predict[i] else 'red'
            ax_column.set_title(str(predict[i]), color=col)
            i += 1
    plt.show()
    

def main():
    model = UniversalClassifier()
    load_checkpoint(torch.load("checkpoints/proto_3.pth.tar"), model)

    loader = get_test_loader("./data/test", 256, get_transforms([64, 64]), 1, True, True)

    test_model(loader, model)

if __name__ == "__main__":
    main()