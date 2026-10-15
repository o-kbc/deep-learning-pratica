import torch
import torchvision

def mnist():

    training_data = torchvision.datasets.FashionMNIST(
        root="data",
        train=True,
        download=True,
        transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
        )
       
    test_data = torchvision.datasets.FashionMNIST(
            root="data",
            train=False,
            download=True,
            transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
        )


    print(training_data)
    print(test_data)

    tLoader = torch.utils.data.DataLoader(training_data, batch_size=4, shuffle=True)

    print(tLoader)

    for x,y in tLoader:
        print(x)
        print(x.shape)
        print(y)
        print(y.shape)
        break


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, images, labels, transforms=None, label_transforms=None):
        pass

    def __len__(self):
        pass

    def __getitem__(self):
        pass


if __name__ == "__main__":
   mnist()