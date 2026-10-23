import tqdm
import torch

from datetime import datetime

from model import NeuralNetwork, get_efficientNet
from dataset import get_mnist, ImageDataset
from metrics import validate

import torchvision.transforms as transforms 

import warnings

warnings.filterwarnings("ignore")


transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ]
    )


def logging(message):
    print(f'{datetime.now().strftime("%Y-%m-%d %H:%M:%S")} {message}')


def train_step(model, loader, optimizer, criterion, device='cpu'):
    
    model = model.to(device)
    model.train()

    predictions = list()
    labels      = list()
    
    for (sample, target) in tqdm.tqdm(loader):
        sample = sample.to(device)
        target = target.to(device)

        optimizer.zero_grad()

        output = model(sample)

        loss = criterion(output, target)
        
        loss.backward()
        optimizer.step()

        _, predicted = torch.max(output, 1)

        predictions = predictions + predicted.cpu().numpy().tolist()
        labels      = labels + target.cpu().numpy().tolist()

    return loss.item(), predictions, labels


def validation_step(model, loader, criterion, device='cpu'):

    model.to(device)
    model.eval()

    predictions = list()
    labels      = list()

    with torch.no_grad():
        for (sample, target) in tqdm.tqdm(loader):
            
            sample = sample.to(device)
            target = target.to(device)
            
            output = model(sample)
            
            loss = criterion(output, target)

            _, predicted = torch.max(output, 1)
            
            predictions = predictions + predicted.cpu().numpy().tolist()
            labels      = labels + target.cpu().numpy().tolist()

    return loss.item(), predictions, labels



if __name__ == "__main__":

    #model  = NeuralNetwork(num_out=10)
    model = get_efficientNet(2)
    opt    = torch.optim.SGD(params=model.parameters(), lr=0.001)
    loss_f = torch.nn.CrossEntropyLoss(reduction='mean')

    #training_data, test_data = get_mnist()

    training_data = ImageDataset(root_dir="./classification/dataset", split="train", transform=transform)
    test_data = ImageDataset(root_dir="./classification/dataset", split="test", transform=transform)

    train_loader = torch.utils.data.DataLoader(training_data, batch_size=8, shuffle=True, 
                                                    num_workers=4, pin_memory=True)
    
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=8, shuffle=True, 
                                                        num_workers=4, pin_memory=True)

    valid_step = 2
    best_acc = 0
    best_epo = 0

    for epoch in range(0, 10):
        logging(f"(Train step - Epoch {epoch})")

        t_loss, t_preds, t_targets = train_step(model, train_loader, opt, loss_f)

        if epoch % valid_step == 0:
            v_loss, v_preds, v_targets = validation_step(model, test_loader, loss_f)
            train_scores, _ = validate(t_loss, t_preds, t_targets)
            valid_scores, _ = validate(v_loss, v_preds, v_targets)

        print(train_scores)
        print(valid_scores)

        if best_acc < valid_scores["accuracy"]:
            logging(f"Best model found at epoch {epoch}")
            best_acc = valid_scores["accuracy"]
            best_epo = epoch
            torch.save(model.state_dict(), "model.pt")

    logging(f"Best ACC: {best_acc} in epoch {best_epo}")