import tqdm
import torch
import torch.nn as nn
import torchvision
from datetime import datetime
from sklearn import metrics

import warnings

warnings.filterwarnings("ignore")


def logging(message):
    print(f'{datetime.now().strftime("%Y-%m-%d %H:%M:%S")} {message}')


class NeuralNetwork(torch.nn.Module):
    def __init__(self, num_out=1):
        super().__init__()
        self.num_out = num_out
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3),
            nn.ReLU()
        )
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 22 * 22, num_out)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x


def get_mnist():

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

    return training_data, test_data


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


def get_scores(targets, predictions):
    return {
        "accuracy": metrics.accuracy_score(targets, predictions),
        "balanced_accuracy": metrics.balanced_accuracy_score(targets, predictions),
        "precision": metrics.precision_score(targets, predictions, average="weighted"),
        "recall": metrics.recall_score(targets, predictions, average="weighted"),
        "f1_score": metrics.f1_score(targets, predictions, average="weighted")
    }


def accuracy_per_class(targets, predictions, labels=None):
    cmatrix = metrics.confusion_matrix(targets, predictions, labels=labels)
    cmatrix = cmatrix / (cmatrix.astype(float).sum(axis=1) + 1E-8)
    accuracies = cmatrix.diagonal()
    return accuracies


def validate(loss, preds, targets):   
    scores = get_scores(targets, preds)
    classes_scores = accuracy_per_class(targets, preds,
                                                labels=list(range(10)))
    return scores, classes_scores


if __name__ == "__main__":

    model  = NeuralNetwork(num_out=10)
    opt    = torch.optim.SGD(params=model.parameters(), lr=0.001)
    loss_f = torch.nn.CrossEntropyLoss(reduction='mean')

    training_data, test_data = get_mnist()

    train_loader = torch.utils.data.DataLoader(training_data, batch_size=8, shuffle=True, 
                                                    num_workers=4, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=8, shuffle=True, 
                                                        num_workers=4, pin_memory=True)

    valid_step = 2
    best_acc = 0
    best_epo = 0
    for epoch in range(0,10):
        logging(f"(Train step - Epoch {epoch})")

        t_loss, t_preds, t_targets = train_step(model, train_loader, opt, loss_f)

        if epoch % valid_step == 0:
            v_loss, v_preds, v_targets = validation_step(model, test_loader, loss_f)
            train_scores, _ = validate(t_loss, t_preds, t_targets)
            valid_scores, _ = validate(v_loss, v_preds, v_targets)

        print(train_scores)
        print(valid_scores)

        if best_acc < valid_scores["accuracy"]:
            best_acc = valid_scores["accuracy"]
            best_epo = epoch

    logging(f"Best ACC: {best_acc} in epoch {best_epo}")