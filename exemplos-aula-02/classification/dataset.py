from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Callable

from PIL import Image
from torch.utils.data import Dataset

from torch.utils.data import DataLoader

import torch
import torchvision.transforms as transforms 
from torchvision import datasets

def get_mnist():

    training_data = datasets.FashionMNIST(
        root="data",
        train=True,
        download=True,
        transform=transforms.Compose([transforms.ToTensor()])
        )
       
    test_data = datasets.FashionMNIST(
            root="data",
            train=False,
            download=True,
            transform=transforms.Compose([transforms.ToTensor()])
        )

    return training_data, test_data



class ImageDataset(Dataset):

    def __init__(self,
        root_dir,
        split,
        transform = None,
        target_transform = None,
    ):
        
        self.split = split.lower()
        self.transform = transform
        self.target_transform = target_transform

        self.root_dir = Path(root_dir).expanduser().resolve()

        if self.split not in {"train", "val", "test"}:
            raise ValueError(
                f"Split inválido: {split!r}. Use 'train', 'val' ou 'test'."
            )

        csv_path = self.root_dir / f"{self.split}.csv"

        if not csv_path.is_file():
            raise FileNotFoundError(f"CSV não encontrado: {csv_path}")

        with csv_path.open("r", encoding="utf-8-sig", newline="") as csv_file:
            reader = csv.DictReader(csv_file)
            required_columns = {"image_path", "class_name", "label"}
            missing_columns = required_columns.difference(reader.fieldnames or [])
            if missing_columns:
                missing = ", ".join(sorted(missing_columns))
                raise ValueError(f"Colunas ausentes em {csv_path.name}: {missing}")

            self.samples = list(reader)

        if not self.samples:
            raise ValueError(f"O arquivo {csv_path} não contém amostras.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):

        sample = self.samples[index]
        image_path = self.root_dir / sample["image_path"]

        if not image_path.is_file():
            raise FileNotFoundError(f"Imagem não encontrada: {image_path}")

        with Image.open(image_path) as pil_image:
            image = pil_image.convert("RGB").copy()

        label = int(sample["label"])

        if self.transform is not None:
            image = self.transform(image)

        if self.target_transform is not None:
            label = self.target_transform(label)

        return image, label


if __name__ == "__main__":
    

    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ]
    )

    train_dataset = ImageDataset(
        root_dir="./classification/dataset",
        split="train",
        transform=transform,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=8,
        shuffle=True,
        num_workers=0,
    )

    images, labels = next(iter(train_loader))
    print(f"Amostras: {len(train_dataset)}")
    print(f"Batch de imagens: {images.shape}")
    print(f"Batch de labels: {labels.shape}")
