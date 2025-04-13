import cv2
import torch
from torch.utils.data import Dataset


class PlantDataset(Dataset):
    def __init__(self, df, transforms=None):
        self.df = df
        self.transforms = transforms

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = cv2.imread(row.image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        label = row[["healthy", "multiple_diseases", "rust", "scab"]].values.astype(float)

        if self.transforms:
            image = self.transforms(image=image)["image"]

        return image, torch.tensor(label, dtype=torch.float32)


class TestDataset(Dataset):
    def __init__(self, df, transforms=None):
        self.df = df
        self.transforms = transforms

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = cv2.imread(row.image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transforms:
            image = self.transforms(image=image)['image']

        return image, row.image_id
