import albumentations as a
from albumentations.pytorch import ToTensorV2
import pandas as pd
from os.path import join
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from dataset import PlantDataset, TestDataset
from torchmetrics.classification import BinaryAUROC


def get_transforms(phase="train"):
    """
    Data transformation. If the phase is 'train' then adds data augmentation.
    Otherwise, does only resize and normalization.
    :param phase: Either 'train', 'validation' or 'test'. Default: 'train'
    :return: Transformation function to be applied to the data.
    """
    if phase == "train":
        return a.Compose([
            a.Resize(224, 224),
            a.HorizontalFlip(),
            a.VerticalFlip(),
            a.ShiftScaleRotate(),
            a.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.2),
            a.GaussNoise(std_range=(0, 16), p=0.1),
            a.CoarseDropout(num_holes_range=(1, 2), hole_height_range=(4, 8), hole_width_range=(4, 8), p=0.1),
            a.GridDistortion(num_steps=5, distort_limit=0.1, p=0.1),
            a.OpticalDistortion(distort_limit=(-0.2, 0.2), p=0.1),
            a.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ])
    elif phase == "validation" or phase == "test":
        return a.Compose([
            a.Resize(224, 224),
            a.Normalize(),
            ToTensorV2(),
        ])
    else:
        raise ValueError(f"Invalid phase: {phase}")


def get_data_loader(data_path, data_file: str, phase="train", train_val_split=0.2, batch_size = 32):
    """
    Prepares data loader for the given phase.
    :param data_path: Absolute path to data folder.
    :param data_file: Absolute path to data file.
    :param phase: Either 'train' or 'test'. Default: 'train'.
    :param train_val_split: Percentage of training and validation data. Default: 0.2.
    :return: If 'train' is passed, returns two data loaders for train and validation.
    If 'test' is passed, returns one data loader for test.
    """
    df = pd.read_csv(join(data_path, data_file))
    df["image_path"] = df["image_id"].apply(lambda x: join(data_path, "images", f"{x}.jpg"))
    if phase == "train":
        train_df, val_df = train_test_split(df, test_size=train_val_split, random_state=42)

        train_dataset = PlantDataset(train_df, transforms=get_transforms("train"))
        val_dataset = PlantDataset(val_df, transforms=get_transforms("validation"))

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)

        return train_loader, val_loader
    elif phase == "test":
        test_dataset = TestDataset(df, transforms=get_transforms("test"))

        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

        return test_loader
    else:
        raise AssertionError(f"Invalid phase: {phase}")


def mean_colwise_roc_auc(y_true, y_pred):
    """
    Computes mean column-wise ROC AUC.
    :param y_true: ground truth labels
    :param y_pred: predicted probabilities
    :return: Mean ROC AUC across all classes
    """

    scores = []
    num_classes = 4
    metric = BinaryAUROC()
    for i in range(num_classes):
        try:
            score = metric(y_pred[:, i], y_true[:, i])
            scores.append(score)
        except ValueError:
            # Vital if there is no label for at least one class
            continue

    return sum(scores) / num_classes if scores else 0.0
