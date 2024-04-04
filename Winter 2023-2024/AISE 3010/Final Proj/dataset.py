import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import load_data

data = load_data.read_data_sets()

class EEGTrainingData(Dataset):
    def __init__(self, transform=None, target_transform=None):
        self.img_labels = data.train.labels
        self.img_dir = None
        self.transform = transform
        self.target_transform = target_transform

        self.eeg_train_data = data.train.data
    
    def __len__(self):
        return data.train.num_examples

    def __getitem__(self, idx):
        feats = self.eeg_train_data
        labels = self.img_labels

        feature = feats[idx]
        label = labels[idx]

        return feature, label

class EEGTestingData(Dataset):
    def __init__(self, transform=None, target_transform=None):
        self.img_labels = data.test.labels
        self.img_dir = None
        self.transform = transform
        self.target_transform = target_transform

        self.eeg_test_data = data.test.data
    
    def __len__(self):
        return data.train.num_examples

    def __getitem__(self, idx):
        feats = self.eeg_test_data
        labels = self.img_labels

        feature = feats[idx]
        label = labels[idx]

        return feature, label