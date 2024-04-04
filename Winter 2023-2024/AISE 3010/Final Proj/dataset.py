import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import load_data

data = load_data.read_data_sets()

class EEGTrainingData(Dataset):
    def __init__(self, transform=None, target_transform=None):
        self.img_labels = torch.from_numpy(data.train.labels).long()
        self.img_dir = None
        self.transform = transform
        self.target_transform = target_transform

        self.eeg_train_data = torch.from_numpy(data.train.data).float()
    
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
        self.img_labels = torch.from_numpy(data.test.labels).long()
        self.img_dir = None
        self.transform = transform
        self.target_transform = target_transform

        self.eeg_test_data = torch.from_numpy(data.test.data).float()
    
    def __len__(self):
        return data.test.num_examples

    def __getitem__(self, idx):
        feats = self.eeg_test_data
        labels = self.img_labels

        feature = feats[idx]
        label = labels[idx]

        return feature, label

