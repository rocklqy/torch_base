from torch.utils.data import Dataset
from PIL import Image
import numpy as np


class ListDataset(Dataset):
    def __init__(self, args, is_train):
        # usually we need args rather than single datalist to init the dataset
        super(self, ListDataset).__init__()
        if is_train:
            data_list = args.train_list
        else:
            data_list = args.val_list
        infos = [line.split() for line in open(data_list).readlines()]
        self.img_paths = [info[0] for info in infos]
        self.label_paths = [info[1] for info in infos]

    def preprocess(self, img, label):
        # you can add other process method or augment here
        return img, label

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img = Image.open(self.img_paths[idx])
        label = Image.open(self.label_paths[idx])
        img, label = self.preprocess(img, label)
        return img, label
