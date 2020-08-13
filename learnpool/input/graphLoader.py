import os
import torch
from torch.utils.data import Dataset


def make_dataset(root, mode):
    """
    Makes a list of files in the for train/valid/test from the main folder
    """
    items = []
    if mode == 'train':
        files_list = os.listdir(os.path.join(root, 'train'))
        files_list.sort()
        for it in range(len(files_list)):
            item = os.path.join(os.path.join(root, 'train'), files_list[it])
            items.append(item)

    elif mode == 'valid':
        files_list = os.listdir(os.path.join(root, 'valid'))
        files_list.sort()

        for it in range(len(files_list)):
            item = os.path.join(os.path.join(root, 'valid'), files_list[it])
            items.append(item)

    elif mode == 'test':
        files_list = os.listdir(os.path.join(root, 'test'))
        files_list.sort()

        for it in range(len(files_list)):
            item = os.path.join(os.path.join(root, 'test'), files_list[it])
            items.append(item)
    else:
        files_list = os.listdir(os.path.join(root, mode))
        files_list.sort()

        for it in range(len(files_list)):
            item = os.path.join(os.path.join(root, mode), files_list[it])
            items.append(item)

    return items


class GeometricDataset(Dataset):

    def __init__(self, mode, root_dir):
        """
        Simple class to load torch files from dataset folder
        Args:
            mode: 'train', 'valid', or 'test'
            root_dir: path to the dataset
        """
        self.root_dir = root_dir
        self.files = make_dataset(root_dir, mode)
        self.mode = mode

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        files_path = self.files[index]
        data = torch.load(files_path)
        return data
