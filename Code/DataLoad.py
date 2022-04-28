from torch.utils.data import Dataset
import numpy as np
import torch
from random import shuffle
import itertools

class BrainDataGenerator(Dataset):
    def __init__(self, train_names, mode='train'):
        """
        :param json_file:
        :param h5_file:
        """
        self.mode = mode

        self.pair = list(itertools.permutations(train_names, 2))
        shuffle(self.pair)

    def __len__(self):
        return len(self.pair)

    def __getitem__(self, index):
        self.data_A = np.load(self.pair[index][0])
        self.data_B = np.load(self.pair[index][1])

        return torch.Tensor(self.data_A),torch.Tensor(self.data_B)
