import os
from scipy.io import loadmat
import numpy as np
import torch

import datasets

class AES_RD(torch.utils.data.Dataset):
    dataset_name = 'AES-RD'
    download_url = r'https://github.com/ikizhvatov/randomdelays-traces/raw/master/ctraces_fm16x4_2.mat'
    compressed_filename = 'aes-rd.mat'
    
    def __init__(self, train=True, transform=None, target_transform=None, truncate_length=None, **kwargs):
        super().__init__()
        
        datasets.check_downloaded(self.__class__.dataset_name)
        self.train = train
        database_file = loadmat(
            os.path.join(datasets.get_path(self.__class__.dataset_name), self.__class__.compressed_filename)
        )
        self.traces = database_file['CompressedTraces'].T
        self.plaintexts = database_file['plaintext']
        assert len(self.traces) == len(self.plaintexts)
        self.length = len(self.traces)
        self.data_shape = (1, self.traces.shape[1])
        self.transform = transform
        self.target_transform = target_transform
        if truncate_length is not None:
            assert 0 < truncate_length < self.length
            self.length = truncate_length
        
    def __getitem__(self, idx):
        idx = id % self.length
        data = self.traces[idx]
        plaintext = self.plaintexts[idx]
        data = (data-np.mean(data))/np.std(data)
        data = torch.tensor(data, dtype=torch.float).view(*self.data_shape)
        if self.transform is not None:
            data = self.transform(data)
        if self.target_transform is not None:
            target = self.target_transform(plaintext)
        return data, target
    
    def __len__(self):
        return self.length