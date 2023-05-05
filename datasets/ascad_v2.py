import os
import h5py
import numpy as np
import torch

import datasets

class ASCADV2(torch.utils.data.Dataset):
    dataset_name = 'ASCAD-V2'
    download_url = r'https://files.data.gouv.fr/anssi/ascadv2/ascadv2-extracted.h5'
    compressed_filename = r'ascadv2-extracted.h5'
    
    def __init__(self, train=True, transform=None, target_transform=None):
        super().__init__()
        
        self.transform = transform
        self.target_transform = target_transform
        
        if not datasets.is_downloaded(self.__class__.dataset_name):
            raise Exception('Dataset has not been downloaded. Re-run with argument \'--download-dataset {}\' to download it.'.format(
                self.__class__.dataset_name))
        
        database_file = h5py.File(os.path.join(datasets.get_path(self.__class__.dataset_name), self.__class__.compressed_filename), 'r')
        
        if train:
            self.data = np.array(database_file['Profiling_traces/traces'], dtype=np.int8)
            self.targets = np.array(database_file['Profiling_traces/labels'])
            self.metadata = database_file['Profiling_traces/metadata']
        else:
            self.data = np.array(database_file['Attack_traces/traces'], dtype=np.int8)
            self.targets = np.array(database_file['Attack_traces/labels'])
            self.metadata = database_file['Attack_traces/metadata']
        assert len(self.data) == len(self.targets) == len(self.metadata)
        self.length = len(self.data)
        self.data_shape = (1, self.data[0].shape[0])
        
    def __getitem__(self, idx, get_metadata=False):
        data, target, metadata = self.data[idx], self.targets[idx], self.metadata[idx] if get_metadata else None
        data = torch.tensor(data, dtype=torch.float).view(*self.data_shape)
        #print(target)
        #target = torch.tensor(target, dtype=torch.long)
        if self.transform is not None:
            data = self.transform(data)
        if self.target_transform is not None:
            target = self.target_transform(target)
        if get_metadata:
            return data, target, metadata
        else:
            return data, target
    
    def __len__(self):
        return self.length