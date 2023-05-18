import os
import copy
import h5py
import numpy as np
import torch

import datasets

AES_Sbox = np.array([
            0x63, 0x7C, 0x77, 0x7B, 0xF2, 0x6B, 0x6F, 0xC5, 0x30, 0x01, 0x67, 0x2B, 0xFE, 0xD7, 0xAB, 0x76,
            0xCA, 0x82, 0xC9, 0x7D, 0xFA, 0x59, 0x47, 0xF0, 0xAD, 0xD4, 0xA2, 0xAF, 0x9C, 0xA4, 0x72, 0xC0,
            0xB7, 0xFD, 0x93, 0x26, 0x36, 0x3F, 0xF7, 0xCC, 0x34, 0xA5, 0xE5, 0xF1, 0x71, 0xD8, 0x31, 0x15,
            0x04, 0xC7, 0x23, 0xC3, 0x18, 0x96, 0x05, 0x9A, 0x07, 0x12, 0x80, 0xE2, 0xEB, 0x27, 0xB2, 0x75,
            0x09, 0x83, 0x2C, 0x1A, 0x1B, 0x6E, 0x5A, 0xA0, 0x52, 0x3B, 0xD6, 0xB3, 0x29, 0xE3, 0x2F, 0x84,
            0x53, 0xD1, 0x00, 0xED, 0x20, 0xFC, 0xB1, 0x5B, 0x6A, 0xCB, 0xBE, 0x39, 0x4A, 0x4C, 0x58, 0xCF,
            0xD0, 0xEF, 0xAA, 0xFB, 0x43, 0x4D, 0x33, 0x85, 0x45, 0xF9, 0x02, 0x7F, 0x50, 0x3C, 0x9F, 0xA8,
            0x51, 0xA3, 0x40, 0x8F, 0x92, 0x9D, 0x38, 0xF5, 0xBC, 0xB6, 0xDA, 0x21, 0x10, 0xFF, 0xF3, 0xD2,
            0xCD, 0x0C, 0x13, 0xEC, 0x5F, 0x97, 0x44, 0x17, 0xC4, 0xA7, 0x7E, 0x3D, 0x64, 0x5D, 0x19, 0x73,
            0x60, 0x81, 0x4F, 0xDC, 0x22, 0x2A, 0x90, 0x88, 0x46, 0xEE, 0xB8, 0x14, 0xDE, 0x5E, 0x0B, 0xDB,
            0xE0, 0x32, 0x3A, 0x0A, 0x49, 0x06, 0x24, 0x5C, 0xC2, 0xD3, 0xAC, 0x62, 0x91, 0x95, 0xE4, 0x79,
            0xE7, 0xC8, 0x37, 0x6D, 0x8D, 0xD5, 0x4E, 0xA9, 0x6C, 0x56, 0xF4, 0xEA, 0x65, 0x7A, 0xAE, 0x08,
            0xBA, 0x78, 0x25, 0x2E, 0x1C, 0xA6, 0xB4, 0xC6, 0xE8, 0xDD, 0x74, 0x1F, 0x4B, 0xBD, 0x8B, 0x8A,
            0x70, 0x3E, 0xB5, 0x66, 0x48, 0x03, 0xF6, 0x0E, 0x61, 0x35, 0x57, 0xB9, 0x86, 0xC1, 0x1D, 0x9E,
            0xE1, 0xF8, 0x98, 0x11, 0x69, 0xD9, 0x8E, 0x94, 0x9B, 0x1E, 0x87, 0xE9, 0xCE, 0x55, 0x28, 0xDF,
            0x8C, 0xA1, 0x89, 0x0D, 0xBF, 0xE6, 0x42, 0x68, 0x41, 0x99, 0x2D, 0x0F, 0xB0, 0x54, 0xBB, 0x16
], dtype=np.uint8)

def to_byte(x):
    return x.astype(np.uint8)

def to_bits(x):
    return np.unpackbits(x).astype(np.uint8)

def to_hw(x):
    return np.sum(to_bits(x), dtype=np.uint8)

def compute_ascadv1_labels(metadata, bytes, mode='bytes'):
    labels = [AES_Sbox[metadata['plaintext'][byte] ^ metadata['key'][byte]] for byte in bytes]
    to_repr = {'bytes': to_byte, 'bits': to_bits, 'hw': to_hw}[mode]
    labels = {mode+'_'+str(byte): to_repr(label) for byte, label in zip(bytes, labels)}
    return labels

class _ASCADBase(torch.utils.data.Dataset):
    def __init__(self, dataset_name, compressed_filename, train=True, transform=None, target_transform=None, truncate_length=None, store_dataset_in_ram=True):
        super().__init__()
        
        if not datasets.is_downloaded(dataset_name):
            raise Exception(
                'Dataset has not been downloaded. Re-run with argument \'--download-dataset {}\' to download it.'.format(dataset_name)
            )
        self.train = train
        self.database_file = h5py.File(
            os.path.join(datasets.get_path(dataset_name), compressed_filename), 'r'
        )
        self.sbox = AES_Sbox
        
        assert len(self.database_file['Profiling_traces/traces']) == len(self.database_file['Profiling_traces/labels']) == len(self.database_file['Profiling_traces/metadata'])
        assert len(self.database_file['Attack_traces/traces']) == len(self.database_file['Attack_traces/labels']) == len(self.database_file['Attack_traces/metadata'])
        self.length = len(self.database_file['Profiling_traces/traces']) if train else len(self.database_file['Attack_traces/traces'])
        if truncate_length is not None:
            assert 0 < truncate_length < self.length
            self.length = truncate_length
        self.data_shape = (1, self.database_file['Profiling_traces/traces'][0].shape[0])
        self.transform = transform
        self.target_transform = target_transform
        
        self.index_database = lambda key: self.database_file['/'.join(('Profiling_traces' if self.train else 'Attack_traces', key))]
        if store_dataset_in_ram:
            self.data = np.array(self.index_database('traces'), dtype=np.int8)
            self.targets = np.array(self.index_database('labels'), dtype=np.uint8)
            self.metadata = {
                'plaintext': np.array(self.index_database('metadata')['plaintext'], dtype=np.uint8),
                'key': np.array(self.index_database('metadata')['key'], dtype=np.uint8),
                'masks': np.array(self.index_database('metadata')['masks'], dtype=np.uint8)
            }
            self.get_data = lambda idx: self.data[idx]
            self.get_target = lambda idx: self.targets[idx]
            self.get_metadata = lambda idx: {key: value[idx] for key, value in self.metadata.items()}
        else:
            self.get_data = lambda idx: np.array(self.index_database('traces')[idx], dtype=np.int8)
            self.get_target = lambda idx: np.array(self.index_database('labels')[idx], dtype=np.uint8)
            self.get_metadata = lambda idx: {
                'plaintext': np.array(self.index_database('metadata')['plaintext'][idx], dtype=np.uint8),
                'key': np.array(self.index_database('metadata')['key'][idx], dtype=np.uint8),
                'masks': np.array(self.index_database('metadata')['masks'][idx], dtype=np.uint8)
            }
    
    def __getitem__(self, idx, return_metadata=False):
        idx = idx % self.length
        data = self.get_data(idx)
        orig_target = self.get_target(idx)
        metadata = self.get_metadata(idx)
        target = self.compute_target(metadata)
        to_repr = {'bytes': to_byte, 'bits': to_bits, 'hw': to_hw}[self.data_repr]
        assert target[self.data_repr+'_2'] == to_repr(orig_target)
        #data = (data - self.trace_range[0]) / (self.trace_range[1]-self.trace_range[0])
        data = (data-np.mean(data))/np.std(data)
        data = torch.tensor(data, dtype=torch.float).view(*self.data_shape)
        target = {key: torch.tensor(value, dtype=torch.long).squeeze() for key, value in target.items()}
        if self.transform is not None:
            data = self.transform(data)
        if self.target_transform is not None:
            target = self.target_transform(target)
        if return_metadata:
            return data, target, metadata
        else:
            return data, target
    
    def __len__(self):
        return self.length
    
class ASCADV1Fixed(_ASCADBase):
    dataset_name = 'ASCAD-V1-Fixed'
    download_url = r'https://www.data.gouv.fr/s/resources/ascad/20180530-163000/ASCAD_data.zip'
    member_to_unzip = r'ASCAD_data/ASCAD_databases/ASCAD.h5'
    compressed_filename = r'ascadv1-fixed.h5'
    
    def __init__(self, bytes=None, data_repr='bytes', **kwargs):
        super().__init__(self.__class__.dataset_name, self.__class__.compressed_filename, **kwargs)
        if bytes is None:
            bytes = np.arange(16)
        self.bytes = np.array(bytes)
        self.bytes.sort()
        self.data_repr = data_repr
        self.compute_target = lambda metadata: compute_ascadv1_labels(metadata, self.bytes, mode=self.data_repr)
        self.trace_range = (-66.0, 47.0)
    
class ASCADV1Variable(_ASCADBase):
    dataset_name = 'ASCAD-V1-Variable'
    download_url = r'https://static.data.gouv.fr/resources/ascad-atmega-8515-variable-key/20190903-083349/ascad-variable.h5'
    compressed_filename = r'ascadv1-variable.h5'
    
    def __init__(self, bytes=None, data_repr='bytes', **kwargs):
        super().__init__(self.__class__.dataset_name, self.__class__.compressed_filename, **kwargs)
        if bytes is None:
            bytes = np.arange(16)
        self.bytes = np.array(bytes)
        self.bytes.sort()
        self.data_repr = data_repr
        self.compute_target = lambda metadata: compute_ascadv1_labels(metadata, self.bytes, mode=self.data_repr)

class ASCADV2(_ASCADBase):
    dataset_name = 'ASCAD-V2'
    download_url = r'https://files.data.gouv.fr/anssi/ascadv2/ascadv2-extracted.h5'
    compressed_filename = r'ascadv2.h5'
    
    def __init__(self, **kwargs):
        super().__init__(self.__class__.dataset_name, self.__class__.compressed_filename, **kwargs)
        
    def __getitem__(self, idx, get_metadata=False):
        data, target, metadata = self.data[idx], self.targets[idx], self.metadata[idx] if get_metadata else None
        data = torch.tensor(data, dtype=torch.float).view(*self.data_shape)
        print(target)
        assert False
        #target = torch.tensor(target, dtype=torch.long)
        if self.transform is not None:
            data = self.transform(data)
        if self.target_transform is not None:
            target = self.target_transform(target)
        if get_metadata:
            return data, target, metadata
        else:
            return data, target