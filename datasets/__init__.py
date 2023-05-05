import os
import gdown
from zipfile import ZipFile
from collections import OrderedDict
from datasets.ascad_v2 import ASCADV2

AVAILABLE_DATASETS = OrderedDict([
    (dataset_class.dataset_name, dataset_class)
    for dataset_class in [ASCADV2]
])

def get_available_datasets():
    return list(AVAILABLE_DATASETS.keys())

def download_file(url, dest, force=False):
    if force or not(os.path.exists(dest)):
        gdown.download(url, dest, quiet=False)

def check_name(dataset_name):
    if not dataset_name in get_available_datasets():
        raise ValueError('Unrecognized dataset name: \'{}\''.format(dataset_name))
        
def get_path(dataset_name):
    check_name(dataset_name)
    return os.path.join('.', 'downloads', dataset_name)

def get_class(dataset_name):
    check_name(dataset_name)
    return AVAILABLE_DATASETS[dataset_name]
    
def is_downloaded(dataset_name):
    check_name(dataset_name)
    return os.path.exists(os.path.join(get_path(dataset_name), 'done.txt'))
    
def download_dataset(dataset_name, force=False):
    check_name(dataset_name)
    if is_downloaded(dataset_name) and not force:
        raise ValueError('Dataset has already been downloaded: \'{}\'. Re-run with --force to re-download.'.format(dataset_name))
    dataset_path = get_path(dataset_name)
    dataset_class = get_class(dataset_name)
    if os.path.exists(dataset_path):
        if os.path.exists(os.path.join(dataset_path, 'done.txt')):
            os.remove(os.path.join(dataset_path, 'done.txt'))
    else:
        os.makedirs(dataset_path, exist_ok=True)
    download_file(dataset_class.download_url, os.path.join(dataset_path, dataset_class.compressed_filename), force=True)
    if hasattr(dataset_class, 'extract_dataset'):
        dataset_class.extract_dataset()
    open(os.path.join(dataset_path, 'done.txt'), 'w').close()