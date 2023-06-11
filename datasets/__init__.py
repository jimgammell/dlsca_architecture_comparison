import os
import time
import gdown
import shutil
import zipfile
from collections import OrderedDict
from datasets.ascad import ASCADV1Fixed, ASCADV1Fixed_DS50, ASCADV1Fixed_DS100, ASCADV1Variable, ASCADV2
from datasets.aes_rd import AES_RD

AVAILABLE_DATASETS = OrderedDict([
    (dataset_class.dataset_name, dataset_class)
    for dataset_class in [ASCADV1Fixed, ASCADV1Fixed_DS50, ASCADV1Fixed_DS100, ASCADV1Variable]
])

def get_available_datasets():
    return list(AVAILABLE_DATASETS.keys())

def download_file(url, dest, force=False):
    if force or not(os.path.exists(dest)):
        gdown.download(url, dest, quiet=False)

def unzip_file(src, dest, member=None, remove=True):
    try:
        with zipfile.ZipFile(src, 'r') as zip_ref:
            if member is None:
                zip_ref.extractall(path=dest)
            else:
                zip_ref.extract(member, path=dest)
        if remove:
            os.remove(src)
    except zipfile.BadZipFile:
        time.sleep(10)
        unzip_file(src, member=member, remove=remove)

def check_name(dataset_name):
    if not dataset_name in get_available_datasets():
        raise ValueError('Unrecognized dataset name: \'{}\''.format(dataset_name))

def check_downloaded(dataset_name):
    if not is_downloaded(dataset_name):
        raise Exception(
            'Dataset has not been downloaded. Re-run with argument \'--download-dataset {}\' to download it.'.format(dataset_name)
        )
        
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
    if dataset_class.download_url.split('.')[-1] == 'zip':
        dl_path = os.path.join(dataset_path, os.path.split(dataset_class.download_url)[-1])
        download_file(dataset_class.download_url, dl_path, force=True)
        unzip_file(dl_path, dataset_path, member=dataset_class.member_to_unzip, remove=True)
        os.rename(
            os.path.join(dataset_path, dataset_class.member_to_unzip), os.path.join(dataset_path, dataset_class.compressed_filename)
        )
        shutil.rmtree(os.path.join(dataset_path, dataset_class.member_to_unzip.split(os.sep)[0]))
    else:
        download_file(dataset_class.download_url, os.path.join(dataset_path, dataset_class.compressed_filename), force=True)
    if hasattr(dataset_class, 'extract_dataset'):
        dataset_class.extract_dataset()
    open(os.path.join(dataset_path, 'done.txt'), 'w').close()