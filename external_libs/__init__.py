import os
from git import Repo

def get_path():
    return os.path.join('.', 'external_libs')

def download_file_from_git(filename, git_repo_url, force=False):
    file_path = os.path.join(get_path(), 'temperature_scaling.py')
    file_exists = os.path.exists(file_path)
    if force or not file_exists:
        if file_exists:
            os.remove(file_path)
        Repo.clone(git_repo_url, file_path)

def download_external_libs(force=False):
    download_file_from_git(
        r'https://github.com/gpleiss/temperature_scaling/blob/master/temperature_scaling.py',
        r'temperature_scaling.py',
        force=force
    )
    download_file_from_git(
        r'https://github.com/davda54/sam/blob/main/sam.py',
        r'sam.py',
        force=force
    )

download_external_libs()