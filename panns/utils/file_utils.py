import os

__all__ = ['create_folder',
           'get_sub_filepaths']

def create_folder(fd):
    if not os.path.exists(fd):
        os.makedirs(fd)
        
        
def get_sub_filepaths(folder):
    paths = []
    for root, dirs, files in os.walk(folder):
        for name in files:
            path = os.path.join(root, name)
            paths.append(path)
    return paths
