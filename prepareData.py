import os
import sys
import random
import shutil


def prepare_dataset(data_dir):
    sources = find_sources(data_dir)


    for nameFile in sources:
        fn = lambda x: str(hash(x) % ((sys.maxsize + 1) * 2)) + '.JPG'
        names = fn(nameFile)
        os.rename(os.path.join(data_dir, nameFile), os.path.join(data_dir,names))


def find_sources(data_dir, file_ext='.JPG', shuffle=True):
    
    sources = [
        (os.path.join(name))
        for name in os.listdir(os.path.join(data_dir))
        if name.endswith(file_ext) or name.endswith(file_ext.lower())

    ]
    random.shuffle(sources)

    return sources 

"""
Start 
"""
prepare_dataset('image_files')


