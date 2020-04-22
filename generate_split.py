import numpy as np
import os
import json

data_path = '../data/RedLights2011_Medium'
gts_path = '../data/hw02_annotations'
split_path = '../data/hw02_splits'
os.makedirs(gts_path, exist_ok=True) # create directory if needed
os.makedirs(split_path, exist_ok=True) # create directory if needed

split_test = True # set to True and run when annotations are available
train_frac = 0.85


# get sorted list of files:
file_names = sorted(os.listdir(data_path))

# remove any non-JPEG files:
file_names = [f for f in file_names if '.jpg' in f]

# split file names into train and test
n_files = len(file_names)
n_train = np.round(n_files*train_frac).astype(int)
n_test = n_files-n_train

rng = np.random.default_rng(2020) # to ensure you always get the same train/test split
idc = rng.permutation(np.arange(n_files))

file_names_train = [file_names[i] for i in idc[:n_train]]
file_names_test = [file_names[i] for i in idc[n_train:]]

assert (len(file_names_train) + len(file_names_test)) == len(file_names)
assert len(np.intersect1d(file_names_train,file_names_test)) == 0

np.save(os.path.join(split_path,'file_names_train.npy'),file_names_train)
np.save(os.path.join(split_path,'file_names_test.npy'),file_names_test)

if split_test:
    with open(os.path.join(gts_path, 'formatted_annotations_students.json'),'r') as f:
        gts = json.load(f)
    
    gts_train = {k: gts[k] for k in file_names_train}
    gts_test = {k: gts[k] for k in file_names_test}
    
    with open(os.path.join(gts_path, 'annotations_train.json'),'w') as f:
        json.dump(gts_train,f)
    
    with open(os.path.join(gts_path, 'annotations_test.json'),'w') as f:
        json.dump(gts_test,f)
    
    