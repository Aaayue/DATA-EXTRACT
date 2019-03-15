import numpy as np
import glob
from os import path
"""
merge several *_TRAIN.npz files into one
"""

# path which contain all the '*_TRAIN.npz' file
train_path = path.join(
    path.expanduser('~'), 'data2/citrus/demo/sample_result/test_merge'
)
# final save npz path
final_train_file = path.join(
    train_path, 'TD_S3_L3a_20190313T164932_12540_12541_TRAIN.npz'
)

train_file = glob.glob(path.join(train_path, '*TRAIN.npz'))

print(train_file)

feats = []
labs = []
for file in train_file:
    a = np.load(file)
    feat = a['features']
    lab = a['labels']
    if len(feats) == 0:
        feats = feat
        labs = lab
    else:
        feats = np.concatenate((feats, feat), axis=0)
        labs = np.concatenate((labs, lab), axis=0)
        print('feature shape ', feats.shape)
        print('labels shape ', labs.shape)

print('final feature shape: ', feats.shape)
print('final label shape: ', labs.shape)

np.savez(final_train_file, features=feats, labels=labs)
