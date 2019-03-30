import numpy as np
import os


class TrainDataFlat:
    def __init__(self, file):
        self.file = file
        self.feature, self.label = self.load_npz()

    def load_npz(self):
        try:
            data = np.load(self.file)
            feature = data['features'].tolist()
            label = data['labels'].tolist()
            return feature, label
        except Exception as e:
            print('Load {} error: {}'.format(self.file, e))
            return None

    def data_flat(self):
        res = []
        keys = list(self.feature.keys())
        keys.sort()
        print(keys)
        for key in keys:
            data = self.feature[key]
            if not list(res):
                res = np.array(data)
            else:
                res = np.concatenate((res, data), axis=1)
            print("     -> ", res.shape)
        return res


def batch_run(file_list):
    sample_name = os.path.join(
        os.path.dirname(file_list[0]),
        os.path.basename(file_list[0]).split('extract')[0] + 'TRAIN.npz'
    )
    print(sample_name)
    final_feat = []
    final_lab = []
    print('process data flatten')
    for file in file_list:
        print('file in operating {}'.format(file))
        TF = TrainDataFlat(file)
        _, lab = TF.load_npz()
        result = TF.data_flat()
        if not final_lab:
            final_feat = result
            final_lab = lab
        else:
            final_feat = np.concatenate((final_feat, result), axis=0)
            final_lab = final_lab + lab
            # final_lab = np.concatenate((final_lab, lab), axis=1)
        print("file feature     -> ", np.array(final_feat).shape)
        print("file lab         -> ", len(final_lab))
        print('\n')
    np.savez(sample_name, features=final_feat, labels=final_lab)
    print("finish data prepare")
    return sample_name
