import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from git.RecognitionofSignLanguage.utils.Cfg import Cfg
from git.RecognitionofSignLanguage.utils.Landmarks import Landmarks as lm


def get_mean_std(INDX, X_train):
    MEAN_X = np.zeros([INDX.size], dtype=np.float32)
    MEAN_Y = np.zeros([INDX.size], dtype=np.float32)
    MEAN_Z = np.zeros([INDX.size], dtype=np.float32)
    STD_X = np.zeros([INDX.size], dtype=np.float32)
    STD_Y = np.zeros([INDX.size], dtype=np.float32)
    STD_Z = np.zeros([INDX.size], dtype=np.float32)

    for col, ll in enumerate(
            tqdm(np.transpose(X_train[:, :, INDX], [2, 3, 0, 1]).reshape([INDX.size, Cfg.N_DIMS, -1]))):
        for dim, l in enumerate(ll):
            v = l[np.nonzero(l)]
            if dim == 0:  # X
                MEAN_X[col] = v.mean()
                STD_X[col] = v.std()
            if dim == 1:  # Y
                MEAN_Y[col] = v.mean()
                STD_Y[col] = v.std()
            if dim == 2:  # Z
                MEAN_Z[col] = v.mean()
                STD_Z[col] = v.std()

    if Cfg.DEPTH:
        MEAN = np.array([MEAN_X, MEAN_Y, MEAN_Z]).T
        STD = np.array([STD_X, STD_Y, STD_Z]).T
    else:
        MEAN = np.array([MEAN_X, MEAN_Y]).T
        STD = np.array([STD_X, STD_Y]).T

    return MEAN, STD

def get_all_mean_std(X_train):
    LIPS_MEAN, LIPS_STD = get_mean_std(lm.LIPS_IDXS, X_train)
    HAND_MEAN, HAND_STD = get_mean_std(lm.LEFT_HAND_IDXS, X_train)
    POSE_MEAN, POSE_STD = get_mean_std(lm.POSE_IDXS, X_train)

    return {'lips_mean': LIPS_MEAN, 'lips_std': LIPS_STD, 'hand_mean': HAND_MEAN, 'hand_std': HAND_STD, 'pose_mean': POSE_MEAN, 'pose_std': POSE_STD}
