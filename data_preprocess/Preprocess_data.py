import os
import numpy as np
import pandas as pd

from tqdm.notebook import tqdm
from sklearn.model_selection import GroupShuffleSplit

from git.RecognitionofSignLanguage.utils.Cfg import Cfg
from Preprocess_layer import PreprocessLayer
from git.RecognitionofSignLanguage.utils.Landmark_indices import Landmarks as lm

def load_relevant_data_subset(pq_path):
    if Cfg.DEPTH:
        data_columns = ['x', 'y', 'z']
    else:
        data_columns = ['x', 'y']
    data = pd.read_parquet(pq_path, columns=data_columns)
    n_frames = int(len(data) / Cfg.ROWS_PER_FRAME)
    data = data.values.reshape(n_frames, Cfg.ROWS_PER_FRAME, len(data_columns))
    return data.astype(np.float32)


def get_data(file_path):
    # Load Raw Data
    data = load_relevant_data_subset(file_path)
    # Process Data Using Tensorflow
    preprocess_layer = PreprocessLayer()
    data = preprocess_layer(data)
    return data

# Get the full dataset
def preprocess_data(more_landmarks = False, depth = True):
    # Create arrays to save data
    X = np.zeros([N_SAMPLES, Cfg.INPUT_SIZE, lm.N_COLS, Cfg.N_DIMS], dtype=np.float32)
    y = np.zeros([N_SAMPLES], dtype=np.int32)
    NON_EMPTY_FRAME_IDXS = np.full([N_SAMPLES, Cfg.INPUT_SIZE], -1, dtype=np.float32)

    # Fill X, y and NON_EMPTY_FRAME_IDXS
    for row_idx, (file_path, sign_ord) in enumerate(tqdm(train[['file_path', 'sign_ord']].values)):
        data, non_empty_frame_idxs = get_data(file_path)
        X[row_idx] = data
        y[row_idx] = sign_ord
        NON_EMPTY_FRAME_IDXS[row_idx] = non_empty_frame_idxs
        # Sanity check, data should not contain NaN values
        if row_idx % 5000 == 0:
            print(f"Generate {row_idx} data")
        if np.isnan(data).sum() > 0:
            print(row_idx)
            return data
    return X, y, NON_EMPTY_FRAME_IDXS

if __name__ == '__main__':

    train = pd.read_csv(f'{Cfg.SAVE_PATH}train.csv')
    N_SAMPLES = len(train)

    # Preprocess All Data From Scratch
    X, y, NON_EMPTY_FRAME_IDXS = preprocess_data(more_landmarks=Cfg.MORE_LANDMARKS, depth=Cfg.DEPTH)

    # Create Folder
    name_folder = f'data_frame_{Cfg.INPUT_SIZE}_more_landmarks_{Cfg.MORE_LANDMARKS}_depth_{Cfg.DEPTH}'
    if not os.path.exists(f'{Cfg.SAVE_PATH}{name_folder}'):
        os.mkdir(f'{Cfg.SAVE_PATH}{name_folder}')
    print(f'Path where save data - {Cfg.SAVE_PATH}{name_folder}')

    # Save data
    np.save(f'{Cfg.SAVE_PATH}{name_folder}/X.npy', X)
    np.save(f'{Cfg.SAVE_PATH}{name_folder}/y.npy', y)
    np.save(f'{Cfg.SAVE_PATH}{name_folder}/NON_EMPTY_FRAME_IDXS.npy', y)

