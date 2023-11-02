from git.RecognitionofSignLanguage.utils.Cfg import Cfg
import numpy as np

class Landmarks:
    LIPS_IDXS0 = np.array([
        61, 185, 40, 39, 37, 0, 267, 269, 270, 409,
        291, 146, 91, 181, 84, 17, 314, 405, 321, 375,
        78, 191, 80, 81, 82, 13, 312, 311, 310, 415,
        95, 88, 178, 87, 14, 317, 402, 318, 324, 308,
    ], dtype=np.int32)
    if Cfg.MORE_LANDMARKS:
        NOSE_IDXS0 = np.array([
            1, 2, 98, 327
        ], dtype=np.int32)
        REYE_IDXS0 = np.array([
            33, 7, 163, 144, 145, 153, 154, 155, 133,
            246, 161, 160, 159, 158, 157, 173,
        ], dtype=np.int32)
        LEYE_IDXS0 = np.array([
            263, 249, 390, 373, 374, 380, 381, 382, 362,
            466, 388, 387, 386, 385, 384, 398], dtype=np.int32)
    else:
        NOSE_IDXS0 = np.array([], dtype=np.int32)
        REYE_IDXS0 = np.array([], dtype=np.int32)
        LEYE_IDXS0 = np.array([], dtype=np.int32)

    # Landmark indices in original data
    LEFT_HAND_IDXS0 = np.arange(468, 489)
    RIGHT_HAND_IDXS0 = np.arange(522, 543)
    LEFT_POSE_IDXS0 = np.array([502, 504, 506, 508, 510], dtype=np.int32)
    RIGHT_POSE_IDXS0 = np.array([503, 505, 507, 509, 511], dtype=np.int32)

    LANDMARK_IDXS_LEFT_DOMINANT0 = np.concatenate(
        (REYE_IDXS0, LEYE_IDXS0, NOSE_IDXS0, LIPS_IDXS0, LEFT_HAND_IDXS0, LEFT_POSE_IDXS0))

    LANDMARK_IDXS_RIGHT_DOMINANT0 = np.concatenate(
        (REYE_IDXS0, LEYE_IDXS0, NOSE_IDXS0, LIPS_IDXS0, RIGHT_HAND_IDXS0, RIGHT_POSE_IDXS0))

    HAND_IDXS0 = np.concatenate((LEFT_HAND_IDXS0, RIGHT_HAND_IDXS0), axis=0)
    N_COLS = LANDMARK_IDXS_LEFT_DOMINANT0.size

    # Landmark indices in processed data
    REYE_IDXS = np.argwhere(np.isin(LANDMARK_IDXS_LEFT_DOMINANT0, REYE_IDXS0)).squeeze()
    LEYE_IDXS = np.argwhere(np.isin(LANDMARK_IDXS_LEFT_DOMINANT0, LEYE_IDXS0)).squeeze()
    NOSE_IDXS = np.argwhere(np.isin(LANDMARK_IDXS_LEFT_DOMINANT0, NOSE_IDXS0)).squeeze()
    LIPS_IDXS = np.argwhere(np.isin(LANDMARK_IDXS_LEFT_DOMINANT0, LIPS_IDXS0)).squeeze()
    LEFT_HAND_IDXS = np.argwhere(np.isin(LANDMARK_IDXS_LEFT_DOMINANT0, LEFT_HAND_IDXS0)).squeeze()
    RIGHT_HAND_IDXS = np.argwhere(np.isin(LANDMARK_IDXS_LEFT_DOMINANT0, RIGHT_HAND_IDXS0)).squeeze()
    HAND_IDXS = np.argwhere(np.isin(LANDMARK_IDXS_LEFT_DOMINANT0, HAND_IDXS0)).squeeze()
    POSE_IDXS = np.argwhere(np.isin(LANDMARK_IDXS_LEFT_DOMINANT0, LEFT_POSE_IDXS0)).squeeze()

    # Start points
    REYE_START = 0
    LEYE_START = REYE_IDXS.size
    NOSE_START = LEYE_START + LEYE_IDXS.size
    LIPS_START = NOSE_START + NOSE_IDXS.size
    LEFT_HAND_START = LIPS_IDXS.size
    RIGHT_HAND_START = LEFT_HAND_START + LEFT_HAND_IDXS.size
    POSE_START = RIGHT_HAND_START + RIGHT_HAND_IDXS.size
