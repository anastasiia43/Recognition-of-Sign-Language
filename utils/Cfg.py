class Cfg:

    MORE_LANDMARKS = False
    DEPTH = False  # Use z coordinate

    if DEPTH:
        N_DIMS = 3
    else:
        N_DIMS = 2

    INPUT_SIZE = 22  # Number frame

    DATA_RAW = '../../../asl-signs/'  # Folder with rare data
    SAVE_PATH = '../preprocess_data/'  # Folder where save preprocess data
    INDEX_MAP_FILE = f'{SAVE_PATH}sign_to_prediction_index_map.json'  # Json file with name class

    FOLDER_SAVE = f'data_frame_{INPUT_SIZE}_more_landmarks_{MORE_LANDMARKS}_depth_{DEPTH}'  # Name folder where save model or preprocess data

    MODEL_OUT_PATH = f'../checkpoints/lstm_checkpoints/{FOLDER_SAVE}/'  # Path for save model
    PATH_DATA = f'{SAVE_PATH}{FOLDER_SAVE}'  # Path for save preprocess data

    # Model Config
    EPOCHS = 150
    BATCH_SIZE = 64
    LR = .0001

    NUM_CLASSES = 250
    DROPOUTS = [0.4, 0.4]
    STARTING_LAYER_SIZE = 1024

    SEED = 36

    ROWS_PER_FRAME = 543
