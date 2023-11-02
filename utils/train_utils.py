import os
import random
import numpy as np
import json
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

# Import own class
from git.RecognitionofSignLanguage.utils.Cfg import Cfg


class TrainUtils:

    def seed_it_all(self, seed=Cfg.SEED):
        os.environ["PYTHONHASHSEED"] = str(seed)
        random.seed(seed)
        np.random.seed(seed)
        tf.random.set_seed(seed)


    def create_callback(self, name_folder):
        earlyStopping = EarlyStopping(
                                    monitor="val_accuracy",
                                    min_delta=0,  # minimium amount of change to count as an improvement
                                    patience=10,  # how many epochs to wait before stopping
                                    restore_best_weights=True)
        reduceLROnPlateau = ReduceLROnPlateau(
                                    monitor="val_accuracy",
                                    factor=0.5,
                                    patience=10),
        modelCheckpoint = ModelCheckpoint(f"{Cfg.MODEL_OUT_PATH}{name_folder}/best_model.hdf5",
                        monitor='val_accuracy',
                        verbose=1,
                        save_weights_only=True,
                        save_freq='epoch',
                        period=5,
                        mode='auto',
                        save_best_only=True)

        return [earlyStopping, reduceLROnPlateau, modelCheckpoint]


    def save_plot(self, history, name, name_folder):
        plt.plot(history.history[name])
        plt.plot(history.history[f'val_{name}'])
        plt.title(f'model {name}')
        plt.ylabel(name)
        plt.xlabel('epoch')
        plt.legend(['train', 'val'], loc='upper left')
        plt.savefig(f'{Cfg.MODEL_OUT_PATH}{name_folder}/model_{name}.png')
        plt.clf()


    def train(self, train, validate, model, callbacks, name_folder):
        history = model.fit(train,
                            batch_size=Cfg.BATCH_SIZE,
                            epochs=Cfg.EPOCHS,
                            validation_data=validate,
                            callbacks=callbacks)

        self.save_plot(history, "loss", name_folder)
        self.save_plot(history, "accuracy", name_folder)
        self.save_plot(history, "top-5-accuracy", name_folder)
        self.save_plot(history, "top-10-accuracy", name_folder)

        return model


    def inference(self, X_test, y_test, model, name_folder):
        loss, acc, top_5_acc, top_10_acc = model.evaluate(X_test, y_test, verbose=2)
        file_save = open(f"{Cfg.MODEL_OUT_PATH}{name_folder}/results.txt", "w")
        result = f"Restored model, accuracy: {100 * acc}, top_5_accuracy: {100 * top_5_acc}, top_10_accuracy: {100 * top_10_acc}\n"
        print(result)
        file_save.writelines(result)

        decoder = lambda x: {v: k for k, v in json.load(open(Cfg.INDEX_MAP_FILE)).items()}.get(x)
        model.evaluate(X_test, y_test)
        for x, y in zip(X_test[:100], y_test[:100]):
            pred_result = f"PRED: {decoder(np.argmax(model.predict(tf.expand_dims(x, axis=0), verbose=0), axis=-1)[0]):<20} – GT: {decoder(y)}"
            print(pred_result)
            file_save.writelines(pred_result+"\n")
        file_save.close()


