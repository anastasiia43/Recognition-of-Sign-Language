import os
import random
import math
import numpy as np
import json
import matplotlib.pyplot as plt
import sklearn
import pandas as pd
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import dataframe_image as dfi

# Import own class
from git.RecognitionofSignLanguage.utils.Cfg import Cfg
from git.RecognitionofSignLanguage.utils.Landmarks import Landmarks as lm

# Import Loss
from git.RecognitionofSignLanguage.loss.ArcFaceLoss import ArcFaceLoss
from git.RecognitionofSignLanguage.loss.CrossentropyLabelSmoothing import CrossentropyLabelSmoothing
from git.RecognitionofSignLanguage.loss.CustomLoss import CustomLoss


class TrainUtils:

    def seed_it_all(self, seed=Cfg.SEED):
        os.environ["PYTHONHASHSEED"] = str(seed)
        random.seed(seed)
        np.random.seed(seed)
        tf.random.set_seed(seed)

    def lrfn(self, current_step, num_warmup_steps, lr_max, num_cycles=0.50, num_training_steps=100):

        if current_step < num_warmup_steps:
            if WARMUP_METHOD == 'log':
                return lr_max * 0.10 ** (num_warmup_steps - current_step)
            else:
                return lr_max * 2 ** -(num_warmup_steps - current_step)
        else:
            progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))

            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress))) * lr_max

    def create_callback(self):
        earlyStopping = EarlyStopping(
            monitor="val_accuracy",
            min_delta=0,  # minimium amount of change to count as an improvement
            patience=15,  # how many epochs to wait before stopping
            restore_best_weights=True)
        reduceLROnPlateau = ReduceLROnPlateau(
            monitor="val_accuracy",
            factor=0.5,
            patience=10),
        modelCheckpoint = ModelCheckpoint(f"{Cfg.MODEL_OUT_PATH}/best_model.hdf5",
                                          monitor='val_accuracy',
                                          verbose=1,
                                          save_weights_only=True,
                                          save_freq='epoch',
                                          period=8,
                                          mode='auto',
                                          save_best_only=True)

        return [earlyStopping, reduceLROnPlateau, modelCheckpoint]

    def save_plot(self, history, name):
        plt.plot(history.history[name])
        plt.plot(history.history[f'val_{name}'])
        plt.title(f'model {name}')
        plt.ylabel(name)
        plt.xlabel('epoch')
        plt.legend(['train', 'val'], loc='upper left')
        plt.savefig(f'{Cfg.MODEL_OUT_PATH}/model_{name}.png')
        plt.clf()
        print(f'Plots saved as {Cfg.MODEL_OUT_PATH}/model_{name}.png')

    def prepare_for_embedding(self, x, MEAN_STD):
        x = tf.slice(x, [0, 0, 0, 0], [-1, Cfg.INPUT_SIZE, lm.N_COLS, Cfg.N_DIMS])
        # LIPS
        lips = tf.slice(x, [0, 0, lm.LIPS_START, 0], [-1, Cfg.INPUT_SIZE, 40, Cfg.N_DIMS])
        lips = tf.where(
            tf.math.equal(lips, 0.0),
            0.0,
            (lips - MEAN_STD['lips_mean']) / MEAN_STD['lips_std'],
        )
        # LEFT HAND
        left_hand = tf.slice(x, [0, 0, 40, 0], [-1, Cfg.INPUT_SIZE, 21, Cfg.N_DIMS])
        left_hand = tf.where(
            tf.math.equal(left_hand, 0.0),
            0.0,
            (left_hand - MEAN_STD['hand_mean']) / MEAN_STD['hand_std'],
        )
        # POSE
        pose = tf.slice(x, [0, 0, 61, 0], [-1, Cfg.INPUT_SIZE, 5, Cfg.N_DIMS])
        pose = tf.where(
            tf.math.equal(pose, 0.0),
            0.0,
            (pose - MEAN_STD['pose_mean']) / MEAN_STD['pose_std'],
        )

        # Flatten
        lips = tf.reshape(lips, [-1, Cfg.INPUT_SIZE, 40 * Cfg.N_DIMS])
        left_hand = tf.reshape(left_hand, [-1, Cfg.INPUT_SIZE, 21 * Cfg.N_DIMS])
        pose = tf.reshape(pose, [-1, Cfg.INPUT_SIZE, 5 * Cfg.N_DIMS])

        return lips, left_hand, pose

    def compice_model(self, config_loss, model):
        # Choose loss
        if config_loss['loss'] == "CustomLoss":
            loss = CustomLoss(config_loss)
        elif config_loss['loss'] == "ArcFaceLoss":
            loss = ArcFaceLoss()
        else:
            loss = CrossentropyLabelSmoothing()

        model.compile(
            loss=loss,
            optimizer=tf.keras.optimizers.Adam(learning_rate=Cfg.LR),
            metrics=["accuracy",
                     tf.keras.metrics.SparseTopKCategoricalAccuracy(k=5, name="top-5-accuracy"),
                     tf.keras.metrics.SparseTopKCategoricalAccuracy(k=10, name="top-10-accuracy")
                     ],
        )

        return model

    def train_and_inference(self, model, X_train, X_val, X_test, y_train, y_val, y_test):
        # Create Folder
        if not os.path.exists(f'{Cfg.MODEL_OUT_PATH}'):
            os.makedirs(f'{Cfg.MODEL_OUT_PATH}')

        callbacks = self.create_callback()

        # change data for save in ram data
        train = tf.data.Dataset.from_tensor_slices((X_train, y_train)).shuffle(4 * Cfg.BATCH_SIZE).batch(Cfg.BATCH_SIZE)
        del X_train, y_train

        validate = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(Cfg.BATCH_SIZE)
        del X_val, y_val

        model = self.train(train, validate, model, callbacks)
        self.inference(X_test, y_test, model)

    def train(self, train, validate, model, callbacks):
        history = model.fit(train,
                            batch_size=Cfg.BATCH_SIZE,
                            epochs=Cfg.EPOCHS,
                            validation_data=validate,
                            callbacks=callbacks)

        self.save_plot(history, "loss")
        self.save_plot(history, "accuracy")
        self.save_plot(history, "top-5-accuracy")
        self.save_plot(history, "top-10-accuracy")

        return model

    def inference(self, X_test, y_test, model):
        _, acc, top_5_acc, top_10_acc = model.evaluate(X_test, y_test, verbose=2)
        file_save = open(f"{Cfg.MODEL_OUT_PATH}/results.txt", "w")
        result = f"Restored model, accuracy: {100 * acc}, top_5_accuracy: {100 * top_5_acc}, top_10_accuracy: {100 * top_10_acc}\n"
        print(result)
        file_save.writelines(result)

        try:
            decoder = lambda x: {v: k for k, v in json.load(open(Cfg.INDEX_MAP_FILE)).items()}.get(x)
            model.evaluate(X_test, y_test)
            for x, y in zip(X_test, y_test[:100]):
                pred_result = f"PRED: {decoder(np.argmax(model.predict(tf.expand_dims(x, axis=0), verbose=0), axis=-1)[0]):<20} – GT: {decoder(y)}"
                print(pred_result)
                file_save.writelines(pred_result + "\n")
            file_save.close()
        except:
            file_save.close()

    def print_classification_report(self, X_test, y_test, model, name_folder):
        train = pd.read_csv(f'{Cfg.SAVE_PATH}train.csv')

        # for 22 frames
        train = train.drop(train.index[[13542, 93042]])
        train = train.reset_index(drop=True)

        SIGN2ORD = train[['sign', 'sign_ord']].set_index('sign').squeeze().to_dict()
        ORD2SIGN = train[['sign_ord', 'sign']].set_index('sign_ord').squeeze().to_dict()
        labels = [ORD2SIGN.get(i).replace(' ', '_') for i in range(Cfg.NUM_CLASSES)]

        y_test_pred = model.predict(X_test,
                                    verbose=2).argmax(
            axis=1)

        # Classification report for all signs
        classification_report = sklearn.metrics.classification_report(
            y_test,
            y_test_pred,
            target_names=labels,
            output_dict=True,
        )
        # Round Data for better readability
        classification_report = pd.DataFrame(classification_report).T
        classification_report = classification_report.round(2)
        classification_report = classification_report.astype({
            'support': np.uint16,
        })
        # Add signs
        classification_report['sign'] = [e if e in SIGN2ORD else -1 for e in classification_report.index]
        classification_report['sign_ord'] = classification_report['sign'].apply(SIGN2ORD.get).fillna(-1).astype(
            np.int16)
        # Sort on F1-score
        classification_report = pd.concat((
            classification_report.head(Cfg.NUM_CLASSES).sort_values('f1-score', ascending=False),
            classification_report.tail(3),
        ))

        # Створення таблиці Matplotlib
        fig, ax = plt.subplots(figsize=(10, 8))
        table_data = classification_report.values
        table_columns = classification_report.columns
        ax.axis('tight')
        ax.axis('off')
        ax.table(cellText=table_data, colLabels=table_columns, cellLoc='center', loc='center')
        plt.tight_layout()

        # Збереження графічного звіту у форматі PNG
        report_path = f'{Cfg.MODEL_OUT_PATH}{name_folder}/classification_report.png'
        plt.savefig(report_path, bbox_inches='tight')
        print(f'Classification report saved as {report_path}')
        plt.close(fig)
