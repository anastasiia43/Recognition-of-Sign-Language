import tensorflow as tf
from git.RecognitionofSignLanguage.utils.Cfg import Cfg

from git.RecognitionofSignLanguage.loss.CrossentropyLabelSmoothing import CrossentropyLabelSmoothing
from git.RecognitionofSignLanguage.loss.ArcFaceLoss import ArcFaceLoss


class CustomLoss(tf.keras.losses.Loss):
    def __init__(self, config):
        super(CustomLoss, self).__init__()
        self.arcface_loss = ArcFaceLoss(s=config['s'], m=config['m'])
        self.scce_loss = CrossentropyLabelSmoothing(label_smoothing=config['smooth'])
        self.WA = config['WA']
    def call(self, y_true, y_pred):
        # Обчислити значення функції втрат для ArcFaceLoss та SCCE
        arcface_loss_value = self.arcface_loss(y_true, y_pred)
        scce_loss_value = self.scce_loss(y_true, y_pred)

        # Змішати функції втрат згідно з вагами
        mixed_loss = self.WA * arcface_loss_value + (1 - self.WA) * scce_loss_value
        return mixed_loss