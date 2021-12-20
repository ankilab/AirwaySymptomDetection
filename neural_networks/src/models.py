import sys
sys.path.insert(0, '../')
from .ResNet import resnet
import efficientnet.tfkeras as efn
import tensorflow as tf

"""
Methods take number of output classes and the params-object, to create TensorFlow models.
"""


def get_resnet18(nb_classes, params):
    model_resnet18 = resnet.resnet_18(num_classes=nb_classes)
    model_resnet18.build(input_shape=(None, params.n_mels, 64, 1))
    return model_resnet18


def get_resnet34(nb_classes, params):
    model_resnet34 = resnet.resnet_34(num_classes=nb_classes)
    model_resnet34.build(input_shape=(None, params.n_mels, 64, 1))
    return model_resnet34


def get_efficientnetb0(nb_classes, params):
    model_efn = efn.EfficientNetB0(input_shape=(params.n_mels, 64, 1),
                                   include_top=True,
                                   weights=None,
                                   classes=nb_classes)
    return model_efn


def get_mobilenetv2(nb_classes, params):
    model_mobilenet = tf.keras.applications.MobileNetV2(input_shape=(params.n_mels, 64, 1),
                                                        include_top=True,
                                                        weights=None,
                                                        classes=nb_classes)
    return model_mobilenet


def get_rnn_amoh(nb_classes, params):
    model = tf.keras.Sequential([
        tf.keras.layers.GRU(128, input_shape=(params.n_mels, 64), return_sequences=True),
        tf.keras.layers.GRU(64, return_sequences=True),
        tf.keras.layers.GRU(32, return_sequences=True),
        tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(64)),
        tf.keras.layers.LSTM(64),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dense(nb_classes, activation='softmax')
    ])
    return model


def get_rnn_basic(nb_classes, params):
    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(128, input_shape=(params.n_mels, 64), return_sequences=True),
        tf.keras.layers.LSTM(64, return_sequences=True),
        tf.keras.layers.LSTM(32),
        tf.keras.layers.Dense(nb_classes, activation='softmax')
        ])
    return model
