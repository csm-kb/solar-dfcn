# (4/12/22) bug with TF 2.8 requires us to do this load trick
import typing
from typing import List
import tensorflow as tf
import numpy as np
from tensorflow import keras
if typing.TYPE_CHECKING:
    from keras.api._v2 import keras
from keras import datasets, layers, models
# ///

class MNISTModel(models.Model):
    def __init__(self):
        super(MNISTModel, self).__init__()
        self.r_layers = []
        # conv layers
        self.r_layers += [
            layers.Conv2D(32, 3, activation='relu')
        ]
        # dense layers
        self.r_layers += [
            layers.Flatten(),
            layers.Dense(128, activation='relu'),
            layers.Dense(10)
        ]
    
    def call(self, x):
        for layer in self.r_layers:
            x = layer(x)
        return x


class SolarFCNModel(models.Model):
    def __init__(self, dropout_rate=0.2):
        super(MNISTModel, self).__init__()

        self.r_blocks: List[layers.Layer] = [[
            # variable initial input
            layers.Input(shape=(None,None,3))
            ]]
        # conv blocks
            # block 1
        self.r_blocks += [[
            # > ingestion conv
            layers.Conv2D(64, 7, 2),
            layers.Dropout(dropout_rate),
            layers.BatchNormalization(),
            layers.Activation(tf.nn.relu),
            # > max pool
            layers.MaxPooling2D()
            ]]
            # block 2
        self.r_blocks += [[
            # > 1
            layers.Conv2D(64, 3, 1),
            layers.Dropout(dropout_rate),
            layers.BatchNormalization(),
            layers.Activation(tf.nn.relu),
            # > 2
            layers.Conv2D(64, 3, 1),
            layers.Dropout(dropout_rate),
            layers.BatchNormalization(),
            layers.Activation(tf.nn.relu)
            ]]
            # block 3
        self.r_blocks += [[
            # > 1
            layers.Conv2D(128, 3, 2),
            layers.Dropout(dropout_rate),
            layers.BatchNormalization(),
            layers.Activation(tf.nn.relu),
            # > 2
            layers.Conv2D(128, 3, 1),
            layers.Dropout(dropout_rate),
            layers.BatchNormalization(),
            layers.Activation(tf.nn.relu),
            # --> average pool at end of last conv block
            layers.AveragePooling2D()
            ]]
        # fully connected (FC) block [idx 4]
            # > 1
        self.r_blocks += [[
            layers.Conv2D(64, 1, 1),
            layers.Dropout(dropout_rate),
            layers.BatchNormalization(),
            layers.Activation(tf.nn.relu),
            # > 2
            layers.Conv2D(64, 1, 1),
            layers.Dropout(dropout_rate),
            layers.BatchNormalization(),
            layers.Activation(tf.nn.relu)
            ]]
        # output FC block [idx 5]
            # solar panel roof ? 1 : 0 (length = 2)
        self.r_blocks += [[
            layers.Conv2D(1, 1, 1),
            layers.Dropout(dropout_rate),
            layers.BatchNormalization(),
            layers.Activation(tf.nn.sigmoid)
            ]]
    
    def call(self, x: np.ndarray):
        # let's do some fancy shortcuts, like ResNet

        # input block + conv1
        for i in range(2):
            for j in range(len(self.r_blocks[i])):
                x = self.r_blocks[i][j](x)
        x_int = x.copy()

        # do conv2, shortcut-add prev
        for j in range(len(self.r_blocks[2])):
            x = self.r_blocks[2][j](x)
        x = layers.Add()([x, x_int])
        x_int = x.copy()

        # do conv3, shortcut-add prev
        for j in range(len(self.r_blocks[3])):
            x = self.r_blocks[3][j](x)
        x = layers.Add()([x, x_int])
        x_int = None

        # do FC to finish
        for i in range(4,len(self.r_blocks)):
            for j in range(len(self.r_blocks[i])):
                x = self.r_blocks[i][j](x)
        
        return x

def Gen_SolarFCNModel(dropout_rate=0.2, output_bias=None,**kwargs) -> models.Model:
    if output_bias is not None:
        output_bias = keras.initializers.Constant(output_bias)
    model = models.Sequential(**kwargs)

    # this doesn't use subclassing, so it can't do shortcuts :<

    # variable initial input
    _blocks = [[layers.Input(shape=(None,None,3))]]
    # conv blocks
        # block 1
    _blocks += [[
        # > ingestion conv
        layers.Conv2D(64, 7, 2),
        layers.BatchNormalization(),
        layers.Activation(tf.nn.relu),
        # > max pool
        layers.MaxPooling2D(),
        layers.Dropout(dropout_rate)
        ]]
        # block 2
    _blocks += [[
        # > 1
        layers.Conv2D(64, 3, 1),
        layers.BatchNormalization(),
        layers.Activation(tf.nn.relu),
        # > 2
        layers.Conv2D(64, 3, 1),
        layers.BatchNormalization(),
        layers.Activation(tf.nn.relu),
        # > 3
        layers.Conv2D(64, 3, 1),
        layers.BatchNormalization(),
        layers.Activation(tf.nn.relu),
        # > 4
        layers.Conv2D(64, 3, 1),
        layers.BatchNormalization(),
        layers.Activation(tf.nn.relu)
        ]]
        # block 3
    _blocks += [[
        # > 1
        layers.Conv2D(128, 3, 2),
        layers.BatchNormalization(),
        layers.Activation(tf.nn.relu),
        # > 2
        layers.Conv2D(128, 3, 1),
        layers.BatchNormalization(),
        layers.Activation(tf.nn.relu),
        # > 3
        layers.Conv2D(128, 3, 1),
        layers.BatchNormalization(),
        layers.Activation(tf.nn.relu),
        # > 4
        layers.Conv2D(128, 3, 1),
        layers.BatchNormalization(),
        layers.Activation(tf.nn.relu),
        # --> average pool at end of last conv block
        layers.AveragePooling2D(),
        layers.Dropout(dropout_rate)
        ]]
    # fully connected (FC) block [idx 4]
        # > 1
    _blocks += [[
        layers.Conv2D(64, 1, 1),
        layers.BatchNormalization(),
        layers.Activation(tf.nn.relu),
        # > 2
        layers.Conv2D(64, 1, 1),
        layers.BatchNormalization(),
        layers.Activation(tf.nn.relu)
        ]]
    # output FC block [idx 5]
        # solar panel roof ? 1 : 0 (length = 2 classes)
    _blocks += [[
        layers.Conv2D(2, 1, 1, bias_initializer=output_bias),
        layers.BatchNormalization(),
        # one more pooling prior to activation to get down to (x,y)
        layers.GlobalMaxPooling2D(),
        layers.Activation(tf.nn.softmax)
        ]]

    for block in _blocks:
        for layer in block:
            # print(layer)
            model.add(layer)

    model.summary()
    return model