import sys
import argparse
import tensorflow as tf

# (4/12/22) bug with TF 2.8 requires us to do this load trick
import typing
from tensorflow import keras
if typing.TYPE_CHECKING:
    from keras.api._v2 import keras
from keras import datasets, layers, models
# ///
from TFModels import MNISTModel
import matplotlib.pyplot as plt
import numpy as np
from tqdm.auto import tqdm

from typing import Any, Tuple

AUTO = tf.data.experimental.AUTOTUNE


def prepare_dataset() -> Tuple[Tuple[Any,Any],Tuple[Any,Any]]:
    (x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()
    print("[*] Processing train_imgs:")
    x_train = x_train / 255.0
    print("[*] Processing test_imgs:")
    x_test = x_test / 255.0

    # add a channels dimension
    x_train = x_train[..., tf.newaxis].astype("float32")
    x_test = x_test[..., tf.newaxis].astype("float32")

    print("[*] Processing complete!")
    return (x_train, y_train), (x_test, y_test)

def main(*args, **kwargs) -> int:
    # prepare args
    EPOCHS = kwargs['epochs']
    BATCH_SIZE = kwargs['batch']

    # prepare data
    (x_train, y_train), (x_test, y_test) = prepare_dataset()
    train_ds = tf.data.Dataset.from_tensor_slices(
        (x_train, y_train)).shuffle(10000).batch(BATCH_SIZE)
    test_ds = tf.data.Dataset.from_tensor_slices(
        (x_test, y_test)).batch(BATCH_SIZE)

    # prepare model, loss/optimizer, and metrics objects
    model = MNISTModel()
    model.summary()
    loss_obj = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    optimizer = keras.optimizers.Adam()

    train_loss = keras.metrics.Mean(name='train_loss')
    train_accuracy = keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

    test_loss = keras.metrics.Mean(name='test_loss')
    test_accuracy = keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')

    # train and test step functions
    @tf.function
    def train_step(images, labels):
        with tf.GradientTape() as tape:
            predictions = model(images, training=True)
            loss = loss_obj(labels, predictions)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        # metric update
        train_loss(loss)
        train_accuracy(labels, predictions)

    @tf.function
    def test_step(images, labels):
        predictions = model(images, training=False)
        t_loss = loss_obj(labels, predictions)
        # metric update
        test_loss(t_loss)
        test_accuracy(labels, predictions)

    # fit the model and test it
    print("[*] Fitting model...")
    for epoch in tqdm(range(EPOCHS)):
        # reset the metrics at the start of the next epoch
        train_loss.reset_states()
        train_accuracy.reset_states()
        test_loss.reset_states()
        test_accuracy.reset_states()

        # perform a train step with the given training dataset
        # --> can batch load and step so a 300 GB dataset doesn't need to be loaded all at once
        for images, labels in train_ds:
            train_step(images, labels)

        # perform a test step with the given test dataset
        for images, labels in test_ds:
            test_step(images, labels)

        print(
            f'\n---------\n'
            f'Epoch {epoch + 1}\n'
            f'\tLoss:\t\t{train_loss.result()}\n'
            f'\tAccuracy:\t{train_accuracy.result() * 100}%\n'
            f'\tTest Loss:\t{test_loss.result()}\n'
            f'\tTest Accuracy:\t{test_accuracy.result() * 100}%\n'
            f'---------'
        )

    print("[*] Complete! See final epoch results for loss/accuracy.")

    return 0

if __name__ == "__main__":
    _desc = """
    Runner for a deep convolutional network model on the geolocated solar panel dataset.
    """
    parser = argparse.ArgumentParser(description=_desc)
    
    parser.add_argument('--epochs', metavar='E', dest='epochs', type=int, nargs='?',
        default=5, help="number of epochs to train the model for (default: 5)")
    parser.add_argument('--batch', metavar='B', dest='batch', type=int, nargs='?',
        default=32, help="test/train batch size (default: 32)")

    args = parser.parse_args()
    ret = main(*args._get_args(), **dict(args._get_kwargs()))
    sys.exit(ret)