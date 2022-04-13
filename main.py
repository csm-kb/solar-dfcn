import gc
import sys
import os
import argparse
import tensorflow as tf

# (4/12/22) bug with TF 2.8 requires us to do this load trick
import typing
from tensorflow import keras
if typing.TYPE_CHECKING:
    from keras.api._v2 import keras
from keras import datasets, layers, models
from keras.preprocessing.image import ImageDataGenerator
# ///
from TFModels import Gen_SolarFCNModel
import matplotlib.pyplot as plt
from PIL import Image as PILImage
from matplotlib import image as mpimage
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from typing import Any, Tuple

AUTO = tf.data.experimental.AUTOTUNE

DATA_DIR = os.path.join('.','data')

def show_float32_image(img: np.ndarray) -> None:
    PILImage.fromarray((img*255/np.max(img)).astype(np.uint8)).show()

def prepare_dataset() -> Tuple[np.ndarray,np.ndarray]:
    t_house_df = pd.DataFrame()

    # load house1 (first house)
    path_csv = os.path.join(DATA_DIR,'house1','house1','house1.csv')
    house_df = pd.read_csv(path_csv)
    print(f'Initial house 1 DF shape: {house_df.shape}')

    # get house images
    path_imgs = os.path.join(DATA_DIR,'house1','house1','roof')
    path_list_imgs = [f for f in os.listdir(path_imgs) if os.path.isfile(os.path.join(path_imgs, f))]
    _path_noext = [os.path.splitext(f)[0] for f in path_list_imgs]
    _t_img_df = pd.DataFrame({'id': np.asarray(_path_noext).astype(np.int64)})
    
    # filter only by the IDs we have images for
    house_df = house_df.merge(_t_img_df, how='inner', on=['id'])
    # filter path list only by the IDs now in the house_df
    path_list_imgs = np.delete(
        path_list_imgs,
        np.where(~_t_img_df['id'].isin(house_df['id']))
    )
    print(f'Existing house 1 DF shape: {house_df.shape}')
    _t_img_df = None # clean up
    # append to total tracker DF
    t_house_df = pd.concat([t_house_df, house_df], ignore_index=True)

    # load images to form base X and y
    x_imgs = [os.path.join(path_imgs,i) for i in path_list_imgs]
    y_base = house_df[['label']].copy().to_numpy().flatten()
    house_df = None

    # load the rest of the houses
    for i in range(2,11):
        path_csv = os.path.join(DATA_DIR,f'house{i}',f'house{i}',f'house{i}.csv')
        house_df = pd.read_csv(path_csv)
        print(f'Initial house {i} DF shape: {house_df.shape}')

        # get house images
        path_imgs = os.path.join(DATA_DIR,f'house{i}',f'house{i}','roof')
        path_list_imgs = [f for f in os.listdir(path_imgs) if os.path.isfile(os.path.join(path_imgs, f))]
        _path_noext = [os.path.splitext(f)[0] for f in path_list_imgs]
        _path_ids_int = np.asarray(_path_noext).astype(np.int64)
        _t_img_df = pd.DataFrame({'id': _path_ids_int})
        
        # filter only by the IDs we have images for
        house_df = house_df.merge(_t_img_df, how='inner', on=['id'])
        # filter path list only by the IDs now in the house_df
        path_list_imgs = np.delete(
            path_list_imgs,
            np.where(~_t_img_df['id'].isin(house_df['id']))
        )
        print(f'Existing house {i} DF shape: {house_df.shape}')
        _t_img_df = None # clean up
        # append to total tracker DF
        t_house_df = pd.concat([t_house_df, house_df], ignore_index=True)

        # load images to form base X and y
        x_imgs += [os.path.join(path_imgs,i) for i in path_list_imgs]
        y_base = np.concatenate([y_base, house_df[['label']].copy().to_numpy().flatten()])
        print(f'\tx_imgs shape: ({len(x_imgs)},)')
        print(f'\ty_base shape: {y_base.shape}')
        house_df = None

    x_imgs = np.asarray(x_imgs)
    print("[*] Dataset prep complete!")
    return (x_imgs, y_base)

def pd_tt_split(df: pd.DataFrame, train_split=0.8):
    # specify seed to always have the same split distribution between runs
    train_df = df.sample(frac=train_split, random_state=12)
    test_df = df.drop(train_df.index)

    return train_df, test_df

class SolarDataGen(keras.utils.Sequence):
    
    def __init__(self, df, X_col, y_col,
                 batch_size,
                 shuffle=True):
        
        self.df = df.copy()
        self.X_col = X_col
        self.y_col = y_col
        self.batch_size = batch_size
        self.input_size = None
        self.shuffle = shuffle
        
        self.n = len(self.df)
        self.n_labels = 2
    
    def __get_input(self, path):
        # xmin, ymin, w, h = bbox['x'], bbox['y'], bbox['width'], bbox['height']

        image = keras.preprocessing.image.load_img(path)
        image_arr = keras.preprocessing.image.img_to_array(image)

        # note: NO WE DON'T WANT TO SELECTIVELY CROP
        # image_arr = image_arr[ymin:ymin+h, xmin:xmin+w]
        # note: NO WE DON'T WANT TO RESIZE
        # image_arr = tf.image.resize(image_arr,(target_size[0], target_size[1])).numpy()

        return image_arr/255.0

    def __get_output(self, label, num_classes):
        return keras.utils.to_categorical(label, num_classes=num_classes)

    def __get_data(self, batches):
        # generate data containing batch_size samples
        img_batch = batches[self.X_col]
        label_batch = batches[self.y_col]

        X_batch = np.asarray([self.__get_input(x) for x in img_batch])
        y_batch = np.asarray([self.__get_output(y, self.n_labels) for y in label_batch])

        return X_batch, y_batch

    def on_epoch_end(self):
        # we will shuffle the dataset to make some randomness
        if self.shuffle:
            self.df = self.df.sample(frac=1).reset_index(drop=True)
    
    def __getitem__(self, index):
        batches = self.df[index * self.batch_size:(index + 1) * self.batch_size]
        X, y = self.__get_data(batches)        
        return X, y
    
    def __len__(self):
        return self.n // self.batch_size


class CleanGCCallback(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        gc.collect() # collect any dangling garbage post-epoch


# // ## ######## ## // #
# // ## - Core-  ## // #
# // ## ######## ## // #

def main(*args, **kwargs) -> int:
    # prepare args
    EPOCHS = kwargs['epochs']
    BATCH_SIZE = kwargs['batch']

    print(f'[*] {EPOCHS} epochs, batch size of {BATCH_SIZE}')

    # prepare data
    (x_imgs, y_base) = prepare_dataset()

    print(f'final x_imgs shape: {x_imgs.shape}')
    print(f'final y_base shape: {y_base.shape}')

    _t_split_df = pd.DataFrame({'img':x_imgs,'label':y_base})

    train_df, test_df = pd_tt_split(_t_split_df, 0.8)

    print(f'train_df: {train_df.shape}')
    print(f'test_df : {test_df.shape}')

    # WE CAN WRITE A GENERATOR FOR OUR DATAFRAMES LETS GOOOOOOO
    train_gen = SolarDataGen(
        train_df,
        X_col='img',
        y_col='label',
        batch_size=BATCH_SIZE
    )

    test_gen = SolarDataGen(
        test_df,
        X_col='img',
        y_col='label',
        batch_size=BATCH_SIZE
    )

    # print stats about model imbalance
    _t_neg, _t_pos = np.bincount(_t_split_df['label'])
    _t_tot = _t_neg + _t_pos
    print(
        f'Model balance:\n',
        f'\tTotal    : {_t_tot}\n',
        f'\tPositive : {_t_pos}\n',
        f'\tNegative : {_t_neg}'
    )
    # calculate bias of model
    initial_bias = np.log([_t_pos/_t_neg])

    # prepare model, loss/optimizer, and metrics objects
    model = Gen_SolarFCNModel(output_bias=initial_bias)
    loss_obj = keras.losses.CategoricalCrossentropy()
    optimizer_obj = keras.optimizers.Adamax()
    

    print(f"[*] Training model on {_t_tot} images...")
    #///
    model.compile(
        optimizer=optimizer_obj,
        loss=loss_obj,
        metrics=['accuracy'],
    )
    history = model.fit(
        train_gen,
        epochs=EPOCHS,
        validation_data=test_gen,
        callbacks=[CleanGCCallback()]
    )

    print(history.history.keys())
    print("[*] Evaluating training results...")
    # build plot of accuracies and losses over epochs for analysis
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    ax1.plot(history.history['accuracy'], label='accuracy', color='green')
    ax1.plot(history.history['val_accuracy'], label='val_accuracy', color='blue')
    ax1.set_ylim([0,1])
    ax1.set_ylabel('Accuracy')
    ax2.plot(history.history['loss'], label='loss', color='red')
    ax2.plot(history.history['val_loss'], label='val_loss', color='orange')
    ax2.set_ylim([0,1])
    ax2.set_ylabel('Loss')

    ax1.set_xlabel('Epoch')
    ax1.set_xlim([0,EPOCHS])

    plt.legend(loc='lower right')
    plt.show()

    print("[*] Testing model...")
    test_loss, test_acc = model.evaluate(test_gen, verbose=2)

    print(
        f'\nResults:\n'
        f'\tTest loss: {test_loss}\n',
        f'\tTest accu: {test_acc}'
    )

    print("[*] Complete!")
    #///

    #///
    # train_loss = keras.metrics.Mean(name='train_loss')
    # train_accuracy = keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

    # test_loss = keras.metrics.Mean(name='test_loss')
    # test_accuracy = keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')

    # # train and test step functions
    # @tf.function
    # def train_step(images, labels):
    #     with tf.GradientTape() as tape:
    #         predictions = model(images, training=True)
    #         loss = loss_obj(labels, predictions)
    #     gradients = tape.gradient(loss, model.trainable_variables)
    #     optimizer_obj.apply_gradients(zip(gradients, model.trainable_variables))
    #     # metric update
    #     train_loss(loss)
    #     train_accuracy(labels, predictions)

    # @tf.function
    # def test_step(images, labels):
    #     predictions = model(images, training=False)
    #     t_loss = loss_obj(labels, predictions)
    #     # metric update
    #     test_loss(t_loss)
    #     test_accuracy(labels, predictions)

    # # fit the model and test it
    # print("[*] Fitting model...")
    # for epoch in tqdm(range(EPOCHS)):
    #     # reset the metrics at the start of the next epoch
    #     train_loss.reset_states()
    #     train_accuracy.reset_states()
    #     test_loss.reset_states()
    #     test_accuracy.reset_states()

    #     # perform a train step with the given training dataset
    #     # --> can batch load and step so a 300 GB dataset doesn't need to be loaded all at once
    #     for images, labels in train_ds:
    #         train_step(images, labels)

    #     # perform a test step with the given test dataset
    #     for images, labels in test_ds:
    #         test_step(images, labels)

    #     print(
    #         f'\n---------\n'
    #         f'Epoch {epoch + 1}\n'
    #         f'\tLoss:\t\t{train_loss.result()}\n'
    #         f'\tAccuracy:\t{train_accuracy.result() * 100}%\n'
    #         f'\tTest Loss:\t{test_loss.result()}\n'
    #         f'\tTest Accuracy:\t{test_accuracy.result() * 100}%\n'
    #         f'---------'
    #     )

    # print("[*] Complete! See final epoch results for loss/accuracy.")
    #///

    return 0

if __name__ == "__main__":
    _desc = """
    Runner for a deep convolutional network model on the geolocated solar panel dataset.
    """
    parser = argparse.ArgumentParser(description=_desc)
    
    parser.add_argument('--epochs', metavar='E', dest='epochs', type=int, nargs='?',
        default=5, help="number of epochs to train the model for (default: 5) -- scales with GPU throughput")
    parser.add_argument('--batch', metavar='B', dest='batch', type=int, nargs='?',
        default=8, help="test/train batch size (default: 8) -- scales with GPU memory availability")

    args = parser.parse_args()
    ret = main(*args._get_args(), **dict(args._get_kwargs()))
    sys.exit(ret)