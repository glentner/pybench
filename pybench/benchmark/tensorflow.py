# SPDX-FileCopyrightText: 2021 Geoffrey Lentner
# SPDX-License-Identifier: Apache-2.0

"""TensorFlow benchmarks."""


# type annotations
from __future__ import annotations
from typing import Tuple

# external libs
import numpy as np

# internal libs
from ..core import Benchmark, BenchmarkError

# public interface
__all__ = ['MnistMLP', 'MnistCNN', 'Resnet50', ]


class MnistMLP(Benchmark):
    """Train simple categorical MLP on MNIST data."""

    name = 'tensorflow.mnist-mlp'
    annotation = '(batchsize: int)'

    loss = 'categorical_crossentropy'
    optimizer = 'rmsprop'

    x_train: np.ndarray
    y_train: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray

    model: 'tensorflow.keras.models.Sequential'
    batchsize: int

    def setup(self, batchsize: int) -> None:
        """Initialize data and model."""
        try:
            self.batchsize = batchsize
            self.prepare_data()
            self.create_model()
        except Exception as error:
            raise BenchmarkError(f'Args for \'{self.name}\': {error}') from error

    def prepare_data(self) -> None:
        """Prepare MNIST dataset for training."""

        # NOTE: import within function context to avoid heavy lift on startup
        from tensorflow import keras

        x_train = np.random.rand(60_000, 28, 28)
        y_train = np.random.rand(60_000)
        x_test  = np.random.rand(10_000, 28, 28)
        y_test  = np.random.rand(10_000)

        x_train = x_train.reshape((60_000, 784)).astype('float32') / 255
        x_test  = x_test.reshape((10_000, 784)).astype('float32') / 255

        num_classes = 10
        y_train = keras.utils.to_categorical(y_train, num_classes)
        y_test  = keras.utils.to_categorical(y_test,  num_classes)

        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test

    def create_model(self) -> None:
        """Simple 3-layer MLP."""

        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import Dense, Dropout

        model = Sequential()
        model.add(Dense(512, activation='relu', input_shape=(784,)))
        model.add(Dropout(0.2))
        model.add(Dense(512, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(10, activation='softmax'))

        self.model = model
        self.model.compile(loss=self.loss, optimizer=self.optimizer, metrics=['accuracy'])

    def task(self) -> None:
        """Train a single epoch of the model."""
        self.model.fit(self.x_train, self.y_train, batch_size=self.batchsize, epochs=1,
                       verbose=0, validation_data=(self.x_test, self.y_test))


class MnistCNN(Benchmark):
    """Train basic categorical CNN on MNIST data."""

    name = 'tensorflow.mnist-cnn'
    annotation = '(batchsize: int)'

    loss = 'categorical_crossentropy'
    optimizer = 'adadelta'

    x_train: np.ndarray
    y_train: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray

    model: 'tensorflow.keras.models.Sequential'
    batchsize: int

    def setup(self, batchsize: int) -> None:
        """Initialize data and model."""
        try:
            self.batchsize = batchsize
            self.prepare_data()
            self.create_model()
        except Exception as error:
            raise BenchmarkError(f'Args for \'{self.name}\': {error}') from error

    def prepare_data(self) -> None:
        """Prepare MNIST dataset for training."""

        # NOTE: import within function context to avoid heavy lift on startup
        from tensorflow import keras

        num_classes = 10
        img_rows, img_cols = 28, 28

        x_train = np.random.rand(60_000, 28, 28)
        y_train = np.random.rand(60_000)
        x_test  = np.random.rand(10_000, 28, 28)
        y_test  = np.random.rand(10_000)

        if keras.backend.image_data_format() == 'channels_first':
            x_train = x_train.reshape((x_train.shape[0], 1, img_rows, img_cols))
            x_test = x_test.reshape((x_test.shape[0], 1, img_rows, img_cols))
        else:
            x_train = x_train.reshape((x_train.shape[0], img_rows, img_cols, 1))
            x_test = x_test.reshape((x_test.shape[0], img_rows, img_cols, 1))

        x_train = x_train.astype('float32') / 255
        x_test = x_test.astype('float32') / 255

        # convert class vectors to binary class matrices
        y_train = keras.utils.to_categorical(y_train, num_classes)
        y_test = keras.utils.to_categorical(y_test, num_classes)

        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test

    def create_model(self) -> None:
        """Basic convolutional network."""

        from tensorflow.keras.backend import image_data_format
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

        img_rows, img_cols = 28, 28
        if image_data_format() == 'channels_first':
            input_shape = (1, img_rows, img_cols)
        else:
            input_shape = (img_rows, img_cols, 1)

        model = Sequential()
        model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(10, activation='softmax'))

        self.model = model
        self.model.compile(loss=self.loss, optimizer=self.optimizer, metrics=['accuracy'])

    def task(self) -> None:
        """Train a single epoch of the model."""
        self.model.fit(self.x_train, self.y_train, batch_size=self.batchsize, epochs=1,
                       verbose=0, validation_data=(self.x_test, self.y_test))


class Resnet50(Benchmark):
    """Train deep convolutional network (Resnet50)."""

    name = 'tensorflow.resnet50'
    annotation = '(batchsize: int)'

    loss = 'categorical_crossentropy'
    optimizer = 'rmsprop'

    classes = 1_000
    rows = 224
    cols = 224
    channels = 3  # RGB

    n_train = 1_000
    n_test = 300

    x_train: np.ndarray
    y_train: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray

    model: 'tensorflow.keras.models.Sequential'
    batchsize: int

    def setup(self, batchsize: int) -> None:
        """Initialize data and model."""
        self.batchsize = batchsize
        self.prepare_data()
        self.create_model()

    def prepare_data(self) -> None:
        """Prepare MNIST dataset for training."""

        # NOTE: import within function context to avoid heavy lift on startup
        from tensorflow import keras

        def gen_images(count: int, n: int, m: int, c: int) -> Tuple[np.ndarray, np.ndarray]:
            """Create `count` number of images with `n` rows, `m` columns, and `c` channels."""
            X = np.round(np.random.rand(count, n, m, c) * 255).astype('float32') / 255  # noqa: upper-case
            y = np.round(np.random.rand(count) * self.classes - 1).astype('float32')
            y = keras.utils.to_categorical(y, num_classes=self.classes)
            return X, y

        shape = (self.rows, self.cols, self.channels)
        self.x_train, self.y_train = gen_images(self.n_train, *shape)
        self.x_test, self.y_test = gen_images(self.n_test, *shape)

    def create_model(self) -> None:
        """ResNet50 defined within Keras."""
        from tensorflow.keras.applications.resnet50 import ResNet50 as Model
        self.model = Model(weights=None)
        self.model.compile(loss=self.loss, optimizer=self.optimizer, metrics=['accuracy'])

    def task(self) -> None:
        """Train a single epoch of the model."""
        self.model.fit(self.x_train, self.y_train, batch_size=self.batchsize, epochs=1,
                       verbose=0, validation_data=(self.x_test, self.y_test))
