#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
# ECE:5995 Deep Learning Final Project
> Alexander Powers     

## An Exploration of Multi-Task Learning
This project explores different network architectures that leverage weight sharing to improve performance on multiple tasks.
    
### The Problem (CIFAR100)
The CIFAR100 dataset consists of RGB images, fine labels(100 classes), and coarse labels(20 classes). Each fine label class is a proper subset of a coarse label class (i.e. one fine label can't have two coarse labels and vice versa).

### Architectures to be trained
#### 1) Independent networks (the control architecture)
```text
input_image --> conv_layers --> fc_layers --> fine_label       
input_image --> conv_layers --> fc_layers --> coarse_label
```       
#### 2) Hard parameter sharing in convolutional layers
```text
                           /--> fc_layers --> fine_label
input_image --> conv_layers      
                           \--> fc_layers --> coarse_label 
```
#### 3) Using coarse label output as weights
```text
input_image   ---->   conv_layers   ---->   fc_layers_1
                                                      \                
                                                    concat -> fc_layers_2  -> fine_label
                                                      /
input_image -> conv_layers -> fc_layers -> coarse_label
``` 
#### 4) Using fine label output as weights
```text
input_image   ---->   conv_layers   ---->   fc_layers_1
                                                      \                
                                                    concat -> fc_layers_2  -> coarse_label
                                                      /
input_image -> conv_layers -> fc_layers -> fine_label
``` 
#### 5) Combination of 2 & 3
```text
                         / -------------> fc_layers_1
                        /                           \
input_image -> conv_layers                        concat -> fc_layers_2 -> fine_label
                        \                           /
                         \-> fc_layers -> coarse_label 
```
#### 6) Combination of 2 & 4
```text
                         / -------------> fc_layers_1
                        /                           \
input_image -> conv_layers                        concat -> fc_layers_2 -> coarse_label
                        \                           /
                         \-> fc_layers -> fine_label 
```
"""

# Commented out IPython magic to ensure Python compatibility.
import numpy as np
from matplotlib import pyplot as plt

# %matplotlib inline
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from tensorflow.keras.datasets import cifar100
import tensorflow as tf
import pickle

# global constants
INPUT_SHAPE = (32, 32, 3)
FINE_CLASS_NAMES = [
    "apple",
    "aquarium_fish",
    "baby",
    "bear",
    "beaver",
    "bed",
    "bee",
    "beetle",
    "bicycle",
    "bottle",
    "bowl",
    "boy",
    "bridge",
    "bus",
    "butterfly",
    "camel",
    "can",
    "castle",
    "caterpillar",
    "cattle",
    "chair",
    "chimpanzee",
    "clock",
    "cloud",
    "cockroach",
    "couch",
    "crab",
    "crocodile",
    "cup",
    "dinosaur",
    "dolphin",
    "elephant",
    "flatfish",
    "forest",
    "fox",
    "girl",
    "hamster",
    "house",
    "kangaroo",
    "keyboard",
    "lamp",
    "lawn_mower",
    "leopard",
    "lion",
    "lizard",
    "lobster",
    "man",
    "maple_tree",
    "motorcycle",
    "mountain",
    "mouse",
    "mushroom",
    "oak_tree",
    "orange",
    "orchid",
    "otter",
    "palm_tree",
    "pear",
    "pickup_truck",
    "pine_tree",
    "plain",
    "plate",
    "poppy",
    "porcupine",
    "possum",
    "rabbit",
    "raccoon",
    "ray",
    "road",
    "rocket",
    "rose",
    "sea",
    "seal",
    "shark",
    "shrew",
    "skunk",
    "skyscraper",
    "snail",
    "snake",
    "spider",
    "squirrel",
    "streetcar",
    "sunflower",
    "sweet_pepper",
    "table",
    "tank",
    "telephone",
    "television",
    "tiger",
    "tractor",
    "train",
    "trout",
    "tulip",
    "turtle",
    "wardrobe",
    "whale",
    "willow_tree",
    "wolf",
    "woman",
    "worm",
]
COARSE_CLASS_NAMES = [
    "aquatic_mammals",
    "fish",
    "flowers",
    "food_containers",
    "fruit_and_vegetables",
    "household_electrical_devices",
    "household_furniture",
    "insects",
    "large_carnivores",
    "large_man-made_outdoor_things",
    "large_natural_outdoor_scenes",
    "large_omnivores_and_herbivores",
    "medium_mammals",
    "non-insect_invertebrates",
    "people",
    "reptiles",
    "small_mammals",
    "trees",
    "vehicles_1",
    "vehicles_2",
]
NUM_FINE_CLASSES = len(FINE_CLASS_NAMES)
NUM_COARSE_CLASSES = len(COARSE_CLASS_NAMES)


"""## Data Preprocessing"""

# load data
(train_images, train_fine_labels), (test_images, test_fine_labels) = cifar100.load_data(
    label_mode="fine"
)
(
    (train_images, train_coarse_labels),
    (test_images, test_coarse_labels),
) = cifar100.load_data(
    label_mode="coarse"
)  # normalize images -> [0,1]
train_images = train_images / 255.0
test_images = test_images / 255.0

# zip labels to do hold out cross validation
combined_labels = list(zip(train_fine_labels, train_coarse_labels))

# train test split the data
train_images, val_images, train_labels, val_labels = train_test_split(
    train_images, combined_labels, test_size=0.2
)

# unzip labels
train_fine_labels, train_coarse_labels = zip(*train_labels)
val_fine_labels, val_coarse_labels = zip(*val_labels)

# convert label lists back to array
train_fine_labels = np.array(train_fine_labels)
train_coarse_labels = np.array(train_coarse_labels)
val_fine_labels = np.array(val_fine_labels)
val_coarse_labels = np.array(val_coarse_labels)

# print shapes before preprocessing
print(train_images.shape, train_fine_labels.shape, train_coarse_labels.shape)

# flatten the cifar100 labels to be a list of integers
train_fine_labels = train_fine_labels.flatten()
test_fine_labels = test_fine_labels.flatten()
val_fine_labels = val_fine_labels.flatten()
train_coarse_labels = train_coarse_labels.flatten()
test_coarse_labels = test_coarse_labels.flatten()
val_coarse_labels = val_coarse_labels.flatten()

# numpy indexing arrays with arrays to one hot encode
# https://stackoverflow.com/questions/38592324/one-hot-encoding-using-numpy
train_fine_labels = np.eye(NUM_FINE_CLASSES)[train_fine_labels]
test_fine_labels = np.eye(NUM_FINE_CLASSES)[test_fine_labels]
val_fine_labels = np.eye(NUM_FINE_CLASSES)[val_fine_labels]
train_coarse_labels = np.eye(NUM_COARSE_CLASSES)[train_coarse_labels]
test_coarse_labels = np.eye(NUM_COARSE_CLASSES)[test_coarse_labels]
val_coarse_labels = np.eye(NUM_COARSE_CLASSES)[val_coarse_labels]

# print data post processing
print(train_images.shape, train_fine_labels.shape, train_coarse_labels.shape)

# create data_generators
train_data_generator = tf.keras.preprocessing.image.ImageDataGenerator(
    featurewise_center=False,
    samplewise_center=False,
    featurewise_std_normalization=False,
    samplewise_std_normalization=False,
    zca_whitening=False,
    rotation_range=60,
    width_shift_range=0.3,
    height_shift_range=0.3,
    zoom_range=0.3,
    shear_range=0.3,
    fill_mode="reflect",
)
train_data_generator.fit(train_images)
val_data_generator = tf.keras.preprocessing.image.ImageDataGenerator()
val_data_generator.fit(val_images)

# credit: https://github.com/keras-team/keras/issues/5036#issuecomment-427326673
def multi_task_generator(generator, image_data, t1_labels, t2_labels, batch_size):
    t1_generator = generator.flow(x=image_data, y=t1_labels, batch_size=batch_size, shuffle=False)
    t2_generator = generator.flow(x=image_data, y=t2_labels, batch_size=batch_size, shuffle=False)
    while True:
        xy1 = t1_generator.next()
        xy2 = t2_generator.next()
        yield xy1[0], [xy1[1], xy2[1]]


# deep learning imports
from tensorflow.keras import Model
from tensorflow.keras.layers import (
    Conv2D,
    Dense,
    Dropout,
    BatchNormalization,
    Flatten,
    Activation,
    Input,
    MaxPool2D,
    Concatenate,
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.metrics import Accuracy, MSE, KLD
from tensorflow.keras.utils import plot_model

# function to create consistent models
def make_model(
    share_conv=False, reuse_coarse_classification=False, reuse_fine_classification=False
):
    if reuse_coarse_classification and reuse_fine_classification:
        raise ValueError("Can only reuse one set of labels")

    inputs = Input(shape=INPUT_SHAPE)
    padding = "same"

    # input setup
    if share_conv:
        conv = Conv2D(64, 3, 1, padding, activation="relu")(inputs)
    else:
        convf = Conv2D(64, 3, 1, padding, activation="relu")(inputs)
        convc = Conv2D(64, 3, 1, padding, activation="relu")(inputs)

    # conv layers
    n_blocks = 2
    n_convs_per_block = 2
    base = 2
    exponent = 6
    for block_index in range(n_blocks):
        filters = base ** exponent
        exponent += 1
        for conv_index in range(n_convs_per_block):
            if share_conv:
                conv = Conv2D(filters, 3, 1, padding, activation="relu")(conv)
            else:
                convf = Conv2D(filters, 3, 1, padding, activation="relu")(convf)
                convc = Conv2D(filters, 3, 1, padding, activation="relu")(convc)

        if share_conv:
            conv = MaxPool2D(2, 2)(conv)
            conv = BatchNormalization()(conv)
        else:
            convf = MaxPool2D(2, 2)(convf)
            convf = BatchNormalization()(convf)
            convc = MaxPool2D(2, 2)(convc)
            convc = BatchNormalization()(convc)

    # flatten
    if share_conv:
        flattener = Flatten()
        flatf = flattener(conv)
        flatc = flattener(conv)
    else:
        flatf = Flatten()(convf)
        flatc = Flatten()(convc)

    dropout_rate = 0.25
    # fc
    # coarse
    def dense_block(neurons, in_layer):
        d = Dense(neurons, activation="relu")(in_layer)
        d = Dropout(dropout_rate)(d)
        d = BatchNormalization()(d)
        return d

    # coarse
    fcc = dense_block(1024, flatc)
    fcc = dense_block(512, fcc)

    # fine
    fcf = dense_block(1024, flatf)
    fcf = dense_block(512, fcf)

    # insert fine labels if resuse_fine_labels
    if reuse_fine_classification:
        # get fine labels
        fcf = dense_block(256, fcf)
        fcf = dense_block(128, fcf)
        fine = Dense(NUM_FINE_CLASSES)(fcf)
        fine_output = Activation("softmax", name="fine_output")(fine)
        # concat
        fcc = Concatenate(axis=1)([fcc, fine])
        # get coarse labels
        fcc = dense_block(256, fcc)
        fcc = dense_block(128, fcc)
        coarse = Dense(NUM_COARSE_CLASSES)(fcc)
        coarse_output = Activation("softmax", name="coarse_output")(coarse)

    # insert coarse labels if resuse_coarse_labels
    elif reuse_coarse_classification:
        # get coarse labels
        fcc = dense_block(256, fcc)
        fcc = dense_block(128, fcc)
        coarse = Dense(NUM_COARSE_CLASSES)(fcc)
        coarse_output = Activation("softmax", name="coarse_output")(coarse)
        # concat
        fcf = Concatenate(axis=1)([fcf, coarse])
        fcf = dense_block(256, fcf)
        fcf = dense_block(128, fcf)
        fine = Dense(NUM_FINE_CLASSES)(fcf)
        fine_output = Activation("softmax", name="fine_output")(fine)

    else:
        fcc = dense_block(256, fcc)
        fcc = dense_block(128, fcc)
        coarse = Dense(NUM_COARSE_CLASSES)(fcc)
        coarse_output = Activation("softmax", name="coarse_output")(coarse)
        fcf = dense_block(256, fcf)
        fcf = dense_block(128, fcf)
        fine = Dense(NUM_FINE_CLASSES)(fcf)
        fine_output = Activation("softmax", name="fine_output")(fine)

    # compile and return model
    losses = {
        "fine_output": "categorical_crossentropy",
        "coarse_output": "categorical_crossentropy",
    }
    loss_weights = {"fine_output": 2.0, "coarse_output": 1.0}
    opt = Adam(lr=5e-4)
    model = Model(inputs=inputs, outputs=[fine_output, coarse_output])
    model.compile(
        optimizer=opt,
        loss=losses,
        loss_weights=loss_weights,
        metrics=["accuracy", "mse"],
    )
    return model


early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="loss", mode="min", patience=10, min_delta=1e-3, verbose=1
)
EPOCHS = 100
BATCH_SIZE = 100

"""# 1) Independent networks (the control architecture)
```text
input_image --> conv_layers --> fc_layers --> fine_label       
input_image --> conv_layers --> fc_layers --> coarse_label
```   
fine accuracy:
"""
model = make_model(
    share_conv=False, reuse_coarse_classification=False, reuse_fine_classification=False
)
# plot_model(model, show_shapes=True)

history = model.fit_generator(
    multi_task_generator(
        train_data_generator,
        train_images,
        train_fine_labels,
        train_coarse_labels,
        BATCH_SIZE,
    ),
    validation_data=multi_task_generator(
        val_data_generator, val_images, val_fine_labels, val_coarse_labels, BATCH_SIZE
    ),
    epochs=EPOCHS,
    steps_per_epoch=int(len(train_images) // BATCH_SIZE),
    validation_steps=int(len(val_images) // BATCH_SIZE),
    callbacks=[early_stopping],
    verbose=2,
)

model.save("FFF.h5")

with open("FFF.pkl", "wb") as f:
    pickle.dump(history.history, f)


"""
#### 2) Hard parameter sharing in convolutional layers
```text
                           /--> fc_layers --> fine_label
input_image --> conv_layers      
                           \--> fc_layers --> coarse_label 
```
"""
model = make_model(
    share_conv=True, reuse_coarse_classification=False, reuse_fine_classification=False
)
# plot_model(model, show_shapes=True)

history = model.fit_generator(
    multi_task_generator(
        train_data_generator,
        train_images,
        train_fine_labels,
        train_coarse_labels,
        BATCH_SIZE,
    ),
    validation_data=multi_task_generator(
        val_data_generator, val_images, val_fine_labels, val_coarse_labels, BATCH_SIZE
    ),
    epochs=EPOCHS,
    steps_per_epoch=int(len(train_images) // BATCH_SIZE),
    validation_steps=int(len(val_images) // BATCH_SIZE),
    callbacks=[early_stopping],
    verbose=2,
)

model.save("TFF.h5")

with open("TFF.pkl", "wb") as f:
    pickle.dump(history.history, f)


"""
#### 3) Using coarse label output as weights
```text
input_image   ---->   conv_layers   ---->   fc_layers_1
                                                      \                
                                                    concat -> fc_layers_2  -> fine_label
                                                      /
input_image -> conv_layers -> fc_layers -> coarse_label
``` 
"""
model = make_model(
    share_conv=False, reuse_coarse_classification=True, reuse_fine_classification=False
)
# plot_model(model, show_shapes=True)

history = model.fit_generator(
    multi_task_generator(
        train_data_generator,
        train_images,
        train_fine_labels,
        train_coarse_labels,
        BATCH_SIZE,
    ),
    validation_data=multi_task_generator(
        val_data_generator, val_images, val_fine_labels, val_coarse_labels, BATCH_SIZE
    ),
    epochs=EPOCHS,
    steps_per_epoch=int(len(train_images) // BATCH_SIZE),
    validation_steps=int(len(val_images) // BATCH_SIZE),
    callbacks=[early_stopping],
    verbose=2,
)

model.save("FTF.h5")

with open("FTF.pkl", "wb") as f:
    pickle.dump(history.history, f)


"""
#### 4) Using fine label output as weights
```text
input_image   ---->   conv_layers   ---->   fc_layers_1
                                                      \                
                                                    concat -> fc_layers_2  -> coarse_label
                                                      /
input_image -> conv_layers -> fc_layers -> fine_label
``` 
"""
model = make_model(
    share_conv=False, reuse_coarse_classification=False, reuse_fine_classification=True
)
# plot_model(model, show_shapes=True)

history = model.fit_generator(
    multi_task_generator(
        train_data_generator,
        train_images,
        train_fine_labels,
        train_coarse_labels,
        BATCH_SIZE,
    ),
    validation_data=multi_task_generator(
        val_data_generator, val_images, val_fine_labels, val_coarse_labels, BATCH_SIZE
    ),
    epochs=EPOCHS,
    steps_per_epoch=int(len(train_images) // BATCH_SIZE),
    validation_steps=int(len(val_images) // BATCH_SIZE),
    callbacks=[early_stopping],
    verbose=2,
)

model.save("FFT.h5")

with open("FFT.pkl", "wb") as f:
    pickle.dump(history.history, f)


"""
#### 5) Combination of 2 & 3
```text
                         / -------------> fc_layers_1
                        /                           \
input_image -> conv_layers                        concat -> fc_layers_2 -> fine_label
                        \                           /
                         \-> fc_layers -> coarse_label 
```
"""
model = make_model(
    share_conv=True, reuse_coarse_classification=True, reuse_fine_classification=False
)
# plot_model(model, show_shapes=True)

history = model.fit_generator(
    multi_task_generator(
        train_data_generator,
        train_images,
        train_fine_labels,
        train_coarse_labels,
        BATCH_SIZE,
    ),
    validation_data=multi_task_generator(
        val_data_generator, val_images, val_fine_labels, val_coarse_labels, BATCH_SIZE
    ),
    epochs=EPOCHS,
    steps_per_epoch=int(len(train_images) // BATCH_SIZE),
    validation_steps=int(len(val_images) // BATCH_SIZE),
    callbacks=[early_stopping],
    verbose=2,
)

model.save("TTF.h5")

with open("TTF.pkl", "wb") as f:
    pickle.dump(history.history, f)


"""
#### 6) Combination of 2 & 4
```text
                         / -------------> fc_layers_1
                        /                           \
input_image -> conv_layers                        concat -> fc_layers_2 -> coarse_label
                        \                           /
                         \-> fc_layers -> fine_label 
```
"""
model = make_model(
    share_conv=True, reuse_coarse_classification=False, reuse_fine_classification=True
)
# plot_model(model, show_shapes=True)

history = model.fit_generator(
    multi_task_generator(
        train_data_generator,
        train_images,
        train_fine_labels,
        train_coarse_labels,
        BATCH_SIZE,
    ),
    validation_data=multi_task_generator(
        val_data_generator, val_images, val_fine_labels, val_coarse_labels, BATCH_SIZE
    ),
    epochs=EPOCHS,
    steps_per_epoch=int(len(train_images) // BATCH_SIZE),
    validation_steps=int(len(val_images) // BATCH_SIZE),
    callbacks=[early_stopping],
    verbose=2,
)

model.save("TFT.h5")

with open("TFT.pkl", "wb") as f:
    pickle.dump(history.history, f)

