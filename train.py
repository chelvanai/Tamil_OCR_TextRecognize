import os
import cv2
import numpy as np
import string
import time
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

import tensorflow.keras.backend as K
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Dense, Reshape, BatchNormalization, Input, Conv2D, MaxPool2D, Lambda, Bidirectional

from tensorflow.compat.v1.keras.layers import CuDNNLSTM
from tensorflow.keras.models import Model
from tensorflow.keras.activations import relu, sigmoid, softmax

from tensorflow.keras.utils import to_categorical, Sequence
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.python.keras.layers import LSTM
from tqdm import tqdm
from collections import Counter

from PIL import Image

final_paths = []
final_texts = []

path = "./tamil_dataset"

for i in os.listdir(path):
    for j in os.listdir(path + "/" + str(i)):
        for x in os.listdir(path + "/" + str(i) + "/" + str(j)):
            final_paths.append(path + "/" + str(i) + "/" + str(j) + "/" + x)
            final_texts.append(str(x).split("_")[1])

vocab = set("".join(map(str, final_texts)))
print(sorted(vocab))

print(Counter("".join(map(str, final_texts))))
char_list = sorted(vocab)
print(char_list)


def encode_to_labels(txt):
    # encoding each output word into digits
    dig_lst = []

    for index, char in enumerate(txt):
        char = tf.strings.unicode_split(char, input_encoding="UTF-8")
        try:
            dig_lst.append(char_list.index(char))
        except:
            print(char)

    return dig_lst


train_final_paths = final_paths[: int(len(final_paths) * 0.90)]
train_final_texts = final_texts[: int(len(final_texts) * 0.90)]

val_final_paths = final_paths[int(len(final_paths) * 0.90):]
val_final_texts = final_texts[int(len(final_texts) * 0.90):]

print(len(train_final_paths), len(val_final_paths))

max_label_len = max([len(str(text)) for text in final_texts])
print(max_label_len)


class My_Generator(Sequence):

    def __init__(self, image_filenames, labels, batch_size):
        self.image_filenames, self.labels = image_filenames, labels
        self.batch_size = batch_size

    def __len__(self):
        return int(np.ceil(len(self.image_filenames) / float(self.batch_size)))

    def __getitem__(self, idx):

        batch_paths = self.image_filenames[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_texts = self.labels[idx * self.batch_size:(idx + 1) * self.batch_size]

        images = []
        training_txt = []
        train_label_length = []
        train_input_length = []

        for im_path, text in zip(batch_paths, batch_texts):

            try:
                text = str(text).strip()
                img = cv2.cvtColor(cv2.imread(im_path), cv2.COLOR_BGR2GRAY)

                ### actually returns h, w
                h, w = img.shape

                ### if height less than 32
                if h < 32:
                    add_zeros = np.ones((32 - h, w)) * 255
                    img = np.concatenate((img, add_zeros))
                    h = 32

                ## if width less than 128
                if w < 128:
                    add_zeros = np.ones((h, 128 - w)) * 255
                    img = np.concatenate((img, add_zeros), axis=1)
                    w = 128

                ### if width is greater than 128 or height greater than 32
                if w > 128 or h > 32:
                    img = cv2.resize(img, (128, 32))

                img = np.expand_dims(img, axis=2)

                # Normalize each image
                img = img / 255.

                images.append(img)
                training_txt.append(encode_to_labels(text))
                train_label_length.append(len(text))
                train_input_length.append(31)
            except:

                pass

        return [np.array(images),
                pad_sequences(training_txt, maxlen=max_label_len, padding='post', value=len(char_list)),
                np.array(train_input_length),
                np.array(train_label_length)], np.zeros(len(images))


batch_size = 4
train_generator = My_Generator(train_final_paths, train_final_texts, batch_size)
val_generator = My_Generator(val_final_paths, val_final_texts, batch_size)

# input with shape of height=32 and width=128
inputs = Input(shape=(32, 128, 1))

conv_1 = Conv2D(16, (3, 3), activation='relu', padding='same')(inputs)
pool_1 = MaxPool2D(pool_size=(2, 2), strides=2)(conv_1)

conv_2 = Conv2D(32, (3, 3), activation='relu', padding='same')(pool_1)
pool_2 = MaxPool2D(pool_size=(2, 2), strides=2)(conv_2)

conv_3 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool_2)
conv_4 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv_3)
pool_4 = MaxPool2D(pool_size=(2, 1))(conv_4)

conv_5 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool_4)
# Batch normalization layer
batch_norm_5 = BatchNormalization()(conv_5)

conv_6 = Conv2D(64, (3, 3), activation='relu', padding='same')(batch_norm_5)
batch_norm_6 = BatchNormalization()(conv_6)
pool_6 = MaxPool2D(pool_size=(2, 1))(batch_norm_6)

conv_7 = Conv2D(64, (2, 2), activation='relu')(pool_6)

squeezed = Lambda(lambda x: K.squeeze(x, 1))(conv_7)

# bidirectional LSTM layers with units=128
blstm_1 = Bidirectional(LSTM(128, return_sequences=True))(squeezed)
blstm_2 = Bidirectional(LSTM(128, return_sequences=True))(blstm_1)

outputs = Dense(len(char_list) + 1, activation='softmax')(blstm_2)

# model to be used at test time
act_model = Model(inputs, outputs)

labels = Input(name='the_labels', shape=[max_label_len], dtype='float32')
input_length = Input(name='input_length', shape=[1], dtype='int64')
label_length = Input(name='label_length', shape=[1], dtype='int64')


def ctc_lambda_func(args):
    y_pred, labels, input_length, label_length = args
    return K.ctc_batch_cost(labels, y_pred, input_length, label_length)


loss_out = Lambda(ctc_lambda_func,
                  output_shape=(1,),
                  name='ctc')([outputs, labels, input_length, label_length])

# model to be used at training time
model = Model(inputs=[inputs, labels, input_length, label_length], outputs=loss_out)

file_path = "C_LSTM_best.hdf5"

model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer='adam')

checkpoint = ModelCheckpoint(filepath=file_path,
                             monitor='val_loss',
                             verbose=1,
                             save_best_only=True,
                             mode='min')

callbacks_list = [checkpoint]

epochs = 35

history = model.fit(train_generator,
                    epochs=epochs,
                    steps_per_epoch=len(train_final_paths) // batch_size,
                    validation_data=val_generator,
                    validation_steps=len(val_final_paths) // batch_size,
                    verbose=1,
                    callbacks=callbacks_list,
                    shuffle=True)


def pre_process_image(path):
    img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2GRAY)
    ### actually returns h, w
    h, w = img.shape

    ### if height less than 32
    if h < 32:
        add_zeros = np.ones((32 - h, w)) * 255
        img = np.concatenate((img, add_zeros))
        h = 32

    ## if width less than 128
    if w < 128:
        add_zeros = np.ones((h, 128 - w)) * 255
        img = np.concatenate((img, add_zeros), axis=1)
        w = 128

    ### if width is greater than 128 or height greater than 32
    if w > 128 or h > 32:
        img = cv2.resize(img, (128, 32))

    img = np.expand_dims(img, axis=2)

    # Normalize each image
    img = img / 255.

    return img


def predict_output(img):
    # predict outputs on validation images
    prediction = act_model.predict(np.array([img]))
    ## shape (batch_size, num_timesteps, vocab_size)

    # use CTC decoder
    out = K.get_value(K.ctc_decode(prediction,
                                   input_length=np.ones(prediction.shape[0]) * prediction.shape[1],
                                   greedy=True)[0][0])

    ## get the final text
    for x in out:

        print("predicted text = ", end='')

        for p in x:

            if int(p) != -1:
                print(char_list[int(p)], end='')

        print('\n')


act_model.load_weights('C_LSTM_best.hdf5')

test_img = pre_process_image("./test_images/20_வரவேற்று,_.jpg")
print(predict_output(test_img))

test_img = pre_process_image("./test_images/183_என்று_.jpg")
print(predict_output(test_img))

test_img = pre_process_image("./test_images/191_தனது_.jpg")
print(predict_output(test_img))

test_img = pre_process_image("./test_images/192_ஆறாவது_.jpg")
print(predict_output(test_img))
