from __future__ import absolute_import, division, print_function, unicode_literals
import functools


import io
import json
import math
import numpy as np
import os
import pandas as pd
import spacy
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.utils import Sequence



class SquadSequence(Sequence):
    def __init__(self, x_set, y_set, batch_size):
        self.x, self.y = x_set, y_set
        self.batch_size = batch_size

    def __len__(self):
        return math.ceil(len(self.x) / self.batch_size)

    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size:(idx + 1) *
        self.batch_size]
        batch_y = self.y[idx * self.batch_size:(idx + 1) *
        self.batch_size]

        return np.array(batch_x), np.array(batch_y)


def load_dataset():
    nlp = spacy.load("en_core_web_lg")

    train_file_path = os.path.join('dev-v2.0.json')

    f = open(train_file_path, "r")

    js = json.load(f)

    cntxt = np.zeros((500, 4), dtype=np.int64)

    ques  = np.zeros(( 20, 4), dtype=np.int64)

    imp   = np.zeros((  1,  ), dtype=np.int64)

    ans   = np.zeros(( 10, 4), dtype=np.float64)


    idx = 0

    x = None
    y = None


    for jsd in js["data"][0]["paragraphs"]:

        para = nlp(jsd["context"])

        # for t in para:
        #     print(t.i, t.head.i, t.pos, t.norm)
       
        idx = 0
        for t in para:
            print('t.i:%d, t.head.i:%d, t.pos:%d, t.norm:%d' %(t.i, t.head.i, t.pos, t.norm))
            cntxt[idx] = [np.int64(np.uint64(t.i)), np.int64(np.uint64(t.head.i)), 
                np.int64(np.uint64(t.pos)), np.int64(np.uint64(t.norm))]
            idx +=1

        for jsd2 in jsd["qas"]:
            qu = nlp(jsd2["question"])

            idx=0
            for t in qu:
                ques[idx] = [np.int64(np.uint64(t.i)), np.int64(np.uint64(t.head.i)), 
                    np.int64(np.uint64(t.pos)), np.int64(np.uint64(t.norm))]
                idx += 1

            if jsd2["is_impossible"] == True:
                imp[0] = 0

            else:
                imp[0] = 1


        if y is None:
            x = np.array( cntxt    )
            y = np.array( ans      )
        else:
            x = np.concatenate((x, cntxt    ))
            y = np.concatenate((y, ans      ))

    print("x shape:  ", (x.reshape((39,2000)).shape))
    print("y shape:  ", y.reshape((39,40)).shape, "y type:   ", type(y[0,0]))
    print("xy shape: ", (np.hstack((x.reshape((39,2000)),y.reshape((39,40))))).shape)

    xy = np.hstack((x.reshape((39,2000)),y.reshape((39,40))))
    return np.array(xy)


if __name__ == "__main__":

    ds = load_dataset()

    xs = ds[:, 0:-2]
    ys = ds[:, -2:]

    dataset = SquadSequence(xs, ys, 10)

    inputs = tf.keras.Input(shape=(2038))
    x = layers.Dense(10, activation='relu')(inputs)
    #x = layers.Reshape((30,300))(inputs)
    # lstm_layer1 = layers.LSTM(300)(inputs)
    outputs = tf.keras.layers.Dense(2)(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    model.compile(optimizer='adam',
                  loss='cosine_similarity',
                  metrics=['accuracy'])

    history = model.fit(x=dataset, verbose=2, shuffle=False)


    if False:
        test_file_path = os.path.join('test1_2.csv')

        df2 = pd.read_csv(test_file_path)

        df2_target = df2.loc[:,'A0_0':'A0_299']
        df2_values = df2.loc[:,'S0_0':'Q9_299']

        dataset2 = tf.data.Dataset.from_tensor_slices((df2_values.values, df2_target.values)).batch(1000).repeat()

        model.evaluate(x=dataset2, steps=1, verbose=2)




