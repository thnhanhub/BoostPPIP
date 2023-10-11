"""
@thnhan
"""
import pickle

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout, Flatten, Embedding, add
from tensorflow.keras.initializers import GlorotUniform
import tensorflow as tf


def get_layer_embedding(Vocal_W1, fixlen, name, train_embeddings=False):
    trained_W = Vocal_W1['W1']
    trained_W = np.vstack([trained_W, np.zeros(shape=(1, trained_W.shape[1]))])  # them vecto 0 bieu dien pad '_'
    layer = Embedding(
        input_dim=trained_W.shape[0], output_dim=trained_W.shape[1],
        weights=[trained_W],
        input_length=fixlen,
        trainable=train_embeddings,
        name=name
    )
    return layer


def fe_embedding(emdedding, n_dim, drop, n_units, kernel_init, name):
    dnn = Sequential()
    # print(n_dim)
    dnn.add(get_layer_embedding(emdedding,
                                n_dim, name=name, train_embeddings=True))
    dnn.add(Flatten())

    # dnn.add(BatchNormalization())
    dnn.add(Dropout(drop))

    dnn.add(Dense(n_units,
                  kernel_initializer=kernel_init,
                  activation='relu', ))
    dnn.add(BatchNormalization())
    dnn.add(Dropout(drop))

    dnn.add(Dense(n_units // 2,
                  kernel_initializer=kernel_init,
                  activation='relu', ))
    dnn.add(BatchNormalization())
    dnn.add(Dropout(drop))

    dnn.add(Dense(n_units // 4,
                  kernel_initializer=kernel_init,
                  activation='relu', ))
    dnn.add(BatchNormalization())
    dnn.add(Dropout(drop))

    dnn.add(Dense(n_units // 8,
                  kernel_initializer=kernel_init,
                  activation='relu', ))
    dnn.add(BatchNormalization())
    dnn.add(Dropout(drop))

    # --- Tang 5
    # dnn.add(Dense(n_units // 16,
    #               kernel_initializer=kernel_init,
    #               activation='relu',
    #               kernel_regularizer=W_regular))
    # dnn.add(BatchNormalization())
    # dnn.add(Dropout(drop))

    return dnn


def net_embedding(dim1, embedding, drop=0.5, n_units=1024, seed=123456):
    # ====== To reproduce
    tf.random.set_seed(seed)
    glouni = GlorotUniform(seed=seed)

    # ====== Extraction
    w1 = fe_embedding(embedding, dim1,
                      drop=drop,
                      n_units=n_units,
                      kernel_init=glouni, name='emb1')

    w2 = fe_embedding(embedding, dim1,
                      drop=drop,
                      n_units=n_units,
                      kernel_init=glouni, name='emb2')
    # print(dim1, dim2)
    in1 = Input(dim1)
    in2 = Input(dim1)
    x1 = w1(in1)
    x2 = w2(in2)

    # --- Merge
    mer = add([x1, x2])
    y = Dense(8, kernel_initializer=glouni,
              activation='relu')(mer)
    y = BatchNormalization()(y)
    y = Dropout(drop)(y)

    # --- Classification
    y = Dense(4, kernel_initializer=glouni,
              activation='relu')(y)
    y = BatchNormalization()(y)
    # y = Dropout(0.25)(y)

    out = Dense(2, kernel_initializer=glouni, activation='softmax')(y)

    final = Model(inputs=[in1, in2], outputs=out)
    tf.keras.utils.plot_model(final, "model.png", show_shapes=True)
    return final
