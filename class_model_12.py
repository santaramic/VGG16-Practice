# -*- coding:utf-8 -*-
import tensorflow as tf


def Model(input_shape=(256, 256, 3)):
    h = inputs = tf.keras.Input(input_shape)   

    h = tf.keras.layers.ZeroPadding2D((3, 3))(h)
    h = tf.keras.layers.Conv2D(filters=64,
                               kernel_size=7,
                               strides=2,
                               padding="valid",
                               use_bias=False)(h)   
    h = tf.keras.layers.BatchNormalization()(h) 
    h = tf.keras.layers.ReLU()(h)   

    h = tf.keras.layers.Conv2D(filters=128,
                               kernel_size=3,
                               strides=1,
                               padding="same",
                               use_bias=False)(h) 
    h = tf.keras.layers.BatchNormalization()(h) 
    h = tf.keras.layers.ReLU()(h)

    h = tf.keras.layers.MaxPooling2D(pool_size=(2,2), strides=2)(h) 

    h = tf.keras.layers.Conv2D(filters=256,
                               kernel_size=3,
                               strides=1,
                               padding="same",
                               use_bias=False)(h)
    h = tf.keras.layers.BatchNormalization()(h) 
    h = tf.keras.layers.ReLU()(h)   

    h = tf.keras.layers.MaxPooling2D(pool_size=(2,2), strides=2)(h) 

    h = tf.keras.layers.Conv2D(filters=512,
                               kernel_size=3,
                               strides=1,
                               padding="same",
                               use_bias=False)(h)
    h = tf.keras.layers.BatchNormalization()(h) 
    h = tf.keras.layers.ReLU()(h)   

    h = tf.keras.layers.Conv2D(filters=512,
                               kernel_size=3,
                               strides=1,
                               padding="same",
                               use_bias=False)(h)
    h = tf.keras.layers.BatchNormalization()(h) 
    h = tf.keras.layers.ReLU()(h)  

    h = tf.keras.layers.Conv2D(filters=512,
                               kernel_size=3,
                               strides=1,
                               padding="same",
                               use_bias=False)(h)
    h = tf.keras.layers.BatchNormalization()(h) 
    h = tf.keras.layers.ReLU()(h)   

    h = tf.keras.layers.MaxPooling2D(pool_size=(2,2), strides=2)(h) 

    h = tf.keras.layers.Conv2D(filters=512,
                               kernel_size=3,
                               strides=1,
                               padding="same",
                               use_bias=False)(h)
    h = tf.keras.layers.BatchNormalization()(h) 
    h = tf.keras.layers.ReLU()(h)   

    h = tf.keras.layers.Conv2D(filters=512,
                               kernel_size=3,
                               strides=1,
                               padding="same",
                               use_bias=False)(h)
    h = tf.keras.layers.BatchNormalization()(h) 
    h = tf.keras.layers.ReLU()(h)   

    h = tf.keras.layers.Conv2D(filters=512,
                               kernel_size=3,
                               strides=1,
                               padding="same",
                               use_bias=False)(h)
    h = tf.keras.layers.BatchNormalization()(h) 
    h = tf.keras.layers.ReLU()(h)   

    h = tf.keras.layers.MaxPooling2D(pool_size=(2,2), strides=2)(h) 

    h = tf.keras.layers.Conv2D(filters=512,
                               kernel_size=3,
                               strides=1,
                               padding="same",
                               use_bias=False)(h)
    h = tf.keras.layers.BatchNormalization()(h) 
    h = tf.keras.layers.ReLU()(h)   

    h = tf.keras.layers.Conv2D(filters=512,
                               kernel_size=3,
                               strides=1,
                               padding="same",
                               use_bias=False)(h)
    h = tf.keras.layers.BatchNormalization()(h) 
    h = tf.keras.layers.ReLU()(h)   

    h = tf.keras.layers.Conv2D(filters=512,
                               kernel_size=3,
                               strides=1,
                               padding="same",
                               use_bias=False)(h)
    h = tf.keras.layers.BatchNormalization()(h) 
    h = tf.keras.layers.ReLU()(h)   

    h = tf.keras.layers.MaxPooling2D(pool_size=(2,2), strides=2)(h)
    # [B, 4, 4, 512]

    h = tf.keras.layers.Flatten()(h)
    h = tf.keras.layers.Dense(1024)(h)
    h = tf.keras.layers.ReLU()(h)
    h = tf.keras.layers.Dense(1)(h)
    #h = tf.nn.sigmoid(h)
    
    return tf.keras.Model(inputs=inputs, outputs=h)
