# -*- coding:utf-8 -*-
from class_model_12 import *   # class_model.py 에 있는 모든 코드의 기능을 다 쓰겠다.
from random import shuffle

import numpy as np
import easydict # 저는 이 기능을 전역변수로서 사용하려고 합니다.
import matplotlib.pyplot as plt
import os

FLAGS = easydict.EasyDict({"img_size": 256,
                           
                           "tr_txt_path": "E:/[1]DB/Cat_and_dog/train.txt",

                           "tr_img_path": "E:/[1]DB/Cat_and_dog/Cat_Dog_500_train/",

                           "val_txt_path": "E:/[1]DB/Cat_and_dog/val.txt",

                           "val_img_path": "E:/[1]DB/Cat_and_dog/Cat_Dog_100_val/",

                           "te_txt_path": "E:/[1]DB/Cat_and_dog/test.txt",

                           "te_img_path": "E:/[1]DB/Cat_and_dog/Cat_Dog_200_test/",
                           
                           "lab_path": "E:/etc_label/label.txt",
                           
                           "batch_size": 10,
                           
                           "epochs": 100,})
# Cat = 0, Dog = 1
optim = tf.keras.optimizers.Adam(0.0001)

def input_func(img_, lab_):

    img = tf.io.read_file(img_)
    img = tf.image.decode_jpeg(img, 3)
    img = tf.image.resize(img, [FLAGS.img_size, FLAGS.img_size]) / 255.

    lab = lab_

    return img, lab

def cal_loss(pratice_model, batch_images, batch_labels):

    with tf.GradientTape() as tape:

        logits = pratice_model(batch_images, True)
        loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)(batch_labels, 
                                                                    logits)

    grads = tape.gradient(loss, pratice_model.trainable_variables)
    optim.apply_gradients(zip(grads, pratice_model.trainable_variables))

    return loss

def main():
    
    pratice_model = Model(input_shape=(FLAGS.img_size, FLAGS.img_size, 3))
    pratice_model.summary()

    
    train_img_data = np.loadtxt(FLAGS.tr_txt_path, dtype="<U200", skiprows=0, usecols=0)
    train_img_data = [FLAGS.tr_img_path + data for data in train_img_data]
    train_lab_data = np.loadtxt(FLAGS.tr_txt_path, dtype=np.int32, skiprows=0, usecols=1)

    val_img_data = np.loadtxt(FLAGS.val_txt_path, dtype="<U200", skiprows=0, usecols=0)
    val_img_data = [FLAGS.val_img_path + data for data in val_img_data]
    val_lab_data = np.loadtxt(FLAGS.val_txt_path, dtype=np.int32, skiprows=0, usecols=1)

    te_img_data = np.loadtxt(FLAGS.te_txt_path, dtype="<U200", skiprows=0, usecols=0)
    te_img_data = [FLAGS.te_img_path + data for data in te_img_data]
    te_lab_data = np.loadtxt(FLAGS.te_txt_path, dtype=np.int32, skiprows=0, usecols=1)

    count = 0
    for epoch in range(FLAGS.epochs):
        TR = list(zip(train_img_data, train_lab_data))
        shuffle(TR)
        train_img_data, train_lab_data = zip(*TR)
        train_img_data, train_lab_data = np.array(train_img_data), np.array(train_lab_data)

        tf_data_ge = tf.data.Dataset.from_tensor_slices((train_img_data, train_lab_data))
        tf_data_ge = tf_data_ge.shuffle(len(train_img_data))
        tf_data_ge = tf_data_ge.map(input_func)
        tf_data_ge = tf_data_ge.batch(FLAGS.batch_size)
        tf_data_ge = tf_data_ge.prefetch(tf.data.experimental.AUTOTUNE)

        tr_iter = iter(tf_data_ge)
        tr_idx = len(train_img_data) // FLAGS.batch_size
        for step in range(tr_idx):
            batch_images, batch_labels = next(tr_iter)

            loss = cal_loss(pratice_model, batch_images, batch_labels)

            if count % 10 == 0:
                print("Epoch: {} [{}/{}] Loss = {}".format(epoch, step+1, tr_idx,
                                                           loss))

            count += 1



    # Train, Test, Validatio
 
if __name__ == "__main__":
    main()
