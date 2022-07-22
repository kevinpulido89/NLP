import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img, array_to_img
from tensorflow.keras.layers import Input, Dense, Dropout, LeakyReLU, BatchNormalization
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, MaxPooling2D, UpSampling2D
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import RMSprop, SGD, Adam, Nadam
from tensorflow.keras import regularizers


def create_dataset_tf(img_folder):
    class_name=[]
    tf_img_data_array=[]   
    
    for dir1 in os.listdir(img_folder):
        for file in os.listdir(os.path.join(img_folder, dir1)):
            image= os.path.join(img_folder,dir1, file)
            image = tf.io.read_file(image)
            image = tf.io.decode_jpeg(image, channels=3)
            image = tf.image.resize(image, (IMG_WIDTH, IMG_HEIGHT))
            image = tf.cast(image / 255., tf.float32)
            tf_img_data_array.append(image)
            class_name.append(dir1)                   
    return tf.stack(tf_img_data_array, axis=0),class_name

IMG_WIDTH=256
IMG_HEIGHT=256
batch_size=4

train_dir = 'datasets/imgs/train/'
test_dir  = 'datasets/imgs/test/'
val_dir   = 'datasets/imgs/val/'

train_dir = 'datasets/imgs/train/'
tf_img_data_train, class_name=create_dataset_tf(train_dir)
print("Shape de tf_img_train: ", tf_img_data_train.shape)

test_dir = 'datasets/imgs/test/'
tf_img_data_test, _ = create_dataset_tf(test_dir)
print("Shape de tf_img_test: ", tf_img_data_test.shape)

val_dir = 'datasets/imgs/val/'
tf_img_data_val, _ =create_dataset_tf(val_dir)
print("Shape de tf_img_val: ", tf_img_data_val.shape)

target_dict={k: v for v, k in enumerate(np.unique(class_name))}
print("Diccionario de Labels: ", target_dict)
