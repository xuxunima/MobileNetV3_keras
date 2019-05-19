import tensorflow as tf
import keras
from keras import Input,Model
from keras.layers import Activation
from layers import Bneck
import os
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
"""MobileNetv3 in keras
"""
from keras.utils.generic_utils import get_custom_objects


def Hswish(x):
    return x * tf.nn.relu6(x + 3) / 6


get_custom_objects().update({'custom_activation': Activation(Hswish)})

def MobileNetv3_large(num_classes=1000,input_shape=(224,224,3)):
    x = Input(shape=input_shape)
    out = keras.layers.Conv2D(16,3,strides=2,padding='same',use_bias=False)(x)
    out = keras.layers.BatchNormalization()(out)
    out = Activation(Hswish)(out)

    out = Bneck(out,3,16,16,16,keras.layers.ReLU(),1,False)
    out = Bneck(out,3,16,64,24,keras.layers.ReLU(),2,False)
    out = Bneck(out,3,24,72,24,keras.layers.ReLU(),1,False)
    out = Bneck(out,5,24,72,40,keras.layers.ReLU(),2,True)
    out = Bneck(out,5,40,120,40,keras.layers.ReLU(),1,True)
    out = Bneck(out,5,40,120,40,keras.layers.ReLU(),1,True)
    out = Bneck(out,3,40,240,80,Activation(Hswish),2,False)
    out = Bneck(out,3,80,200,80,Activation(Hswish),1,False)
    out = Bneck(out,3,80,184,80,Activation(Hswish),1,False)
    out = Bneck(out,3,80,184,80,Activation(Hswish),1,False)
    out = Bneck(out,3,80,480,112,Activation(Hswish),1,True)
    out = Bneck(out,3,112,672,112,Activation(Hswish),1,True)
    out = Bneck(out,5,112,672,160,Activation(Hswish),1,True)
    out = Bneck(out,5,160,672,160,Activation(Hswish),2,True)
    out = Bneck(out,5,160,960,160,Activation(Hswish),1,True)

    out = keras.layers.Conv2D(filters=960,kernel_size=3,strides=1,padding='same')(out)
    out = keras.layers.BatchNormalization()(out)
    out = Activation(Hswish)(out)
    out = keras.layers.GlobalAveragePooling2D()(out)
    out = keras.layers.Reshape((1,1,-1))(out)
    out = keras.layers.Conv2D(filters=1280,kernel_size=1,strides=1)(out)
    out = Activation(Hswish)(out)
    out = keras.layers.Conv2D(filters=num_classes,kernel_size=1,strides=1)(out)
    out = keras.layers.Flatten()(out)
    out = keras.layers.Softmax()(out)
    model = Model(inputs=x,outputs=out)
    return model

def generate(batch, shape, ptrain, pval):
    """Data generation and augmentation

    # Arguments
        batch: Integer, batch size.
        size: Integer, image size.
        ptrain: train dir.
        pval: eval dir.

    # Returns
        train_generator: train set generator
        validation_generator: validation set generator
        count1: Integer, number of train set.
        count2: Integer, number of test set.
    """

    #  Using the data Augmentation in traning data
    datagen1 = ImageDataGenerator(
        rescale=1. / 255,
        shear_range=0.2,
        zoom_range=0.2,
        rotation_range=90,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True)

    datagen2 = ImageDataGenerator(rescale=1. / 255)

    train_generator = datagen1.flow_from_directory(
        ptrain,
        target_size=shape,
        batch_size=batch,
        class_mode='categorical')

    validation_generator = datagen2.flow_from_directory(
        pval,
        target_size=shape,
        class_mode='categorical')

    count1 = 0
    for root, dirs, files in os.walk(ptrain):
        for each in files:
            count1 += 1

    count2 = 0
    for root, dirs, files in os.walk(pval):
        for each in files:
            count2 += 1

    return train_generator, validation_generator, count1, count2

def train():
    model = MobileNetv3_large(5)
    save_dir = "./snapshot"
    shape = (224, 224, 3)
    n_class = 5
    batch = 32
    train_dir = "E:\\NN\\dataset\\mini_flowers\\train"
    val_dir = "E:\\NN\\dataset\\mini_flowers\\val"

    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    opt = Adam(lr=0.001)

    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

    train_generator, validation_generator, count1, count2 = generate(batch, shape[:2], train_dir, val_dir)

    hist = model.fit_generator(
        train_generator,
        validation_data=validation_generator,
        steps_per_epoch=count1 // batch,
        validation_steps=count2 // batch,
        epochs=50,
    )

if __name__ == "__main__":
    train()