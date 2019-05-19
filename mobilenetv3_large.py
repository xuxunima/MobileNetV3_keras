import numpy as np
import tensorflow as tf
import keras
from keras import Input,Model
from layers import Bneck, HSwish, SEModule
import os
from keras.preprocessing.image import ImageDataGenerator
"""MobileNetv3 in keras
"""

def MobileNetv3_large(num_classes=1000,input_shape=(224,224,3)):
    x = Input(shape=input_shape)
    out = keras.layers.Conv2D(16,3,strides=2,padding='same',use_bias=False)(x)
    out = keras.layers.BatchNormalization()(out)
    out = HSwish()(out)

    out = Bneck(3,16,16,16,keras.layers.ReLU(),1,None)(out)
    out = Bneck(3,16,64,24,keras.layers.ReLU(),2,None)(out)
    out = Bneck(3,24,72,24,keras.layers.ReLU(),1,None)(out)
    out = Bneck(5,24,72,40,keras.layers.ReLU(),2,SEModule(72))(out)
    out = Bneck(5,40,120,40,keras.layers.ReLU(),1,SEModule(120))(out)
    out = Bneck(5,40,120,40,keras.layers.ReLU(),1,SEModule(120))(out)
    out = Bneck(3,40,240,80,HSwish(),2,None)(out)
    out = Bneck(3,80,200,80,HSwish(),1,None)(out)
    out = Bneck(3,80,184,80,HSwish(),1,None)(out)
    out = Bneck(3,80,184,80,HSwish(),1,None)(out)
    out = Bneck(3,80,480,112,HSwish(),1,SEModule(480))(out)
    out = Bneck(3,112,672,112,HSwish(),1,SEModule(672))(out)
    out = Bneck(5,112,672,160,HSwish(),2,SEModule(672))(out)
    out = Bneck(5,160,960,160,HSwish(),1,SEModule(960))(out)
    out = Bneck(5,160,960,160,HSwish(),1,SEModule(960))(out)

    out = keras.layers.Conv2D(filters=960,kernel_size=1,strides=1,use_bias=False)(out)
    out = keras.layers.BatchNormalization()(out)
    out = HSwish()(out)
    out = keras.layers.GlobalAveragePooling2D()(out)
    out = keras.layers.Reshape((1,1,-1))(out)
    out = keras.layers.Conv2D(filters=1280,kernel_size=1,strides=1)(out)
    out = HSwish()(out)
    out = keras.layers.Conv2D(filters=num_classes,kernel_size=1,strides=1,activation='softmax')(out)
    out = keras.layers.Flatten()(out)
    model = Model(inputs=x,outputs=out)
    return model


def train():
    batch_size = 32
    target_size = (224,224)
    data_dir = "E:\\study\\dataset\\mini_flowers"
    train_dir = os.path.join(data_dir,"train")
    val_dir = os.path.join(data_dir,"val")
    img_gen = ImageDataGenerator(rotation_range=20,
                                 width_shift_range=0.2,
                                 height_shift_range=0.2,
                                 horizontal_flip=True,
                                 rescale=1/255.,
                                 zoom_range=0.5)
    train_generator = img_gen.flow_from_directory(directory=train_dir,
                                                  batch_size=batch_size,
                                                  shuffle=True,
                                                  target_size=target_size,
                                                  class_mode='categorical')
    val_generator = img_gen.flow_from_directory(directory=val_dir,
                                                batch_size=batch_size,
                                                target_size=target_size,
                                                class_mode='categorical')
    model = MobileNetv3_large(5)
    model.compile(loss='categorical_crossentropy',
                  optimizer=keras.optimizers.Adam(lr=0.001),
                  metrics=['acc'])


    snapshots = "./snapshots"
    if not os.path.exists(snapshots):
        os.mkdir(snapshots)
    callback = tf.keras.callbacks.ModelCheckpoint(filepath=os.path.join(snapshots,"mobilenetv3_{{epoch:02d}}.h5".format()),
                                                  verbose=1,
                                                  period=10,
                                                  monitor='val_loss')
    model.fit_generator(train_generator,steps_per_epoch=int(1000/batch_size),
                        epochs=50,validation_data=val_generator,
                        validation_steps=int(500/batch_size),
                        callbacks=[callback]
                                 )


if __name__ == '__main__':
    train()