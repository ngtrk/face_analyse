import os

import cv2
import numpy as np
import tensorflow as tf

from sklearn.model_selection import train_test_split
from keras.applications.efficientnet import EfficientNetB1



def load_images(path, IMG_SHAPE):
    resized = []
    age = []

    for img in os.listdir(path):
        ages = img.split('_')[0]
        img = cv2.imread(os.path.join(path, img))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        resized.append(tf.image.resize(img / 255.0, (IMG_SHAPE[0], IMG_SHAPE[1])))
        age.append(np.array(ages))


    age = np.array(age, dtype=np.int8)
    resized = np.array(resized, dtype=np.float32)

    return resized, age



def create_model(IMG_SHAPE):
    global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
    base_model = EfficientNetB1(input_shape=IMG_SHAPE, include_top=False)
    
    input_layer = tf.keras.Input(shape=IMG_SHAPE)
    x = base_model(input_layer, training=False)
    x = global_average_layer(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.Dense(64, activation='relu')(x)
    output_layer = tf.keras.layers.Dense(1, activation='linear')(x)

    return tf.keras.Model(input_layer, output_layer)



def main(path, IMG_SHAPE, LR, EPOCHS, batch_size, filename, run_eagerly=True, steps_per_epoch=256):
    images, age = load_images(path, IMG_SHAPE)
    train_img_age, test_img_age, train_target_age, test_target_age = train_test_split(images, age, random_state=42)

    model = create_model(IMG_SHAPE)

    model.compile(run_eagerly=run_eagerly, optimizer=tf.keras.optimizers.Adam(learning_rate=LR), loss='mse', metrics=['mae'])

    tf.keras.backend.clear_session()

    model.fit(train_img_age, train_target_age, epochs=EPOCHS, steps_per_epoch=steps_per_epoch, 
              batch_size=batch_size, validation_data=(test_img_age, test_target_age))

    model.save(filename)



if __name__ == '__main__':

    path = 'data/UTKFace/'
    IMG_SHAPE = (64, 64, 3)
    LR = 0.0001
    EPOCHS = 15
    batch_size = 8
    filename = 'trained_models/age_model.h5'

    
    main(path, IMG_SHAPE, LR, EPOCHS, batch_size, filename)

