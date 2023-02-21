import os
import cv2
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split




def load_images(path, IMG_SHAPE):
    images = []
    gender = []

    for img in os.listdir(path):
        genders = img.split('_')[1]
        img = cv2.imread(os.path.join(path, img))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        images.append(tf.image.resize(img, (IMG_SHAPE[0], IMG_SHAPE[1])))

        gender.append(np.array(genders))

    gender = np.array(gender, np.int8)
    images = np.array(images, dtype=np.float32)

    return images, gender



def create_model(IMG_SHAPE):
    preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input
    prediction_layer = tf.keras.layers.Dense(1)

    
    base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE, include_top=False, weights='imagenet')
    global_average_layer = tf.keras.layers.GlobalAveragePooling2D()

    input_layer = tf.keras.Input(shape=IMG_SHAPE)
    x = preprocess_input(input_layer)
    x = base_model(x, training=False)
    x = global_average_layer(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.Flatten()(x)
    output_layer = prediction_layer(x)

    return tf.keras.Model(input_layer, output_layer)



def main(path, IMG_SHAPE, LR, EPOCHS, batch_size, filename, run_eagerly=True, steps_per_epoch=256):

    images, gender = load_images(path, IMG_SHAPE)
    train_img_gender, test_img_gender, train_target_gender, test_target_gender = train_test_split(images, gender, random_state=42)


    model = create_model(IMG_SHAPE)
    model.compile(run_eagerly=run_eagerly, optimizer=tf.keras.optimizers.Adam(learning_rate=LR, clipnorm=.8, decay=.003),
                  loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                  metrics=['accuracy']
                  )
    
    tf.keras.backend.clear_session()


    model.fit(train_img_gender, train_target_gender,
              epochs=EPOCHS, batch_size=batch_size, steps_per_epoch=steps_per_epoch, validation_data=(test_img_gender, test_target_gender))

    model.save(filename)



if __name__ == '__main__':

    path = 'data/UTKFace/'
    IMG_SHAPE = (96, 96, 3)
    LR = 0.0001
    EPOCHS = 10
    batch_size = 8
    filename = 'trained_models/gender_model.h5'

    
    main(path, IMG_SHAPE, LR, EPOCHS, batch_size, filename)


