import os
import shutil
import tensorflow as tf
from keras.utils import image_dataset_from_directory



def get_data(class_names, img_path, train_path, val_path, batch_size, IMG_SIZE, AUTOTUNE):

    for x in class_names:
        if not os.path.exists(os.path.join(train_path, x)):
            os.mkdir(os.path.join(train_path, x))

        all_images = os.listdir(os.path.join(img_path, x))

        for i in range(int(len(all_images) * .8)):
            shutil.copyfile(os.path.join(img_path, x, all_images[i]), 
                            os.path.join(train_path, x, all_images[i]))

        if not os.path.exists(os.path.join(val_path, x)):
            os.mkdir(os.path.join(val_path, x))

        for i in range(int(len(all_images) * .8), len(all_images)):
            shutil.copyfile(os.path.join(img_path, x, all_images[i]), 
                            os.path.join(val_path, x, all_images[i]))



    train_dataset = image_dataset_from_directory(train_path, shuffle=True, batch_size=batch_size,
                                                 image_size=IMG_SIZE)


    val_dataset = image_dataset_from_directory(val_path, shuffle=True, batch_size=batch_size,
                                               image_size=IMG_SIZE)


    train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
    val_dataset = val_dataset.prefetch(buffer_size=AUTOTUNE)

    return train_dataset, val_dataset



data_augmentation = tf.keras.Sequential([
  tf.keras.layers.RandomFlip('horizontal'),
  tf.keras.layers.RandomRotation(0.2),
])



def create_model(class_names, IMG_SIZE):
    preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input
    global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
    prediction_layer = tf.keras.layers.Dense(len(class_names))


    IMG_SHAPE = IMG_SIZE + (3,)
    base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE, include_top=False, weights='imagenet')


    input_layer = tf.keras.Input(shape=IMG_SHAPE)
    x = data_augmentation(input_layer)
    x = preprocess_input(x)
    x = base_model(x, training=False)
    x = global_average_layer(x)

    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.Flatten()(x)
    output_layer = prediction_layer(x)

    return tf.keras.Model(input_layer, output_layer)



def main(class_names, IMG_SIZE, LR, img_path, train_path, val_path, batch_size, EPOCHS, AUTOTUNE, filename):

    train_dataset, val_dataset = get_data(class_names, img_path, train_path, val_path, batch_size, IMG_SIZE, AUTOTUNE)


    model = create_model(class_names, IMG_SIZE)

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LR, decay=.001, clipnorm=.4),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])

    tf.keras.backend.clear_session()


    model.fit(train_dataset, epochs=EPOCHS, validation_data=val_dataset)
    model.save(filename)




if __name__ == '__main__':

    class_names = ['afraid', 'angry', 'disgusted', 'happy', 'neutral', 'sad', 'surprised']
    batch_size = 32
    IMG_SIZE = (160, 160)
    AUTOTUNE = tf.data.AUTOTUNE
    LR = 0.0001
    EPOCHS = 20
    img_path = 'data/KDEF_GREYSCALED_CROPPED/'
    train_path = 'data/train/'
    val_path = 'data/val/'
    filename = 'trained_models/emotion_model.h5'


    if not os.path.exists(train_path):
        os.mkdir(train_path)
    if not os.path.exists(val_path):
        os.mkdir(val_path)


    main(class_names, IMG_SIZE, LR, img_path, train_path, val_path, batch_size, EPOCHS, AUTOTUNE, filename)



