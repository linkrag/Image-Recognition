import pathlib

import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
import tensorflow as tf

from keras.preprocessing import image
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

data_dir = 'downloads'
data_dir = pathlib.Path(data_dir)

batch_size = 32
img_height = 180
img_width = 180

# Classificação
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size)

class_names = train_ds.class_names

AUTOTUNE = tf.data.experimental.AUTOTUNE

train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

normalization_layer = layers.experimental.preprocessing.Rescaling(1. / 255)

normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
image_batch, labels_batch = next(iter(normalized_ds))
first_image = image_batch[0]

num_classes = 10

# Modelo da Rede

data_augmentation = keras.Sequential(
    [
        layers.experimental.preprocessing.RandomFlip("horizontal",
                                                     input_shape=(img_height,
                                                                  img_width,
                                                                  3)),
        layers.experimental.preprocessing.RandomRotation(0.1),
        layers.experimental.preprocessing.RandomZoom(0.1),
    ]
)

model = Sequential([
    data_augmentation,
    # Número de canais e tamanho do filtro; sem alterar o tamanho da imagem
    layers.experimental.preprocessing.Rescaling(1. / 255),
    layers.Conv2D(16, 3, padding='same', activation='relu'),

    # Elimina algumas conexões entre as camadas para evitar overfitting
    layers.Dropout(0.2),

    # Faz a normalização do batch para começar a próxima camada com a mesma ditribuição
    layers.BatchNormalization(),

    # Mais uma camada para melhorar o aprendizado da RN
    layers.Conv2D(32, 3, padding='same', activation='relu'),

    # Faz com que a RN fique mais bobusta para aprender padrões relevantes
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Dropout(0.2),
    layers.BatchNormalization(),

    # Mais camadas para a RN trabalhar
    layers.Conv2D(64, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Dropout(0.2),
    layers.BatchNormalization(),
    layers.Conv2D(128, 3, padding='same', activation='relu'),
    layers.Dropout(0.2),
    layers.BatchNormalization(),

    # Faz o nivelamento dos dados
    layers.Flatten(),
    layers.Dropout(0.2),

    # Camda densa, especificando a quantidade de neuronios, decrescendo até aproximar a quantidade de
    # classes nesse modelo (100) / a função constraint normaliza os dados evitando o overfitting, por isso ele
    # vem antes.
    layers.Dense(512, activation='relu'),
    layers.Dense(num_classes),
    layers.Dropout(0.2),
    layers.BatchNormalization(),

    layers.Dense(256, activation='relu'),
    layers.Dense(num_classes),
    layers.Dropout(0.2),
    layers.BatchNormalization(),

    layers.Dense(128, activation='relu'),
    layers.Dense(num_classes),
    layers.Dropout(0.2),
    layers.BatchNormalization()

])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
# """
# Treino
epochs = 50
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs
)
# """

# model.summary()

"""
test_image = image.load_img('Teste/Martelo.jpg', target_size=(img_height, img_height))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis=0)
# result = model.predict(test_image)
# train_ds.class_indices

predictions = model.predict(test_image)
score = tf.nn.softmax(predictions[0])

print(
    "This image most likely belongs to {} with a {:.2f} percent confidence."
        .format(class_names[np.argmax(score)], 100 * np.max(score))
)
"""
