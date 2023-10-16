import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# 1. Dataset Creation:

# Create directories for dataset
if not os.path.exists('dataset'):
    os.mkdir('dataset')
    os.mkdir('dataset/circles')
    os.mkdir('dataset/squares')

num_samples = 100
img_size = (64, 64)

# Generate circle images
for i in range(num_samples):
    img = np.zeros((img_size[0], img_size[1], 3), dtype=np.uint8)
    cv2.circle(img, (img_size[0]//2, img_size[1]//2), img_size[0]//3, (0, 255, 0), -1)
    cv2.imwrite(f'dataset/circles/circle_{i}.png', img)

# Generate square images
for i in range(num_samples):
    img = np.zeros((img_size[0], img_size[1], 3), dtype=np.uint8)
    cv2.rectangle(img, (img_size[0]//4, img_size[1]//4), (3*img_size[0]//4, 3*img_size[1]//4), (255, 0, 0), -1)
    cv2.imwrite(f'dataset/squares/square_{i}.png', img)

# 2. Data Augmentation & Splitting:

datagen = ImageDataGenerator(
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2
)

dataset_path = 'dataset'
batch_size = 16
train_gen = datagen.flow_from_directory(
    dataset_path,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='binary',
    subset='training'
)
val_gen = datagen.flow_from_directory(
    dataset_path,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='binary',
    subset='validation'
)

# 3. Model Definition:

model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])
model.summary()

# 4. Train the Model:

epochs = 10
history = model.fit(
    train_gen,
    steps_per_epoch=train_gen.samples // train_gen.batch_size,
    epochs=epochs,
    validation_data=val_gen,
    validation_steps=val_gen.samples // val_gen.batch_size
)

# 5. Evaluate the Model:

val_loss, val_accuracy = model.evaluate(val_gen, steps=val_gen.samples // val_gen.batch_size)
print(f"Validation Accuracy: {val_accuracy * 100:.2f}%")
