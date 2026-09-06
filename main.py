import numpy as np
import pandas as pd
import os
import shutil
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Flatten, Dense, Dropout
from keras.applications import VGG16
import matplotlib.pyplot as plt
import random

# الفئات المطلوبة فقط
categories = ["Bikes", "Cars", "Motorcycles"]
main_dir = r"E:\car_tracking\Vehicle Image Classification\Vehicles"
split_dirs = ["train", "val", "test"]
base_split_dir = "vehicles_split"

# إنشاء مجلدات التقسيم
for split_dir in split_dirs:
    for category in categories:
        os.makedirs(os.path.join(base_split_dir, split_dir, category), exist_ok=True)

# تقسيم البيانات ونقل الصور
for category in categories:
    category_path = os.path.join(main_dir, category)
    images = os.listdir(category_path)
    train_images, temp_images = train_test_split(images, test_size=0.3, random_state=42)
    val_images, test_images = train_test_split(temp_images, test_size=0.5, random_state=42)

    def move_images(image_list, destination):
        for image in image_list:
            shutil.copy(os.path.join(category_path, image), os.path.join(base_split_dir, destination, category, image))

    move_images(train_images, "train")
    move_images(val_images, "val")
    move_images(test_images, "test")

# إعداد المولدات
train_datagen = ImageDataGenerator(rescale=1./255, rotation_range=20, width_shift_range=0.2,
                                   height_shift_range=0.2, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
val_test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(os.path.join(base_split_dir, "train"), target_size=(150, 150),
                                                    batch_size=32, class_mode='categorical')
val_generator = val_test_datagen.flow_from_directory(os.path.join(base_split_dir, "val"), target_size=(150, 150),
                                                     batch_size=32, class_mode='categorical')
test_generator = val_test_datagen.flow_from_directory(os.path.join(base_split_dir, "test"), target_size=(150, 150),
                                                      batch_size=32, class_mode='categorical')

# بناء النموذج
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(150, 150, 3))
base_model.trainable = False

model = Sequential([
    base_model,
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(3, activation='softmax')  # عدد الفئات = 3
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit(train_generator, epochs=20, validation_data=val_generator)

# تقييم
test_loss, test_acc = model.evaluate(test_generator)
print(f"Test accuracy: {test_acc * 100:.2f}% - main.py:71")

# Fine-tuning
base_model.trainable = True
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5), loss='categorical_crossentropy', metrics=['accuracy'])
history_fine = model.fit(train_generator, epochs=10, validation_data=val_generator)

test_loss, test_acc = model.evaluate(test_generator)
print(f"Test accuracy after finetuning: {test_acc * 100:.2f}% - main.py:79")

# عرض التنبؤات
test_images, test_labels = next(test_generator)
indices = random.sample(range(len(test_images)), 9)

predictions = model.predict(test_images[indices])
predicted_labels = np.argmax(predictions, axis=1)
true_labels = np.argmax(test_labels[indices], axis=1)

plt.figure(figsize=(10, 10))
for i, idx in enumerate(indices):
    plt.subplot(3, 3, i + 1)
    plt.imshow(test_images[idx])
    plt.axis('off')
    true_label = categories[true_labels[i]]
    predicted_label = categories[predicted_labels[i]]
    color = 'green' if true_label == predicted_label else 'red'
    plt.title(f"True: {true_label}\nPred: {predicted_label}", color=color)
plt.tight_layout()
plt.show()

model.save("vehicle_classifier.h5")
model.save("vehicle_model.keras")