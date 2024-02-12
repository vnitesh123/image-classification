

import os
import csv
import numpy as np
import tensorflow as tf
from keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import Sequence
from tensorflow.keras.preprocessing import image

# Load the trained model
model = Sequential([
    Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(128, 128, 3)),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, kernel_size=(3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(1, activation='sigmoid'),
])

model.load_weights('model_weights_cnn.h5')
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Path to the folder containing test images
test_data_folder = input("Enter the path to the folder containing test images: ")

# Path to save the CSV file
output_csv_file = 'evaluation.csv'

# List to store predictions
predictions = []

for filename in os.listdir(test_data_folder):
    img_path = os.path.join(test_data_folder, filename)

    # resize the image to 128x128
    img = image.load_img(img_path, target_size=(128, 128))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0

    # Performing prediction
    prediction = model.predict(img_array)
    predicted_class = 1 if prediction > 0.5 else 0

    # Appending filename and predicted class to predictions list
    predictions.append((filename, predicted_class))

# Writing predictions to CSV file
with open(output_csv_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Filename', 'Predicted Class'])
    writer.writerows(predictions)

print("Predictions saved to", output_csv_file)
