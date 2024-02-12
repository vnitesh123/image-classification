

import os
import csv
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model

# Load model
model = load_model('model.h5')

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

# Write predictions to CSV file
with open(output_csv_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Filename', 'Predicted Class'])
    writer.writerows(predictions)

print("Predictions saved to", output_csv_file)

