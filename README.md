# Binary Classification of sclerotic and non-sclerotic glomeruli

## Intructions to run the code

1. Use [this](https://colab.research.google.com/drive/1rOaK4RgaarKPHuXXddgikapPMHxJwhzV?usp=sharing) collab notebook to run the code step by step
2. Or use the image_classification.ipynb file in this repository
3. Edit the paths for the dataset in the notebook or .ipynb file to continue the execution for splitting the data into test, train, and validation sets
4. If you don't want to train the model. Skip the training phase and evaluate the model by preparing the test generation data and loading the pre trained weights provided in this repo(model_weights_cnn.h5 file).

## Methodology used

### Preprocessing Phase
Firstly it was checked how many images of each class were present in the dataset provided. Post that the data was split into train set, validation set and test sets in a proportion of 0.6, 0.2 and 0.2 respectively. A random spit of data is performed and then 3 directiories for each set(train, test, validation) were created. Within this directories, sub directories were created for each class of images.

### Model Development phase
CNNs were chosed as the model to be trained due to their widespread adoption and better results for image classification tasks. The model was developed using keras sequencial model architecture whre each image was resized to 128*128 for performing the convolutional operations. The layer of the model include 2 sets of convolutional layers + max pooling layers, with relu activations followed by a flattening layer and dense layers. The data in the train set is slightly imbalanced with 80% of images comprising of non-sclerotic glomeruli and for that reason, augmentation operations were added in the training phase for sclerotic glomeruli data. The model has been trained for 5 epochs.

### Metrics
1. One of the metrics chosen for the evaluation are accuaracy and the overall test accuracy obtained was close to 90%.
2. A confusion matrix was created for the visualization of predicted classes and true classes from which it has been obsereved that the number of False negatives and false positives were huge and the model is underporforming in classifying the sclerotic glomeruli images. Some of the ways to improve this metrics further is by resizing the images to a bmuch better resolution than 128x128 so that the model is able to distinguish the images well and also add more samples of sclerotic glomeruli as the augmentations could have resulted in the model to learn redundant patterns and not being able to capture the complex variations.

Link to model weights - [file](https://drive.google.com/file/d/1Fwv-8R9nwqnA_xhGUFhA-EAAqmBRdjn2/view?usp=sharing)

### Evaluation
A script evaluation.py was created that accepts the folder to test images as an argument and also the model file needs to be in the same directory as of the script with file name 'model_weights_cnn.h5'. The output csv file contains the image file name and the predicted class.
