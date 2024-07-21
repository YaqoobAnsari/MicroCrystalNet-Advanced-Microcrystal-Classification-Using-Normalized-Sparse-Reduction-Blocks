import tensorflow as tf
import pandas as pd
import cv2
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.utils import to_categorical
import pathlib
import cv2
import numpy as np
import os
import time
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, Activation
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from collections import Counter
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.metrics import Precision, Recall
from sklearn.metrics import roc_curve, auc
from tensorflow.keras.utils import plot_model, to_categorical
from contextlib import redirect_stdout


# Define the dictionary for class labels
crystal_labels_dict = {
    'rhombic': 0,
    'polyhedral': 1,
    'unclear': 2,
    'amorphous': 3,
    'spherical': 4
}

# Load data from CSV file
def load_data_from_csv(csv_file):
    df = pd.read_csv(csv_file)
    image_paths = df["Path"].values
    labels = df["Sec-class"].values
    return image_paths, labels

# Load images from the given paths
def load_images(image_paths, dim):
    images = [cv2.resize(cv2.imread(path), dim) for path in image_paths]
    return np.array(images) / 255.0

# Convert labels to numerical values
def labels_to_numbers(labels, labels_dict):
    return np.array([labels_dict[label] for label in labels])

# Load the model with the provided architecture
def load_trained_model(weights_path, dim):
    model = create_model(dim)
    model.load_weights(weights_path)
    return model

# Define the model architecture (same as the one you provided)

def create_model(dim):
    model = Sequential()
    
    # First Convolutional Block
    model.add(Conv2D(32, kernel_size=(3, 3), input_shape=(dim[0], dim[1], 3)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.45))
    
    # Second Convolutional Block
    model.add(Conv2D(64, (3, 3)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.45))
    
    # Third Convolutional Block
    model.add(Conv2D(128, (3, 3)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.45))
    
    # Fourth Convolutional Block (added)
    '''
    model.add(Conv2D(256, (3, 3)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    '''
    
    # Fully Connected Layers
    model.add(Flatten())
    model.add(Dense(512))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    
    model.add(Dense(5, activation='softmax'))

    model.compile(loss=tf.keras.losses.CategoricalCrossentropy(),
                  optimizer='adam',
                  metrics=['accuracy', Precision(), Recall(), tf.keras.metrics.AUC(name='auc')])
    
    return model

# Define image dimensions
dim = (75, 75)

# Load test data
print("Loading test data...")
test_image_paths, test_labels_text = load_data_from_csv('Augmented_Test.csv')
X_test = load_images(test_image_paths, dim)
y_test = labels_to_numbers(test_labels_text, crystal_labels_dict)
y_test = to_categorical(y_test, num_classes=5)
print(f"Loaded {len(X_test)} test samples.")

# Load the model with its saved weights
model_weights_path = "best_model_fold_0.h5"
print("Loading trained model...")
model = load_trained_model(model_weights_path, dim)

# Evaluate the model
print("Evaluating the model...")
loss, accuracy, precision, recall, auc = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Loss: {loss}")
print(f"Test Accuracy: {accuracy}")
print(f"Test Precision: {precision}")
print(f"Test Recall: {recall}")
print(f"Test AUC: {auc}")

# Generate predictions and detailed classification report
y_pred_prob = model.predict(X_test)
y_pred = np.argmax(y_pred_prob, axis=1)
y_test_labels = np.argmax(y_test, axis=1)

print("\nClassification Report:")
print(classification_report(y_test_labels, y_pred, target_names=list(crystal_labels_dict.keys())))

print("\nConfusion Matrix:")
cm = confusion_matrix(y_test_labels, y_pred)
print(cm)

print("\nFinished evaluating the model.")


# Clear GPU memory
tf.keras.backend.clear_session()
print("GPU memory cleared.")

