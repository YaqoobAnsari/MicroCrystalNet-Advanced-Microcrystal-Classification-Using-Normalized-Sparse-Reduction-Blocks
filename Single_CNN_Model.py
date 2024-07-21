# -*- coding: utf-8 -*-
"""
Created on Sat Nov 18 16:46:29 2023

@author: User
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Nov  5 11:06:47 2023

@author: User
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Sep 20 12:46:28 2023

@author: User
"""

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import os
from tensorflow.keras.applications import ResNet50, EfficientNetB0
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Concatenate, Dense, Dropout, GlobalAveragePooling2D, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import confusion_matrix, roc_curve, auc
import seaborn as sns
from sklearn.metrics import precision_recall_curve, average_precision_score
from sklearn.preprocessing import label_binarize
from datetime import datetime 
from sklearn.manifold import TSNE  
import pandas as pd

# Paths
train_dir = "../Train Data/"
val_dir = "../Validation Data/"
test_dir = "../Test Data/"

# Hyperparameters
BATCH_SIZE = 32
EPOCHS = 12
LEARNING_RATE = 1e-4
LR_PATIENCE = 4  # Number of epochs with no improvement to decrease learning rate
LR_FACTOR = 0.9  # Factor by which the learning rate will be reduced
current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')


# Custom metric for top-2 accuracy
def top_2_accuracy(y_true, y_pred):
    return tf.keras.metrics.top_k_categorical_accuracy(y_true, y_pred, k=2)

def recall_m(y_true, y_pred):
    true_positives = tf.keras.backend.sum(tf.keras.backend.round(tf.keras.backend.clip(y_true * y_pred, 0, 1)))
    possible_positives = tf.keras.backend.sum(tf.keras.backend.round(tf.keras.backend.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + tf.keras.backend.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = tf.keras.backend.sum(tf.keras.backend.round(tf.keras.backend.clip(y_true * y_pred, 0, 1)))
    predicted_positives = tf.keras.backend.sum(tf.keras.backend.round(tf.keras.backend.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + tf.keras.backend.epsilon())
    return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2 * ((precision * recall) / (precision + recall + tf.keras.backend.epsilon()))

    
# Data Generators
'''
train_datagen = ImageDataGenerator(rescale=1./255, rotation_range=20, zoom_range=0.2, width_shift_range=0.2,
                                   height_shift_range=0.2, horizontal_flip=True, brightness_range=[0.7, 1.3],
                                   shear_range=0.2, fill_mode='nearest')
'''

# Data Generators (without data augmentation)
train_datagen = ImageDataGenerator(rescale=1.0 / 255)
val_datagen = ImageDataGenerator(rescale=1.0 / 255)
test_datagen = ImageDataGenerator(rescale=1.0 / 255)

train_generator = train_datagen.flow_from_directory(train_dir, target_size=(64, 64), batch_size=BATCH_SIZE, class_mode='categorical', color_mode='grayscale')
val_generator = val_datagen.flow_from_directory(val_dir, target_size=(64, 64), batch_size=BATCH_SIZE, class_mode='categorical', color_mode='grayscale')
test_generator = test_datagen.flow_from_directory(test_dir, target_size=(64, 64), batch_size=BATCH_SIZE, class_mode='categorical', color_mode='grayscale', shuffle=False)



for i in range (1): 
    # Create the directory
    resdir_name = "../Model Results/"
    if not os.path.exists(resdir_name):
        os.makedirs(resdir_name)
        
    
    dir_name = resdir_name + f"CNN Model_complex_2_{current_time}_{i}"  
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    
    model_name = f"CNN_{current_time}_run1"
    
    # Updated paths to save weights, summary, plots, and test results
    weights_save_path = os.path.join(dir_name, f'{model_name}_{i}_weights.h5')
    model_save_path = os.path.join(dir_name, f'{model_name}.h5')
    summary_save_path = os.path.join(dir_name, f'{model_name}_{i}_summary.txt')
    training_plot_save_path = os.path.join(dir_name, f'{model_name}_{i}_training_plot.png')
    test_results_save_path = os.path.join(dir_name, 'test_results_{i}.csv')
    prediction_save_path = os.path.join(dir_name, f"{model_name}_{i}_test_predictions.npy")
    
    # Define the input shape for grayscale images (64x64)
    input_shape = (64, 64, 1)
    
    # Model Building
    input_tensor = Input(shape=input_shape)
    
    # Convolutional layers
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(input_tensor)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2))(x)
    
    # Flatten the feature map
    x = Flatten()(x)
    
    # Fully connected layers
    x = Dense(256, activation='relu')(x) 
    output_tensor = Dense(4, activation='softmax')(x)  # Assuming 4 classes for your task
    
    # Create the model
    model = Model(inputs=input_tensor, outputs=output_tensor)
        
     # Define a callback to save the best weights during training
    checkpoint = ModelCheckpoint("best_weights.h5", save_best_only=True, save_weights_only=True, monitor='val_loss', mode='min', verbose=1)
    
    # Define a callback to reduce learning rate on a plateau
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=LR_FACTOR, patience=LR_PATIENCE, verbose=1)
    
    # Define a callback for early stopping
    early_stopping = EarlyStopping(monitor='val_loss', patience=LR_PATIENCE, verbose=1)
    
    # Compile the model
    model.compile(optimizer=Adam(learning_rate=LEARNING_RATE),
                  loss='categorical_crossentropy',
                  metrics=['accuracy', top_2_accuracy, f1_m, precision_m, recall_m])
    
    
    # Print the model summary
    model.summary() 
    # Save the model summary to a text file
    with open(summary_save_path, 'w') as f:
        model.summary(print_fn=lambda x: f.write(x + '\n'))
        
    
    start_time = datetime.now().strftime('%H-%M-%S')
        
    # Train the model
    model_hist = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=EPOCHS,
        callbacks=[checkpoint, reduce_lr, early_stopping]
    )
        
    # Save the best weights
    model.save_weights(weights_save_path)
    
    # Save the model
    model.save(model_save_path)
    
    
    end_time = datetime.now().strftime('%H-%M-%S')
        
    # Plotting
    all_history = {}
    all_history['loss'] = model_hist.history['loss']  
    all_history['val_loss'] = model_hist.history['val_loss'] 
    all_history['accuracy'] = model_hist.history['accuracy'] 
    all_history['val_accuracy'] = model_hist.history['val_accuracy']  
    all_history['top_2_accuracy'] = model_hist.history['top_2_accuracy']  
    all_history['val_top_2_accuracy'] = model_hist.history['val_top_2_accuracy'] 
    
    # Extract a batch from the generator
    images, labels = next(train_generator)

    # Get class indices
    class_indices = train_generator.class_indices

    # Invert the class_indices dictionary to get the mapping from index to class names
    idx_to_class = {v: k for k, v in class_indices.items()}
    # Number of classes (assuming 4 in your case)
    # Dictionary to hold one image per class
    class_images = {}

    # Iterate over the batch and select one image per class
    for img, label in zip(images, labels):
        label_idx = np.argmax(label)  # Convert one-hot encoding to index
        class_name = idx_to_class[label_idx]
        if class_name not in class_images:
            class_images[class_name] = img
        if len(class_images) == len(class_indices):  # Stop when we have one image per class
            break

    num_classes = len(class_indices)
     
    # Create a subplot
    plt.figure(figsize=(12, 3))  # Adjust the figure size as per your page width

    # Plot each class image
    for i, (class_name, img) in enumerate(class_images.items()):
        ax = plt.subplot(1, num_classes, i + 1)
        ax.imshow(img[:, :, 0], cmap='gray')  # Assuming grayscale images
        ax.set_title(class_name)
        ax.axis('off')

    # Set the main title for the subplot
    plt.suptitle("Training data for different microcrystal forms")

    # Adjust layout to ensure there's a gap between images
    plt.tight_layout()
    plt.subplots_adjust(top=0.85)  # Adjust top spacing to accommodate the main title

    # Save the figure to the specified directory
    figure_save_path = os.path.join(dir_name, 'training_data_visualization.png')
    plt.savefig(figure_save_path)
     
    # Plot Loss and Accuracy 
    plt.figure(figsize=(14, 6))
    plt.subplot(1, 2, 1)
    plt.plot(all_history['loss'], label='Train Loss')   
    plt.plot(all_history['val_loss'], label='Validation Loss')   
    plt.legend()
    plt.title('Loss Evolution')
    plt.xlabel('Epochs')  # Add x-axis label
    plt.ylabel('Loss')    # Add y-axis label
    
    plt.subplot(1, 2, 2)
    plt.plot(all_history['accuracy'], label='Train Accuracy')   
    plt.plot(all_history['val_accuracy'], label='Validation Accuracy')   
    plt.legend()
    plt.title('Accuracy Evolution')
    plt.xlabel('Epochs')  # Add x-axis label
    plt.ylabel('Accuracy')  # Add y-axis label
    
    plt.tight_layout()
    plt.savefig(os.path.join(dir_name, f"hybrid_training_plot_{model_name}_{i}.png"))

    
    
    # Load the best weights and evaluate
    model.load_weights("best_weights.h5")
    test_loss, test_accuracy, test_top2_accuracy, f1_score, precision, recall = model.evaluate(test_generator)
    
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Top-1 Accuracy: {test_accuracy:.4f}")
    print(f"Test Top-2 Accuracy: {test_top2_accuracy:.4f}")
    print(f"Test F1-Score: {f1_score:.4f}")
    print(f"Test Precision: {precision:.4f}")
    print(f"Test Recall: {recall:.4f}")
    
    # Save test scores
    predictions = model.predict(test_generator)
    np.save(prediction_save_path, predictions)
    
    
    # Assuming all other variables like `test_loss`, `test_accuracy`, etc., are already defined in your script

    model_info = {
        'Model Name': model_name,
        'Learning Rate': LEARNING_RATE,
        'Batch Size': BATCH_SIZE,
        'Start Time': start_time,
        'End Time': end_time,
        'Test Loss': test_loss,
        'Test Accuracy': test_accuracy,
        'Test Top-2 Accuracy': test_top2_accuracy,
        'Test F1-Score': f1_score,
        'Test Precision': precision,
        'Test Recall': recall
    }
    
    # Convert to DataFrame
    model_results_df = pd.DataFrame([model_info])
    
    # Define the path to save the CSV
    csv_save_path = os.path.join(dir_name, f'{model_name}_test_results.csv')
    
    # Save as CSV
    model_results_df.to_csv(csv_save_path, index=False)

    # Get true labels and predicted labels
    true_labels = test_generator.classes
    predicted_labels = np.argmax(predictions, axis=1)
    
     
    # Run t-SNE
    tsne = TSNE(n_components=2, random_state=42)
    tsne_results = tsne.fit_transform(predictions)
    
    # Save t-SNE embeddings
    np.save(os.path.join(dir_name, "tsne_embeddings.npy"), tsne_results)
    
    # Plot and save t-SNE 2D scatter plot
    plt.figure(figsize=(10, 8))
    sns.scatterplot(x=tsne_results[:, 0], y=tsne_results[:, 1], hue=true_labels, palette="viridis", legend='full')
    plt.title('t-SNE projection of the test data')
    plt.xlabel('t-SNE axis 1')
    plt.ylabel('t-SNE axis 2')
    plt.legend()
    plt.savefig(os.path.join(dir_name, "t-sne_2d_scatter.png"))
    plt.show()
    
    # Plot and save t-SNE 2D kernel density estimate plot
    plt.figure(figsize=(10, 8))
    sns.kdeplot(x=tsne_results[:, 0], y=tsne_results[:, 1], cmap="viridis")
    plt.title('t-SNE projection with Kernel Density Estimate')
    plt.xlabel('t-SNE axis 1')
    plt.ylabel('t-SNE axis 2')
    plt.savefig(os.path.join(dir_name, "t-sne_2d_kde.png"))
    plt.show()
    
    # Confusion matrix
    cm = confusion_matrix(true_labels, predicted_labels)
    plt.figure(figsize=(12, 12))
    sns.heatmap(cm, annot=True, fmt='g')
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.savefig(os.path.join(dir_name, f"{model_name}_{i}_confusion_matrix.png"))

    
    # ROC Curve
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    n_classes = 4
    y_test_bin = tf.keras.utils.to_categorical(true_labels, n_classes)
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], predictions[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    class_names = ["polyhedral", "amorphous", "rhombic", "spherical"]
    plt.figure(figsize=(10, 8))
    for i in range(n_classes):
        plt.plot(fpr[i], tpr[i], label=f'ROC curve of class {class_names[i]} (area = {roc_auc[i]:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic for multi-class data')
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(dir_name, f"{model_name}_{i}_roc_curve.png"))

    
    # Calculate precision and recall for each class
    y_true = label_binarize(test_generator.classes, classes=[0, 1, 2, 3])
    y_pred = model.predict(test_generator)
    
    precision = dict()
    recall = dict()
    average_precision = dict()
    n_classes = 4
    
    for i in range(n_classes):
        precision[i], recall[i], _ = precision_recall_curve(y_true[:, i], y_pred[:, i])
        average_precision[i] = average_precision_score(y_true[:, i], y_pred[:, i])
    
    # Plot the precision-recall curves for each class
    class_names = ["polyhedral", "amorphous", "rhombic", "spherical"]
    plt.figure(figsize=(10, 8))
    for i in range(n_classes):
        plt.plot(recall[i], precision[i], marker='.', label=f'Class {class_names[i]} (AP = {average_precision[i]:.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve for Each Class')
    plt.legend(loc='best')
    
    # Save the precision-recall curves
    plt.savefig(os.path.join(dir_name, f"{model_name}_{i}_precision_recall_curve.png"))

