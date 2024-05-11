import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout, BatchNormalization, Conv2D, MaxPooling2D, AveragePooling2D 
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
from tensorflow.keras.callbacks import ReduceLROnPlateau
from Segmentation_FeaturesExtract import Segmentation, FeaturesExtraction
from sklearn.metrics import confusion_matrix, roc_curve, auc
from scipy import interp
from itertools import cycle
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

#Read the normalized features data
train_path = 'normalized_data.csv'
train_dataset_df = pd.read_csv(train_path)

#Define the column names which represent the entity of the dataset
column_names = ['threshold',
                'zero-crossing_rate',
                'mean_mfcc', 'std_mfcc',
                'spectral_centroid',
                'spectral_rolloff',
                'spectral_flux',
                'mean_frequency_domain_real','std_frequency_domain',
                'energy_entropy',
                'sound_type']

#Extract the features names
feature_names = column_names[:-1]
#Extract the label name
label_name = column_names[-1]

#Extract the features data and convert to float32
train_features_dataset = train_dataset_df.drop(label_name, axis=1)
train_features_dataset = np.asarray(train_features_dataset).astype('float32')

#Extract the label data and convert to int32
train_label_dataset = train_dataset_df.drop(feature_names, axis=1)
train_label_dataset = np.asarray(train_label_dataset).astype('int32')

#Split the features data into 90% of training data and 10% of validate data
X_train, X_test, y_train_0, y_test_0 = train_test_split(train_features_dataset, train_label_dataset, random_state = 3, test_size = 0.10)

#Change the category of the label of train and test dataset into 4 classes
y_train = to_categorical(y_train_0, num_classes = 4)
y_test = to_categorical(y_test_0, num_classes = 4)

#reshape the feature data into (1,5,2) dimension
#the values of the shapeed represent height, width, depth
train_features_dataset = train_features_dataset.reshape(train_features_dataset.shape[0], 1, 5, 2)
train_label_dataset = to_categorical(train_label_dataset, num_classes = 4)

#Split the 90% of train data into 90% of training data and 10% of testing data 
X_train, X_validate, y_train, y_validate = train_test_split(X_train, y_train, random_state = 0, test_size = 0.10)

#reshape the dataset of training, validate, and testing into (1,5,2) dimension
#the values of the shapeed represent height, width, depth
X_train = X_train.reshape(X_train.shape[0], 1, 5, 2)
X_test = X_test.reshape(X_test.shape[0], 1, 5, 2)
X_validate = X_validate.reshape(X_validate.shape[0], 1, 5, 2)

#Build the 2D convolutional neural network model
#Initialize the sequential
model = Sequential()
#Define the input layer
model.add(Conv2D(32, (2,2), activation='relu', input_shape = (1, 5, 2), data_format = 'channels_first'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(1, 1)))
model.add(Dropout(0.2))
#Define the first hidden layer
model.add(Conv2D(12, (2,2), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(1, 1)))
model.add(Dropout(0.2))
#Define the second hidden layer
model.add(Conv2D(6, (2,2), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(1, 1)))
model.add(Dropout(0.2))
#Define the third hidden layer
model.add(Conv2D(6, (2,2), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(1, 1)))
model.add(Dropout(0.2))
#Define the average pooling layer
model.add(AveragePooling2D(pool_size=(1,1)))
#Define the flatten layer
model.add(Flatten())
#Define the first dense layer
model.add(Dense(16, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.2))
#Define the second dense layer
model.add(Dense(12, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.2))
#Define the third dense layer
model.add(Dense(6, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.2))
#Define the final dense layer
model.add(Dense(4, activation='softmax'))
#Display the summary of the 2D convolution neural network 
model.summary()

#Define the optimizer
optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
#Add the optimizer to model
model.compile(optimizer=optimizer, loss = "categorical_crossentropy", metrics=['accuracy'])
#Start the training 
history = model.fit(X_train, y_train, batch_size = 600, epochs=10000, validation_data=(X_validate, y_validate))
#Evaluate the model with test dataset
loss, accuracy = model.evaluate(X_test, y_test, verbose=1)
#Evaluate the model with validate dataset
loss_v, accuracy_v = model.evaluate(X_validate, y_validate, verbose=1)
#Display the accuracy and the loss rate of validate and test dataset
print("Validation: accuracy = %f  ;  loss_v = %f" % (accuracy_v, loss_v))
print("Test: accuracy = %f  ;  loss = %f" % (accuracy, loss))
#Display the accuracy and the loss rate for overall training process
print("The training accuracy is :", history.history['accuracy'])
print("The training loss is :", history.history['loss'])

def plot_learningCurve(history, epoch):
  """
  Plot the learning curve of the 2D convolutional neural network

  Parameters
  ----------
  history: dictionary
      Record the training loss values and metrics values at successive epochs,
      as well as validation loss values and validation metrics values
  epoch: int
      The number of dataset that is passed forward and backward in the convolutional neural network  
  """
  # Plot training & validation accuracy values
  epoch_range = range(1, epoch+1)
  plt.plot(epoch_range, history.history['accuracy'])
  plt.plot(epoch_range, history.history['val_accuracy'])
  plt.title('Model accuracy')
  plt.ylabel('Accuracy')
  plt.xlabel('Epoch')
  plt.legend(['Train', 'Val'], loc='upper left')
  plt.show()

  # Plot training & validation loss values
  plt.plot(epoch_range, history.history['loss'])
  plt.plot(epoch_range, history.history['val_loss'])
  plt.title('Model loss')
  plt.ylabel('Loss')
  plt.xlabel('Epoch')
  plt.legend(['Train', 'Val'], loc='upper left')
  plt.show()

# Function to plot confusion matrix    
def plot_confusion_matrix(cm, classes, dataset_type, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`
    title is set to `Confusion matrix` by default

    Parameters
    ----------
    cm: array
      The record for the true positive, true negative, false negative, and false positive
    classes: int
      The number of classes
    normalize: boolean
      To indicate whether to normalize or not
    title: str
      The title of the plotted confusion matrix
    cmap: str
      The color of the confusion matrix
    """
    #Define the feature of the confusion matrix
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    #Normalize the data if it is true
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    #Divide the 
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    #Label the confusion matrix
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.title('Confusion Matrix for '+dataset_type)
    plt.show()


def plot_roc(n_classes, y_test, y_score, dataset_type):
  """
  Plot the receiver operating characteristic (ROC) curve

  Parameters
  ----------
  n_classes: int
    The number of classes
  y_test: array
    The true label of the heart sound class
  y_score: array
    The predicted label of teh heart sound class
  dataset_type: str
    The label for differenciate between validate and test dataset
  """
  #Plot linewidth
  lw = 2

  #Compute ROC curve and ROC area for each class
  fpr = dict()
  tpr = dict()
  roc_auc = dict()

  for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test[:,i], y_score[:,i])
    roc_auc[i] = auc(fpr[i], tpr[i])

  #Compute micro-average ROC curve and ROC area
  fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
  roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

  # Compute macro-average ROC curve and ROC area

  # First aggregate all false positive rates
  all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

  # Then interpolate all ROC curves at this points
  mean_tpr = np.zeros_like(all_fpr)
  for i in range(n_classes):
      mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

  # Finally average it and compute AUC
  mean_tpr /= n_classes

  fpr["macro"] = all_fpr
  tpr["macro"] = mean_tpr
  roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

  # Plot all ROC curves
  plt.figure(1)
  plt.plot(fpr["micro"], tpr["micro"],
           label='micro-average ROC curve (area = {0:0.2f})'
                 ''.format(roc_auc["micro"]),
           color='deeppink', linestyle=':', linewidth=4)

  plt.plot(fpr["macro"], tpr["macro"],
           label='macro-average ROC curve (area = {0:0.2f})'
                 ''.format(roc_auc["macro"]),
           color='navy', linestyle=':', linewidth=4)

  colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'red'])
  for i, color in zip(range(n_classes), colors):
      plt.plot(fpr[i], tpr[i], color=color, lw=lw,
               label='ROC curve of class {0} (area = {1:0.2f})'
               ''.format(i, roc_auc[i]))

  plt.plot([0, 1], [0, 1], 'k--', lw=lw)
  plt.xlim([0.0, 1.0])
  plt.ylim([0.0, 1.05])
  plt.xlabel('False Positive Rate')
  plt.ylabel('True Positive Rate')
  plt.title('Some extension of Receiver operating characteristic to multi-class ('+dataset_type+')')
  plt.legend(loc="lower right")
  plt.show()

#Plot the learning curve
plot_learningCurve(history, 10000)

# Predict the values from the validation dataset
Y_pred_validate = model.predict(X_validate)
# Predict the values from the test dataset
Y_pred_test = model.predict(X_test)
# Convert predictions classes to one hot vectors 
Y_pred_classes_validate = np.argmax(Y_pred_validate,axis = 1)
Y_pred_classes_test = np.argmax(Y_pred_test,axis = 1)
# Convert validation observations to one hot vectors
Y_true_validate = np.argmax(y_validate,axis = 1)
# Convert test observations to one hot vectors
Y_true_test = np.argmax(y_test,axis = 1) 
# compute the confusion matrix for validate dataset
confusion_mtx_validate = confusion_matrix(Y_true_validate, Y_pred_classes_validate)
# compute the confusion matrix for test set
confusion_mtx_test = confusion_matrix(Y_true_test, Y_pred_classes_test)

# plot the confusion matrix for validate dataset
plot_confusion_matrix(confusion_mtx_validate, classes = range(4), dataset_type="Validate dataset")
# plot the confusion matrix for test dataset
plot_confusion_matrix(confusion_mtx_test, classes = range(4), dataset_type="Test dataset")

# Plot the ROC curve for validation dataset
plot_roc(n_classes = 4, y_test= y_validate, y_score=Y_pred_validate, dataset_type="Validate dataset")
# Plot the ROC curve for test dataset
plot_roc(n_classes = 4, y_test= y_test, y_score=Y_pred_test, dataset_type="Test dataset")

#save the trained model to heartModel.h5
model.save("heartModel.h5")
