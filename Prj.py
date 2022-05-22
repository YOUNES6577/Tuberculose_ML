#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 20 18:41:26 2022
@author: anes
"""

import matplotlib.pyplot as plt
import seaborn as sns

import keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Dropout
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
import cv2
import os
import numpy as np
import splitfolders
from glob import glob
from shutil import copyfile
import pandas as pd
#%
#%% Step 2:- Loading the data
dataframe = pd.read_csv('jaypee_metadata.csv')
dataset = np.array(dataframe)

train_path=''
for dirname, _, filenames in os.walk('./image_'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
        

nrml_imgs=dataframe[dataframe['findings']=="False"].study_id.unique()
Tubs_imgs=dataframe[dataframe['findings']=="Tuberculosis"].study_id.unique() 

all_images= glob('image_/*.jpg')
print(len(all_images))


train_path='data/train'
test_path='data/test'

os.mkdir('./data')
os.mkdir(os.path.join('./',train_path))
os.mkdir(os.path.join('./',test_path))

os.mkdir(os.path.join('./'+train_path,'False'))
os.mkdir(os.path.join('./'+train_path,'Tuberculosis'))

os.mkdir(os.path.join('./'+test_path,'False'))
os.mkdir(os.path.join('./'+test_path,'Tuberculosis'))

print(len(os.listdir(train_path+'/False')),end=('\t'))
print(len(os.listdir(train_path+'/Tuberculosis')))
print(len(os.listdir(test_path+'/False')),end=('\t'))
print(len(os.listdir(test_path+'/Tuberculosis')))

for labels in all_images :
    img_name= labels.split('/')[1]
    if "TRAIN" in img_name:
        if img_name in nrml_imgs :
            copyfile(labels,os.path.join(train_path+'/False',img_name))
        elif img_name in Tubs_imgs:
            copyfile(labels,os.path.join(train_path+'/Tuberculosis',img_name))
    elif "TEST" in img_name :
        if img_name in nrml_imgs :
            copyfile(labels,os.path.join(test_path+'/False',img_name))
        elif img_name in Tubs_imgs:
            copyfile(labels,os.path.join(test_path+'/Tuberculosis',img_name))

labels = dataframe['findings'].unique().tolist()
img_size = 256

def get_data(data_dir):
    data = []
    for label in labels:
        path = os.path.join(data_dir, label)
        class_num = labels.index(label)
        for img in os.listdir(path):
            try:
                img_arr = cv2.imread(os.path.join(path, img))[..., ::-1]  # convert BGR to RGB format
                resized_arr = cv2.resize(img_arr, (img_size, img_size))  # Reshaping images to preferred size
                data.append([resized_arr, class_num])
            except Exception as e:
                print(e)
    return np.array(data)


splitfolders.ratio("data/train", output = "data/Train_val", seed = 823, ratio = (.8, .2))

train_path = "data/Train_val/train"
val_path = "data/Train_val/val"
test_path = "data/test"

train = get_data(train_path)
val = get_data(val_path)
test=get_data(test_path)

#%%Step 3:- Visualize the data
train_list = []
for i in train:
    if i[1] == 0:
        train_list.append("False")
    else:
        train_list.append("Tuberculosis")


test_list = []
for i in val:
    if i[1] == 0:
        test_list.append("False")
    else:
        test_list.append("Tuberculosis")


sns.set_style("darkgrid")
sns.countplot(train_list)
sns.set_style("darkgrid")
sns.countplot(test_list)

plt.figure(figsize=(5, 5))
plt.imshow(train[1][0])
plt.title(labels[train[0][1]])

plt.figure(figsize=(5, 5))
plt.imshow(train[-1][0])
plt.title(labels[train[-1][1]])
#%%Step 4:- Data Preprocessing and Data Augmentation

x_train = []
y_train = []
x_val = []
y_val = []
x_test=[]
y_test=[]

for feature, label in train:
    x_train.append(feature)
    y_train.append(label)

for feature, label in val:
    x_val.append(feature)
    y_val.append(label)

for feature, label in test:
        x_test.append(feature)
        y_test.append(label)

# Normalize the data
x_train = np.array(x_train) / 255
x_val = np.array(x_val) / 255
x_test=np.array(x_test) / 255

x_train.reshape(-1, img_size, img_size, 1)
y_train = np.array(y_train)

x_val.reshape(-1, img_size, img_size, 1)
y_val = np.array(y_val)

x_test.reshape(-1, img_size, img_size, 1)
y_test=np.array(y_test)

datagen = ImageDataGenerator(
    rescale=(1 / 255),
    featurewise_center=False,  # set input mean to 0 over the dataset
    samplewise_center=False,  # set each sample mean to 0
    featurewise_std_normalization=False,  # divide inputs by std of the dataset
    samplewise_std_normalization=False,  # divide each input by its std
    zca_whitening=False,  # apply ZCA whitening
    rotation_range=30,  # randomly rotate images in the range (degrees, 0 to 180)
    zoom_range=0.2,  # Randomly zoom image
    width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
    height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
    horizontal_flip=True,  # randomly flip images
    vertical_flip=False,
)  # randomly flip images

datagen.fit(x_train)
#%% The Art of Transfer Learning MobileNetV2
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(256, 256, 3), include_top=False, weights="imagenet"
)
base_model.trainable = True
MobileNetV2 = tf.keras.Sequential(
    [
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(units = 4, kernel_initializer = 'uniform', activation = 'relu'),
        tf.keras.layers.Dense(1, kernel_regularizer=tf.keras.regularizers.l2(0.01), activation="linear"),
    ]
)
#base_model.summary()
base_learning_rate = 0.001

MobileNetV2.compile(
    optimizer=tf.keras.optimizers.Adam(lr=base_learning_rate),
    loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
    metrics=["accuracy"],
)

history_MobileNetV2 = MobileNetV2.fit(
    x_train, y_train, epochs=20, validation_data=(x_val, y_val)
)
#%%
model=Sequential()
model.add(tf.keras.applications.MobileNetV2(weights='imagenet',input_shape=(256,256,3),include_top=False))
model.add(tf.keras.layers.GlobalAveragePooling2D())
model.add(Dense(4096,activation='relu',name='fc1'))
model.add(Dense(4096,activation='relu',name='fc2'))
model.add(Dense(1000,activation='relu',name='fc3'))
model.add(Dropout(0.5))
model.add(Dense(1, kernel_regularizer=tf.keras.regularizers.l2(0.01), activation="linear"))

model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.001),
loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
metrics=["accuracy"])

model.summary()

svmhisto=model.fit(x_train,y_train, epochs=20,validation_data=(x_val,y_val))

#%%Save Model 
MobileNetV2.save('MobileNetV2.h5')

#%% Step 6:- Evaluating the result

acc = svmhisto.history["accuracy"]
val_acc = svmhisto.history["val_accuracy"]
loss = svmhisto.history["loss"]
val_loss = svmhisto.history["val_loss"]

epochs_range = range(20)

plt.figure(figsize=(15, 15))
plt.subplot(2, 2, 1)
plt.plot(epochs_range, acc, label="Training Accuracy")
plt.plot(epochs_range, val_acc, label="Validation Accuracy")
plt.legend(loc="lower right")
plt.title("Training and Validation Accuracy")

plt.subplot(2, 2, 2)
plt.plot(epochs_range, loss, label="Training Loss")
plt.plot(epochs_range, val_loss, label="Validation Loss")
plt.legend(loc="upper right")
plt.title("Training and Validation Loss")
plt.show()

pd.DataFrame(svmhisto.history).plot(figsize=(5,5))
plt.title('Pre-trained Trainnig Performance')
plt.xlabel('Epohcs')
plt.xlabel('metrics')
plt.show()
#%% Prediction

predictions = model.predict_classes(x_test)
predictions = predictions.reshape(1, -1)[0]
print(
    classification_report(
        y_test, predictions, target_names=["False (Class 0)", "Tubs (Class 1)"]
    )
)
