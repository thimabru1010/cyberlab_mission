#!/usr/bin/env python
# coding: utf-8

# In[1]:


from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator, array_to_img
import matplotlib.pyplot as plt
import numpy as np
import json
import random
import argparse
callbacks = []


# In[ ]:


# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--modelName",
    help="Model's name output", type=str, default='airplanes_classification.model')
ap.add_argument("-hm", "--historyModel",
    help="History's name output", type=str, default='history.json')
args = vars(ap.parse_args())


# In[2]:


model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=(150, 150,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

model.add(Conv2D(128, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

model.add(Flatten())
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam',metrics=['accuracy'])
model.summary()


# In[7]:


batch_size = 20

aug2=ImageDataGenerator(rescale=1./255, rotation_range=30, 
                    horizontal_flip=True, 
                    zoom_range=0.3)
       
aug = ImageDataGenerator(rescale=1./255)

train_folder = '/home/thiago/GitHub/deep_learning/estagio/dataset/train'
test_folder = '/home/thiago/GitHub/deep_learning/estagio/dataset/test'

train_generator = aug2.flow_from_directory(train_folder,                                                    target_size=(150, 150),                                                    batch_size=batch_size,                                                    class_mode='binary', shuffle=True, seed=42)

test_generator = aug.flow_from_directory(test_folder,                                                  target_size=(150, 150),                                                  batch_size=batch_size,                                                  class_mode='binary', shuffle=True, seed=42)


# In[5]:


print(train_generator.class_indices)


# In[4]:


epochs     = 50
history = model.fit_generator(train_generator,                    steps_per_epoch=30,                    epochs=epochs,                    validation_data=test_generator,                    validation_steps=10,                    verbose=1,callbacks=callbacks)


# In[1]:


#2 está com suffe and seed

# Saving model
model.save(args['modelName'])

# Saving history
with open(args['historyModel'], 'w') as f:
    json.dump(history.history, f)

