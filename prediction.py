#!/usr/bin/env python
# coding: utf-8

# In[8]:


from keras.models import load_model
from keras.preprocessing import image
import matplotlib.pyplot as plt
import numpy as np
import json
import argparse


# In[ ]:


# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--modelName", required=True,
    help="Model's name")
ap.add_argument("-hm", "--historyModel", required=True,
    help="History's name json file")
ap.add_argument("-p", "--predictionSingle",
    help="Image path of a single prediction", type=str, default='test.jpg')
ap.add_argument("-f", "--predictionFolder",
    help="If wants to run all the predictions from prediction folder",type=int, default=1)

args = vars(ap.parse_args())


# In[5]:


model = load_model(args['modelName'])


# In[6]:


model.summary()


# In[9]:


if args['predictionFolder'] == 1:
    img_path = 'dataset/prediction'
    for i in range(34):
        img_path_azul = img_path + '/azul'+str(i)+'.jpg'
        test_image = image.load_img(img_path_azul, target_size = (150, 150))
        test_image = image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis = 0)
        result = model.predict(test_image)
        if result[0][0] == 1:
            prediction = 'gol'
        else:
            prediction = 'azul'
        print(result[0][0], prediction)

    print('\n')

    for i in range(2,36):
        img_path_gol = img_path + '/gol'+str(i)+'.jpg'
        test_image = image.load_img(img_path_gol, target_size = (150, 150))
        test_image = image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis = 0)
        result = model.predict(test_image)
        if result[0][0] == 1:
            prediction = 'gol'
        else:
            prediction = 'azul'
        print(result[0][0], prediction)
elif args['predictionFolder'] == 0:
    test_image = image.load_img(args['predictionSingle'], target_size = (150, 150))
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis = 0)
    result = model.predict(test_image)
    if result[0][0] == 1:
        prediction = 'gol'
    else:
        prediction = 'azul'
    print(result[0][0], prediction)
else:
    print("Wrong input")
    


# In[12]:


with open(args['historyModel'], 'r') as f:
    history = json.load(f)


# In[14]:


plt.figure(figsize=(15,5))
plt.subplot(121)
plt.plot(history['val_loss'])
plt.plot(history['loss'])
plt.ylabel('Loss')
plt.xlabel('epoch')
plt.legend(['Val','Train'], loc='upper left')

plt.subplot(122)
plt.plot(history['val_acc'])
plt.plot(history['acc'])
plt.ylabel('Acc')
plt.xlabel('epoch')
plt.legend(['Val','Train'], loc='upper left')
plt.show()

