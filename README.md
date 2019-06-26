# cyberlab_mission
Construct a CNN network for cyberlab's mission

## Dataset

The first difficult of the process was to download all the data from the site www.airliners.net. So, I had to download all the images from the site inspecting the html page's code and discovering how to get a link with just the image I wanted. After this, is necessary to crop the image because there is some airliners' logo at the bottom of the image.

to run the program follow the steps below:

'''
python download_data.py --path --name --batch --iterations --startPage
'''

* --path: -p: path to download folder (required)

Path you will download the images. Make sure you created all the right folders.

* --name: -n: company's name: azul or gol (required)

* --batch: -b: number of photos to download if wanted less than 252 (default=252)

The connection with the site don't support more than 252 downloads. So, if you want to download more than you should download in batches. Make sure to use always multiples of 36. Because one page has 36 images.

* --iterations: -i: number of iterations to download per batch (default=1)

This argument goes along with number of batches. If you want to download less than 252 images don't get worried. Just update the argument if you want to download more than 252 images and make sure your batch is a multiple of 36.

Tip: To download the test/validation and training data remember to check the page where the training stopped (it will be displayed at the terminal).

## Model

To create the model I used a smaller version of VGG16Net architecture because it's a good classical architecture to solve a binary classifier problem.

'''
model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=(150, 150,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(1))
model.add(Activation('sigmoid'))
'''

As a characteristic of the model, I used just 3x3 filter in all the convolutional layers, a pooling of type Max (MaxPooling) to reduce the size of images and a fully connected (Dense) layer right after a flattening layer (used to transform in 1D vectors). After the fully connected we have a dropout to avoid overfitting of 0.5. The Output layer only have one node since we're using a binary classifier. As well, we're using 'sigmoid' function instead of 'softmax' as is used in a multiclass neural network.

To compile the model I used the gradient descendant optimizer ('adam') and binary loss as long as is a binary classification.

'''
model.compile(loss='binary_crossentropy', optimizer='adam',metrics=['accuracy'])
'''

### Training

To train the model I used 1000 images of each class approximated and 500 images of each class to validation. So, it's a total of 2000 images for training and 1000 images for validation. Then, I used data augmentation to enhance the model and prevent overfitting. To use data augmentation it is necessary to use 'keras.preprocessing.image' and create a ImageDataGenerator object.

'''
aug = ImageDataGenerator(rescale=1./255)

train_folder = '/home/thiago/GitHub/deep_learning/estagio/modules/dataset/train'
test_folder = '/home/thiago/GitHub/deep_learning/estagio/modules/dataset/test'

train_generator = aug.flow_from_directory(train_folder,                                                    target_size=(150, 150),                                                    batch_size=batch_size,                                                    class_mode='binary', shuffle=True, seed=42)

test_generator = aug.flow_from_directory(test_folder,                                                  target_size=(150, 150),                                                  batch_size=batch_size,                                                  class_mode='binary', shuffle=True, seed=42)
'''

As long we used data augmentation and so on ImageDataGenerator, we have to use the fit_generator function to train the model.

'''
epochs     = 20

history = model.fit_generator(train_generator,                    steps_per_epoch=30,                    epochs=epochs,                    validation_data=test_generator,                    validation_steps=10,                    verbose=1,callbacks=callbacks)
'''

To run the training model follow the steps bellow:

'''
python airplanes_classification_model.py --modelName --historyModel
'''

* --modelName: -m: Model's name output (default='airplanes_classification.model')

* --historyModel: -hm: History's name output (default='history.json')

------------------------------- Terminar

## Prediction

To predict the model I downloaded 70 images 35 of each class (azul and gol), different from the others.

If you run the model you will get the prediction for each of the 70 images with the name of the predicted company, as wanted. For the gol company the model output a value of 1.0 and 0.0 for azul ones. and the plot graph with loss and val_loss per epoch and accuary and val_acc per epoch. 

To run the program follow the steps bellow:

'''
python prediction.py --modelName --historyModel --predictionSingle --predictionFolder
'''

* --modelName: -m: Model's name input (default='airplanes.model')

* --historyModel: -hm: History's name json file (default='history.json')

* --predictionSingle: -p: Image path of a single prediction (default='test.jpg')

Change this argument to the single image you want to predict.

* --predictionFolder: -f: If wants to run all the predictions from prediction folder (default=1)

Change this argument to 0 if you do not want to predict all the 70 images and want to predict just a single image. Remember to input the name of image you want to predict.

--------------------------------------------------------------------------------------

Warning: Make sure everything is on the same folder!






