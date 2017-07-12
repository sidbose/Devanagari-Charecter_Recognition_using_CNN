# Importing the Keras libraries and packages
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator

# mini batch gradient decent
batch_size = 32
# number of predictable classes from 46
num_classes = 46
# epochs number
epochs = 30
# input image dimension
img_rows, img_cols = 32, 32
# number of samples
no_of_training_samples = 78200
no_of_test_samples = 13800

'''
Building the CNN model
'''

# Initialising the CNN
model = Sequential()

# step 1 - convolutional layer with rectified linear unit activation
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(img_rows, img_cols, 1)))

# step 2 - Poolong
model.add(MaxPooling2D(pool_size=(2, 2)))

# step 3 - Adding second convolutional layer to improve accuracy
model.add(Conv2D(64, activation='relu', kernel_size=(3, 3)))

# step 4 - second maxpooling layer
model.add(MaxPooling2D(pool_size=(2, 2)))

# step 5 - adding Dropout that helps prevent overfitting.
model.add(Dropout(0.5))

# step6 - Flattening
model.add(Flatten())

# step 7 - full connection
# hidden layer
model.add(Dense(units=640, activation='relu'))

# output layer
# output a softmax to squash the matrix into output probabilities
model.add(Dense(units=num_classes, activation='softmax'))

# step 8 - compiling the model
# Adaptive learning rate (adaDelta) is a popular form of gradient descent rivaled only by adam and adagrad
# categorical ce since we have multiple classes (46)
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

'''
Fitting the CNN to images
'''
train_datagen = ImageDataGenerator(rescale=1./255,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1./255)


'''
Preparing training and test dataset. We will be using handy flow_from_directory of keras
'''

training_set = train_datagen.flow_from_directory('/input/Train',
                                                 target_size=(img_rows, img_rows),
                                                 batch_size=batch_size,
                                                 class_mode='categorical',
                                                 color_mode='grayscale')

test_set = test_datagen.flow_from_directory('/input/Test',
                                            target_size=(img_rows, img_rows),
                                            batch_size=batch_size,
                                            class_mode='categorical',
                                            color_mode='grayscale')

'''
call back function to see training in tensorboard later
'''
tbCallBack = keras.callbacks.TensorBoard(log_dir='/output/logs', histogram_freq=10, write_graph=True)
history = model.fit_generator(training_set,
                              steps_per_epoch=(no_of_training_samples/batch_size),
                              nb_epoch=epochs,
                              validation_data=test_set,
                              validation_steps=batch_size,
                              callbacks=[tbCallBack])

print("Final Accuracy: %.2f%%" % (history.history['acc'][epochs-1]*100))


# Save the model
# serialize model to JSON
model_json = model.to_json()
with open("/output/model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("/output/model.h5")
print("Saved model... Hurray!!!")

# http://machinelearningmastery.com/display-deep-learning-model-training-history-in-keras/
# https://www.kaggle.com/rishianand/devanagari-character-set
# command for running the job in floydhub
# floyd run --gpu --data <data_id> "python script.py"
# floyd run --gpu --data ugPh9ghXNGanuQZVFdcyF2 "python deep_devnagari.py"



