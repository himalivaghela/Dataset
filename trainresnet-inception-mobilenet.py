import numpy as np
from matplotlib import pyplot as plt
import os
import tensorflow as tf
from tensorflow.keras.models import Model,load_model, Sequential
from tensorflow.keras.layers import  GlobalAveragePooling2D, Dropout, Dense, Flatten
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import  Adam
import cv2
from keras.callbacks import ModelCheckpoint, TensorBoard
##from tensorflow.keras.applications.resnet50 import ResNet50
##from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications import MobileNetV2
import gc
##from numba import cuda 
##device = cuda.get_current_device()
##device.reset()
gc.collect()

train_path = "Dataset//training"
test_path = "Dataset//test"
train_datagen = ImageDataGenerator(rescale=1./255,shear_range=0.2,zoom_range=0.2,horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1./255)
print("Sec 1")

##for im in os.listdir(train_path):
##    num = 0
##    img_class = (os.path.join(train_path, im))
##    img = os.listdir(img_class)
##    img_path = os.path.join(train_path, img_class, img[0])
##    image = cv2.imread(img_path)
##    plt.imshow(image)   
##    plt.axis('off')
##    plt.title(im)
##    plt.show()

train_generator = train_datagen.flow_from_directory(
    train_path,
    target_size=(224, 224),
    batch_size=8,
    color_mode = 'rgb',
    class_mode='categorical')

test_generator = test_datagen.flow_from_directory(
    test_path,
    target_size=(224, 224),
    batch_size=8,
    color_mode = 'rgb',
    class_mode='categorical')

print("Sec 2")
base_model = MobileNetV2(weights= 'imagenet', include_top=False, input_shape= (224, 224, 3))
resnet = base_model.output
resnet = GlobalAveragePooling2D()(resnet)
resnet = Dropout(0.25)(resnet)
predictions = Dense(6, activation= 'softmax')(resnet)
Resnet50 = Model(inputs = base_model.input, outputs = predictions)
adam = Adam(learning_rate = 0.0001)
Resnet50.compile(optimizer= adam, loss='categorical_crossentropy', metrics=['accuracy'])
print("Sec 3")
##tl_checkpoint_1 = ModelCheckpoint(filepath='Mobilenet_model_v1.weights.best.hdf5',
##                                  save_best_only=True,
##                                  verbose=1)
History = Resnet50.fit(train_generator, 
                                 batch_size = 8,
                                 epochs = 50,
                                 validation_data = (test_generator))


epochs= []
for i in range(50):
    epochs.append(i)

model_json = base_model.to_json()
with open("MobileNetV2model.json", "w") as json_file:
    json_file.write(model_json)
base_model.save_weights("MobileNetV2model.h5")
print("Saved model to disk")

    
plt.figure(figsize = (15, 10))    
plt.plot(epochs,History.history['accuracy'], label="Train")
plt.plot(epochs,History.history['val_accuracy'], label="Test")
plt.title("Train-Test Accuracy")
plt.xlabel("Number of Epochs")
plt.xlabel("Accuracy")
plt.legend()
plt.show()


plt.figure(figsize = (15, 10))   
plt.plot(epochs,History.history['loss'], label="Train")
plt.plot(epochs,History.history['val_loss'], label="Test")
plt.title("Train-Test Loss")
plt.xlabel("Number of Epochs")
plt.xlabel("Accuracy")
plt.legend()
plt.show()
