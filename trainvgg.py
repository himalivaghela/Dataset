import os
from keras.models import Model
from tensorflow.keras.optimizers import Adam
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.applications import MobileNetV2, ResNet50, InceptionV3s
from keras.layers import Dense, Dropout, Flatten
from pathlib import Path
import numpy as np
from livelossplot.inputs.keras import PlotLossesCallback
from sklearn.metrics import accuracy_score
import seaborn as sns
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
BATCH_SIZE = 4
train_generator = ImageDataGenerator(rotation_range=90, 
                                     brightness_range=[0.1, 0.7],
                                     width_shift_range=0.5, 
                                     height_shift_range=0.5,
                                     horizontal_flip=True, 
                                     vertical_flip=True,
                                     validation_split=0.15,
                                     preprocessing_function=preprocess_input) # VGG16 preprocessing

test_generator = ImageDataGenerator(preprocessing_function=preprocess_input) 

print("Section 1")
download_dir = Path('Dataset//')
print(download_dir)

train_data_dir = download_dir/'training'
test_data_dir = download_dir/'test'

class_subset = sorted(os.listdir(download_dir/'training'))[:10]
print(class_subset)
print("Section 2")

traingen = train_generator.flow_from_directory(train_data_dir,
                                               target_size=(224, 224),
                                               class_mode='categorical',
                                               classes=class_subset,
                                               subset='training',
                                               batch_size=BATCH_SIZE, 
                                               shuffle=True,
                                               seed=42)

validgen = train_generator.flow_from_directory(train_data_dir,
                                               target_size=(224, 224),
                                               class_mode='categorical',
                                               classes=class_subset,
                                               subset='validation',
                                               batch_size=BATCH_SIZE,
                                               shuffle=True,
                                               seed=42)

testgen = test_generator.flow_from_directory(test_data_dir,
                                             target_size=(224, 224),
                                             class_mode=None,
                                             classes=class_subset,
                                             batch_size=1,
                                             shuffle=False,
                                             seed=42)
print("Section 3")

def create_model(input_shape, n_classes, optimizer='rmsprop', fine_tune=0):
    """
    Compiles a model integrated with VGG16 pretrained layers
    
    input_shape: tuple - the shape of input images (width, height, channels)
    n_classes: int - number of classes for the output layer
    optimizer: string - instantiated optimizer to use for training. Defaults to 'RMSProp'
    fine_tune: int - The number of pre-trained layers to unfreeze.
                If set to 0, all pretrained layers will freeze during training
    """
    
    # Pretrained convolutional layers are loaded using the Imagenet weights.
    # Include_top is set to False, in order to exclude the model's fully-connected layers.
    conv_base = VGG16(include_top=False,
                     weights='imagenet', 
                     input_shape=input_shape)
    
    # Defines how many layers to freeze during training.
    # Layers in the convolutional base are switched from trainable to non-trainable
    # depending on the size of the fine-tuning parameter.
    if fine_tune > 0:
        for layer in conv_base.layers[:-fine_tune]:
            layer.trainable = False
    else:
        for layer in conv_base.layers:
            layer.trainable = False

    # Create a new 'top' of the model (i.e. fully-connected layers).
    # This is 'bootstrapping' a new top_model onto the pretrained layers.
    top_model = conv_base.output
    top_model = Flatten(name="flatten")(top_model)
    top_model = Dense(4096, activation='relu')(top_model)
    top_model = Dense(1072, activation='relu')(top_model)
    top_model = Dropout(0.2)(top_model)
    output_layer = Dense(n_classes, activation='softmax')(top_model)
    
    # Group the convolutional base and new fully-connected layers into a Model object.
    model = Model(inputs=conv_base.input, outputs=output_layer)

    # Compiles the model for training.
    model.compile(optimizer=optimizer, 
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    return model

print("Section 4")
###############################################################################
input_shape = (224, 224, 3)
##optim_1 = Adam(learning_rate=0.001)
optim_2 = Adam(lr=0.0001)
n_classes=6

n_steps = traingen.samples // BATCH_SIZE
n_val_steps = validgen.samples // BATCH_SIZE
n_epochs = 50

# First we'll train the model without Fine-tuning
##vgg_model = create_model(input_shape, n_classes, optim_1, fine_tune=0)
vgg_model = create_model(input_shape, n_classes, optim_2, fine_tune=2)
##plot_loss_1 = PlotLossesCallback()
plot_loss_2 = PlotLossesCallback()
# ModelCheckpoint callback - save best weights
tl_checkpoint_1 = ModelCheckpoint(filepath='tl2_model_v1.weights.best.hdf5',
                                  save_best_only=True,
                                  verbose=1)

# EarlyStopping
early_stop = EarlyStopping(monitor='val_loss',
                           patience=10,
                           restore_best_weights=True,
                           mode='min')


vgg_history = vgg_model.fit(traingen,
                            batch_size=BATCH_SIZE,
                            epochs=n_epochs,
                            validation_data=validgen,
                            steps_per_epoch=n_steps,
                            validation_steps=n_val_steps,
                            callbacks=[tl_checkpoint_1],
                            verbose=1)

##callbacks=[tl_checkpoint_1, early_stop, plot_loss_1],
vgg_model.load_weights('tl2_model_v1.weights.best.hdf5') # initialize the best trained weights
print("Section 5")
true_classes = testgen.classes
class_indices = traingen.class_indices
class_indices = dict((v,k) for k,v in class_indices.items())

vgg_preds = vgg_model.predict(testgen)
vgg_pred_classes = np.argmax(vgg_preds, axis=1)

vgg_acc = accuracy_score(true_classes, vgg_pred_classes)
print("VGG16 Model Accuracy without Fine-Tuning: {:.2f}%".format(vgg_acc * 100))



test_generator = ImageDataGenerator(rescale=1/255.)

testgen = test_generator.flow_from_directory(download_dir/'test',
                                             target_size=(224, 224),
                                             batch_size=1,
                                             class_mode=None,
                                             classes=class_subset, 
                                             shuffle=False,
                                             seed=42)

##scratch_preds = scratch_model.predict(testgen)
##scratch_pred_classes = np.argmax(scratch_preds, axis=1)
##
##scratch_acc = accuracy_score(true_classes, scratch_pred_classes)
##print("From Scratch Model Accuracy with Fine-Tuning: {:.2f}%".format(scratch_acc * 100))

class_names = testgen.class_indices.keys()

cm = confusion_matrix(true_classes, vgg_pred_classes)
print(cm)

##def plot_heatmap(y_true, y_pred, class_names, title):
##    cm = confusion_matrix(y_true, y_pred)
##    sns.heatmap(
##        cm, 
##        annot=True, 
##        square=True, 
##        xticklabels=class_names, 
##        yticklabels=class_names,
##        fmt='d', 
##        cmap=plt.cm.Blues,
##        cbar=False,
##        ax=ax
##    )
##    ax.set_title(title, fontsize=16)
##    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
##    ax.set_ylabel('True Label', fontsize=12)
##    ax.set_xlabel('Predicted Label', fontsize=12)
##
####fig, (ax1, ax2, ax3) = plt.subplots(1, 1, figsize=(20, 10))
##
####plot_heatmap(true_classes, scratch_pred_classes, class_names, ax1, title="Custom CNN")    
##plot_heatmap(true_classes, vgg_pred_classes, class_names, title="Transfer Learning (VGG16) Fine-Tuning")    
####plot_heatmap(true_classes, vgg_pred_classes_ft, class_names, ax3, title="Transfer Learning (VGG16) with Fine-Tuning")    
##
####fig.suptitle("Confusion Matrix Model Comparison", fontsize=24)
####fig.tight_layout()
####fig.subplots_adjust(top=1.25)
##plt.show()

