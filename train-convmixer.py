from tensorflow.keras import layers
from tensorflow import keras
import cv2
import matplotlib.pyplot as plt
##import tensorflow_addons as tfa
import tensorflow as tf
import numpy as np
from imutils import paths
from sklearn.model_selection import train_test_split
learning_rate = 0.001
weight_decay = 0.0001
batch_size = 16
num_epochs = 50
import tensorflow_addons as tfa
print("Section 1")
##from keras.datasets import cifar10
##from keras.utils.np_utils import to_categorical   
##
##
##(X_train, y_train), (X_test, y_test) = cifar10.load_data()
##print(X_train)
##print(type(X_train))
##print(y_train)
##print(type(y_train))
##
##print("Shape of training data:")
##print(X_train.shape)
##print(y_train.shape)
##print("Shape of test data:")
##print(X_test.shape)
##print(y_test.shape)
i=0
data = []
labels = []
for imagePath in paths.list_images("Dataset//training//"):
    # extract the make of the car
##    print(imagePath)
    make1 = imagePath.split("Dataset//training//")[1]
    make2 = make1.split("\\")[0]
    if make2 == 'Banana':
        make = 0
    elif make2 == 'Coconut':
        make = 1
    elif make2 == 'Mango':
        make = 2
    elif make2 == 'Not specified':
        make = 3
    elif make2 == 'Oil Palm':
        make = 4
    elif make2 == 'Papaya':
        make = 5
    
        
##    print(make)
##    i=i+1
##    if (i==100):
##            break
    image = cv2.imread(imagePath)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    data.append(image)
    labels.append([make])


train_size = 0.9
test_size = 0.1
data1 = np.array(data)
print(data1.shape)
##print(data1)
labels1 = np.array(labels)
print(labels1.shape)


x_train, x_test, y_train, y_test = train_test_split(data1, labels1, train_size = train_size,test_size = test_size)
####x_train = np.expand_dims(x_train, axis=-1) # <--- add channel axis
####x_train = x_train.astype('float32') / 255
####y_train = tf.keras.utils.to_categorical(y_train, num_classes=6)

val_indices = int(len(x_train) * test_size)
new_x_train, new_y_train = x_train[val_indices:], y_train[val_indices:]
x_val, y_val = x_train[:val_indices], y_train[:val_indices]

print(new_x_train.shape,new_y_train.shape)
print(f"Training data samples: {len(new_x_train)}")
print(f"Validation data samples: {len(x_val)}")
print(f"Test data samples: {len(x_test)}")
##
image_size = 32
auto = tf.data.AUTOTUNE



data_augmentation = keras.Sequential(
    [layers.RandomCrop(image_size, image_size), layers.RandomFlip("horizontal"),],
    name="data_augmentation",
)
print("Section 2")



def make_datasets(images, labels, is_train=False):
    dataset = tf.data.Dataset.from_tensor_slices((images, labels))
    if is_train:
        dataset = dataset.shuffle(batch_size * 10)
    dataset = dataset.batch(batch_size)
    if is_train:
        dataset = dataset.map(
            lambda x, y: (data_augmentation(x), y), num_parallel_calls=auto
        )
    return dataset.prefetch(auto)


train_dataset = make_datasets(new_x_train, new_y_train, is_train=True)
val_dataset = make_datasets(x_val, y_val)
test_dataset = make_datasets(x_test, y_test)
print(test_dataset)
print("Section 3")

def activation_block(x):
    x = layers.Activation("gelu")(x)
    return layers.BatchNormalization()(x)


def conv_stem(x, filters: int, patch_size: int):
    x = layers.Conv2D(filters, kernel_size=patch_size, strides=patch_size)(x)
    return activation_block(x)


def conv_mixer_block(x, filters: int, kernel_size: int):
    # Depthwise convolution.
    x0 = x
    x = layers.DepthwiseConv2D(kernel_size=kernel_size, padding="same")(x)
    x = layers.Add()([activation_block(x), x0])  # Residual.

    # Pointwise convolution.
    x = layers.Conv2D(filters, kernel_size=1)(x)
    x = activation_block(x)

    return x


def get_conv_mixer_256_8(
    image_size=32, filters=256, depth=8, kernel_size=5, patch_size=2, num_classes=6
):
    """ConvMixer-256/8: https://openreview.net/pdf?id=TVHS5Y4dNvM.
    The hyperparameter values are taken from the paper.
    """
    inputs = keras.Input((image_size, image_size,3))
    x = layers.Rescaling(scale=1.0 / 255)(inputs)

    # Extract patch embeddings.
    x = conv_stem(x, filters, patch_size)

    # ConvMixer blocks.
    for _ in range(depth):
        x = conv_mixer_block(x, filters, kernel_size)

    # Classification block.
    x = layers.GlobalAvgPool2D()(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    return keras.Model(inputs, outputs)

def run_experiment(model):
    optimizer = tfa.optimizers.AdamW(
        learning_rate=learning_rate, weight_decay=weight_decay
    )

    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    checkpoint_filepath = "checkpoint//"
    checkpoint_callback = keras.callbacks.ModelCheckpoint(
        checkpoint_filepath,
        monitor="val_accuracy",
        save_best_only=True,
        save_weights_only=True,
    )

    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=num_epochs,
        callbacks=[checkpoint_callback],
    )

    model.load_weights(checkpoint_filepath)
    _, accuracy = model.evaluate(test_dataset)
    print(f"Test accuracy: {round(accuracy * 100, 2)}%")

    return history, model

print("Section 4")
conv_mixer_model = get_conv_mixer_256_8()
history, conv_mixer_model = run_experiment(conv_mixer_model)
##
def visualization_plot(weights, idx=1):
    # First, apply min-max normalization to the
    # given weights to avoid isotrophic scaling.
    p_min, p_max = weights.min(), weights.max()
    weights = (weights - p_min) / (p_max - p_min)

    # Visualize all the filters.
    num_filters = 256
    plt.figure(figsize=(8, 8))

    for i in range(num_filters):
        current_weight = weights[:, :, :, i]
        if current_weight.shape[-1] == 1:
            current_weight = current_weight.squeeze()
        ax = plt.subplot(16, 16, idx)
        ax.set_xticks([])
        ax.set_yticks([])
        plt.imshow(current_weight)
        idx += 1
    plt.show()

# We first visualize the learned patch embeddings.
patch_embeddings = conv_mixer_model.layers[2].get_weights()[0]
visualization_plot(patch_embeddings)
for i, layer in enumerate(conv_mixer_model.layers):
    if isinstance(layer, layers.DepthwiseConv2D):
        if layer.get_config()["kernel_size"] == (5, 5):
            print(i, layer)

idx = 26  # Taking a kernel from the middle of the network.

kernel = conv_mixer_model.layers[idx].get_weights()[0]
kernel = np.expand_dims(kernel.squeeze(), axis=2)
visualization_plot(kernel)
