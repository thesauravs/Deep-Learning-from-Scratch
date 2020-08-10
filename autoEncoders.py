import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

print(tf.__version__)

data = keras.datasets.mnist

(train_images, _), (test_images, _) = data.load_data()

# Normalizing
train_images = train_images/255
test_images = test_images/255

print(train_images.shape)

#Visualize the train image
plt.imshow(train_images[1], cmap='gray')
plt.show

input_img = tf.keras.Input(shape=(784,))
# Encoding
encoded = tf.keras.layers.Dense(16, activation ='relu')(input_img)

# Decoding
decoded = tf.keras.layers.Dense(784, activation = 'sigmoid')(encoded)

autoEncoder = tf.keras.Model(input_img, decoded)
encoder = tf.keras.Model(input_img, encoded)

encoded_input = tf.keras.Input(shape=(16,))

decoder_layers = autoEncoder.layers[-1]
decoder = tf.keras.Model(encoded_input, decoder_layers(encoded_input))

# Compile
autoEncoder.compile(optimizer = 'Adam', loss='binary_crossentropy')

# Flattening the images
train_images = train_images.reshape((len(train_images), np.prod(train_images.shape[1:])))
test_images = test_images.reshape((len(test_images), np.prod(test_images.shape[1:])))
print(train_images.shape)
print(test_images.shape)

autoEncoder.fit(train_images, train_images, epochs = 5,
                shuffle= True,
                validation_data = (test_images, test_images))

encoded_img = encoder.predict(test_images)
decoded_img = decoder.predict(encoded_img)

n = 10
plt.figure(figsize=(7,7))
for i in range(n):
    ax = plt.subplot(2, n, i+1)
    plt.imshow(test_images[i].reshape(28,28))
    plt.gray
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    
    ax = plt.subplot(2, n, i+1+n)
    plt.imshow(decoded_img[i].reshape(28,28))
    plt.gray
    ax.get_xaxis().set_visible(False)

plt.show()

