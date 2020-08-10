import os
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

np.random.seed(2)

print(tf.__version__)

# Folder path
folder_path = r'/Users/thesauravs/Documents/GitHub/Deep-Learning-from-Scratch/Sinusoidal'
os.chdir(folder_path)

#Loading the dataset
data = pd.read_csv('sine.csv')
data.head()

# Quick look at the joint distribution of a few pairs of columns
sns.pairplot(data[["x", "sin(x)"]], diag_kind="kde")

# To look at overall statistics
data_stats = data.describe()
data_stats.pop('sin(x)')
data_stats = data_stats.transpose()
print(data_stats)

# Split features from labels
train_labels = data.pop('sin(x)')
print(train_labels)
#print(len(data.keys()))

print(data)
# Normalizing the features to train faster
#data = (data - data_stats['mean']) / data_stats['std']
#print(data)

# Build the model
model = keras.Sequential([
        layers.Dense(128, activation='relu', input_shape=[len(data.keys())]),
        layers.Dense(64, activation='relu'),
        layers.Dense(32, activation='relu'),  
        layers.Dense(16, activation='relu'),
#        layers.Dense(8, activation='relu'),
#        layers.Dense(4, activation='relu'),
#        layers.Dense(2, activation='relu'),        
        layers.Dense(1)
        ])

# Try learning_rate = 0.1, 0.001, 0.0001, 0.00001 and analyse the loss graph
optimizer = keras.optimizers.Adam(learning_rate=0.0001)

model.compile(loss='mse',
              optimizer=optimizer,
              metrics=['mae','mse'])

model.summary()

# The patience parameter is the amount of epochs to check for improvement
early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)

history = model.fit(data, train_labels,
                    epochs=200,
                    validation_split=0.2,
                    verbose=1,
                    callbacks=[early_stop])

hist = pd.DataFrame(history.history)
hist['epoch'] = history.epoch

plt.figure()
plt.xlabel('Epoch')
plt.ylabel('Mean Squared Error')
plt.ylim([0, hist['loss'][0]])
plt.plot(hist['epoch'], hist['mse'], label='Train error')
plt.plot(hist['epoch'], hist['val_mse'], label='Val error')
plt.legend()
plt.show()

hist.head(7)
hist.tail(7)

predictions = model.predict(data[:])

plt.scatter(list(data['x']), train_labels, color='g', s=0.05, label='Original')
plt.scatter(list(data['x']), predictions[:], color='r', s=0.05, label='Predicted')
plt.xlabel('X')
plt.ylabel('sin(X)')
plt.title('Sine Wave')
plt.legend(loc='best', fontsize='large')
plt.show()

model.save('sine.h5')
