#%%

import tensorflow as tf
print(tf.__version__)
from tensorflow import keras

import os
import cv2
import numpy as np
from PIL import Image

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid

n_epochs = 5
img_x,img_y = 32,32

x_train = np.load("coco/train_total_3232.npy")
x_test = np.load("coco/test_total_3232.npy")

print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

sample_images = x_train[:32]
# sample_labels = y_train[:32]
# sample_images = x_train[:x_train.shape[1]]
# sample_labels = y_train[:x_train.shape[1]]

fig = plt.figure(figsize=(16., 8.))
grid = ImageGrid(fig, 111,  # similar to subplot(111)
                 nrows_ncols=(4, 8),  # creates 2x2 grid of axes
                 axes_pad=0.3,  # pad between axes in inch.
                 )

# for ax, image, label in zip(grid, sample_images, sample_labels):
for ax, image in zip(grid, sample_images):
  ax.imshow(image)
#   ax.set_title(label[0])

plt.show()

class createAugment(keras.utils.Sequence):
  'Generates data for Keras'
  def __init__(self, X, y, batch_size=32, dim=(32, 32), n_channels=3, shuffle=True):
      'Initialization'
      self.batch_size = batch_size
      self.y = y
      self.X = X
      self.dim = dim
      self.n_channels = n_channels
      self.shuffle = shuffle
      self.on_epoch_end()

  def __len__(self):
      'Denotes the number of batches per epoch'
      return int(np.floor(len(self.X) / self.batch_size))

  def __getitem__(self, index):
      'Generate one batch of data'
      # Generate indexes of the batch
      indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

      # Generate data
      return self.__data_generation(indexes)

  def on_epoch_end(self):
      'Updates indexes after each epoch'
      self.indexes = np.arange(len(self.X))
      if self.shuffle == True:
          np.random.shuffle(self.indexes)

  def __data_generation(self, idxs):
      X_batch = np.empty((self.batch_size, self.dim[0], self.dim[1], self.n_channels))
      y_batch = np.empty((self.batch_size, self.dim[0], self.dim[1], self.n_channels))

#       x = y = np.random.randint(0, self.dim[1]-1, 1)[0]
#       w = h = np.random.randint(self.dim[1]//10, self.dim[1]//6, 1)[0]
#       x = y = self.dim[1] // 2
#       w = h = self.dim[1] // 8
      x = y = np.random.randint(6, 25, 1)[0]
      w = h = np.random.randint(3, 6, 1)[0]
      
      for i, idx in enumerate(idxs):
        tmp_image = self.X[idx].copy()

        mask = np.full(tmp_image.shape, 255, np.uint8)
        mask[y-h:y+h,x-w:x+w] = 0
        res = np.bitwise_and(tmp_image, mask)

        X_batch[i,] = res/255
        y_batch[i] = self.y[idx]/255
        
      return X_batch, y_batch
  
traingen = createAugment(x_train, x_train, dim=(img_x,img_y))
testgen = createAugment(x_test, x_test, dim=(img_x,img_y))

## Examples
sample_idx = 7 ## Change this to see different batches

sample_images, sample_labels = traingen[sample_idx]

fig = plt.figure(figsize=(16., 8.))
grid = ImageGrid(fig, 111,  # similar to subplot(111)
                 nrows_ncols=(4, 8),  # creates 2x2 grid of axes
                 axes_pad=0.3,  # pad between axes in inch.
                 )

for ax, image in zip(grid, sample_images):
  ax.imshow(image)

plt.show()

#%%
def unet_like(input_x=32, input_y=32):
  inputs = keras.layers.Input((input_x, input_y, 3))
  conv1 = keras.layers.Conv2D(input_x, (3, 3), activation='relu', padding='same')(inputs)
  conv1 = keras.layers.Conv2D(input_x, (3, 3), activation='relu', padding='same')(conv1)
  pool1 = keras.layers.MaxPooling2D(pool_size=(2, 2))(conv1)
  conv2 = keras.layers.Conv2D(input_x*2, (3, 3), activation='relu', padding='same')(pool1)
  conv2 = keras.layers.Conv2D(input_x*2, (3, 3), activation='relu', padding='same')(conv2)
  pool2 = keras.layers.MaxPooling2D(pool_size=(2, 2))(conv2)
  conv3 = keras.layers.Conv2D(input_x*4, (3, 3), activation='relu', padding='same')(pool2)
  conv3 = keras.layers.Conv2D(input_x*4, (3, 3), activation='relu', padding='same')(conv3)
  pool3 = keras.layers.MaxPooling2D(pool_size=(2, 2))(conv3)
  conv4 = keras.layers.Conv2D(input_x*8, (3, 3), activation='relu', padding='same')(pool3)
  conv4 = keras.layers.Conv2D(input_x*8, (3, 3), activation='relu', padding='same')(conv4)
  pool4 = keras.layers.MaxPooling2D(pool_size=(2, 2))(conv4)
  conv5 = keras.layers.Conv2D(input_x*16, (3, 3), activation='relu', padding='same')(pool4)
  conv5 = keras.layers.Conv2D(input_x*16, (3, 3), activation='relu', padding='same')(conv5)

  up6 = keras.layers.concatenate([keras.layers.Conv2DTranspose(input_x*8, (2, 2), strides=(2, 2), padding='same')(conv5), conv4], axis=3)
  conv6 = keras.layers.Conv2D(input_x*8, (3, 3), activation='relu', padding='same')(up6)
  conv6 = keras.layers.Conv2D(input_x*8, (3, 3), activation='relu', padding='same')(conv6)
  up7 = keras.layers.concatenate([keras.layers.Conv2DTranspose(input_x*4, (2, 2), strides=(2, 2), padding='same')(conv6), conv3], axis=3)
  conv7 = keras.layers.Conv2D(input_x*4, (3, 3), activation='relu', padding='same')(up7)
  conv7 = keras.layers.Conv2D(input_x*4, (3, 3), activation='relu', padding='same')(conv7)
  up8 = keras.layers.concatenate([keras.layers.Conv2DTranspose(input_x*2, (2, 2), strides=(2, 2), padding='same')(conv7), conv2], axis=3)
  conv8 = keras.layers.Conv2D(input_x*2, (3, 3), activation='relu', padding='same')(up8)
  conv8 = keras.layers.Conv2D(input_x*2, (3, 3), activation='relu', padding='same')(conv8)
  up9 = keras.layers.concatenate([keras.layers.Conv2DTranspose(input_x, (2, 2), strides=(2, 2), padding='same')(conv8), conv1], axis=3)
  conv9 = keras.layers.Conv2D(input_x, (3, 3), activation='relu', padding='same')(up9)
  conv9 = keras.layers.Conv2D(input_x, (3, 3), activation='relu', padding='same')(conv9)
  conv10 = keras.layers.Conv2D(3, (3, 3), activation='sigmoid', padding='same')(conv9)

  return keras.models.Model(inputs=[inputs], outputs=[conv10])  

#%%
## Metric
def dice_coef(y_true, y_pred):
    y_true_f = keras.backend.flatten(y_true)
    y_pred_f = keras.backend.flatten(y_pred)
    intersection = keras.backend.sum(y_true_f * y_pred_f)
    return (2. * intersection + 1) / (keras.backend.sum(y_true_f) + keras.backend.sum(y_pred_f) + 1)
#%%

keras.backend.clear_session()

from tensorflow.keras.models import load_model

# Load the model from the .h5 file
custom_objects = {'dice_coef': dice_coef}

model5 = load_model('models/impaint_coco3232_epch'+str(n_epochs)+'.h5', custom_objects=custom_objects)
model10 = load_model('models/impaint_coco3232_epch'+str(10)+'.h5', custom_objects=custom_objects)

import pickle
with open('models/history_epch'+str(n_epochs)+'.pickle', 'rb') as f:
    history = pickle.load(f)

# Plot training & validation loss values
fig = plt.figure(figsize=(6,4))
plt.plot(history['loss'])
plt.plot(history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'])
plt.show()


#%%
im = sample_images[0]
plt.imshow(im)
impainted_image = model5.predict(im.reshape((1,)+im.shape))

plt.imshow(impainted_image.reshape(impainted_image.shape[1:]))
plt.show()

## Examples
rows = 16
sample_idx = np.random.randint(0, len(testgen), rows)

fig, axs = plt.subplots(nrows=rows, ncols=3, figsize=(6, 32))

for i, idx in enumerate(sample_idx):
#   sample_images, sample_labels = traingen[idx]
#   img_idx = np.random.randint(0, len(sample_images)-1, 1)[0]
  img_idx=i
  impainted_image = model5.predict(sample_images[img_idx].reshape((1,)+sample_images[img_idx].shape))
  axs[i][0].imshow(sample_labels[img_idx])
  axs[i][1].imshow(sample_images[img_idx])
  axs[i][2].imshow(impainted_image.reshape(impainted_image.shape[1:]))
  axs[i][0].axis('off')  
  axs[i][1].axis('off')  
  axs[i][2].axis('off')  
plt.show()
# fig.savefig("results_epch5.png",bbox_inches='tight')

#%%

rows = 8
ncols = 3
sample_idx = np.random.randint(0, len(testgen), rows)

k = 2

fig, axs = plt.subplots(nrows=ncols, ncols=rows, figsize=(rows*2, ncols*2))

# add labels to the left of the images
fig.text(0.105, 0.78, 'Original', va='center', rotation='vertical', fontsize=12, fontweight='bold')
fig.text(0.105, 0.50, 'Input', va='center', rotation='vertical', fontsize=12, fontweight='bold')
fig.text(0.105, 0.24, 'Output', va='center', rotation='vertical', fontsize=12, fontweight='bold')
# fig.text(0.105, 0., 'Output 10', va='center', rotation='vertical', fontsize=12, fontweight='bold')

# add labels to the top of the columns
# for i in range(rows):
#     axs[0][i].set_title(f"Sample {i}", fontsize=12)

for i, idx in enumerate(sample_idx):
#   sample_images, sample_labels = traingen[idx]
#   img_idx = np.random.randint(0, len(sample_images)-1, 1)[0]
  img_idx=i
  impainted_image5 = model5.predict(sample_images[img_idx].reshape((1,)+sample_images[img_idx].shape))
  impainted_image10 = model10.predict(sample_images[img_idx].reshape((1,)+sample_images[img_idx].shape))
  axs[0][i].imshow(sample_labels[img_idx])
  axs[1][i].imshow(sample_images[img_idx])
#   axs[2][i].imshow(impainted_image5.reshape(impainted_image.shape[1:]))
  axs[2][i].imshow(impainted_image10.reshape(impainted_image.shape[1:]))
  axs[0][i].axis('off')
  axs[1][i].axis('off')
  axs[2][i].axis('off')
#   axs[3][i].axis('off')

# plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1, wspace=0.1, hspace=0.1)
fig.savefig("figs/results_epch" + str(n_epochs) + "rows" + str(rows) + ".png",bbox_inches='tight')

plt.show()

#%%