#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import cv2
import numpy as np
from numpy import expand_dims
from numpy import zeros
from numpy import ones
from numpy import vstack
from numpy.random import randn
from numpy.random import randint
from keras.datasets.mnist import load_data
from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Reshape
from keras.layers import Flatten
from keras.layers import Conv2D
from keras.layers import Conv2DTranspose
from keras.layers import LeakyReLU
from keras.layers import Dropout
from matplotlib import pyplot
import matplotlib.pyplot as plt
g_model_save = 0
d_model_save = 0


# In[ ]:


def define_discriminator(in_shape=(64,64,1)):
    model = Sequential()
    model.add(Conv2D(64, (3,3), strides=(2, 2), padding='same', input_shape=in_shape))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.4))
    model.add(Conv2D(64, (3,3), strides=(2, 2), padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.4))
    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))
    opt = Adam(lr=0.0002, beta_1=0.5)
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
    return model
 
def define_generator(latent_dim):
    model = Sequential()
    # foundation for 16x16 image
    n_nodes = 128 * 16 * 16
    model.add(Dense(n_nodes, input_dim=latent_dim))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Reshape((16, 16, 128)))
    # upsample to 32x32
    model.add(Conv2DTranspose(128, (4,4), strides=(2,2), padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    # upsample to 64x64
    model.add(Conv2DTranspose(128, (4,4), strides=(2,2), padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Conv2D(1, (16,16), activation='sigmoid', padding='same'))
    return model
 

def define_gan(g_model, d_model):
    d_model.trainable = False
    model = Sequential()
    model.add(g_model)
    model.add(d_model)
    opt = Adam(lr=0.0002, beta_1=0.5)
    model.compile(loss='binary_crossentropy', optimizer=opt)
    return model

def load_real_samples(): 
    training_data_ARR = []
    def training_data():
        DATA_DIR = os.getcwd()
        path_COVID_19_RADIOGRAPHY = os.path.join(DATA_DIR,"COVID-19 Radiography Database/COVID-19")
        #path_COVID_19_RADIOGRAPHY
        for img in os.listdir(path_COVID_19_RADIOGRAPHY):
            img_arr = cv2.imread(os.path.join(path_COVID_19_RADIOGRAPHY,img),cv2.IMREAD_GRAYSCALE)
            IMG_SIZE = 64
            new_arr = cv2.resize(img_arr,(IMG_SIZE,IMG_SIZE))
            training_data_ARR.append([new_arr,1])
    training_data()
    
    COVID_19_DATASET = []
    for x in np.array(training_data_ARR)[:,0]:
        COVID_19_DATASET.append(np.array(x))

    trainX = np.array(COVID_19_DATASET)
    X = expand_dims(trainX, axis=-1)
    X = X.astype('float32')
    X = X / 255.0
    return X
 

def generate_real_samples(dataset, n_samples):
    ix = randint(0, dataset.shape[0], n_samples)
    X = dataset[ix]
    y = ones((n_samples, 1))
    return X, y
 
def generate_latent_points(latent_dim, n_samples):
    x_input = randn(latent_dim * n_samples)
    x_input = x_input.reshape(n_samples, latent_dim)
    return x_input
 
def generate_fake_samples(g_model, latent_dim, n_samples):
    x_input = generate_latent_points(latent_dim, n_samples)
    X = g_model.predict(x_input)
    y = zeros((n_samples, 1))
    return X, y
 

def save_plot(examples, epoch, n=10):
    for i in range(n * n):
        pyplot.subplot(n, n, 1 + i)
        pyplot.axis('off')
        pyplot.imshow(examples[i, :, :, 0], cmap='gray_r')
    filename = 'generated_plot_e%03d.png' % (epoch+1)
    pyplot.savefig(filename)
    pyplot.close()
 
# evaluate the discriminator, plot generated images, save generator model
def summarize_performance(epoch, g_model, d_model, dataset, latent_dim, n_samples=100):
    X_real, y_real = generate_real_samples(dataset, n_samples)
    _, acc_real = d_model.evaluate(X_real, y_real, verbose=0)
    x_fake, y_fake = generate_fake_samples(g_model, latent_dim, n_samples)
    _, acc_fake = d_model.evaluate(x_fake, y_fake, verbose=0)
    print('>Accuracy real: %.0f%%, fake: %.0f%%' % (acc_real*100, acc_fake*100))
    # save plot
    #save_plot(x_fake, epoch)
    # save the generator model tile file
    filename = 'generator_model_%03d.h5' % (epoch + 1)
    g_model.save(filename)

# train the generator and discriminator
def train(g_model, d_model, gan_model, dataset, latent_dim, n_epochs=50, n_batch=219):
    bat_per_epo = int(dataset.shape[0] / n_batch)
    half_batch = int(n_batch / 2)
    # manually enumerate epochs
    for i in range(n_epochs):
        for j in range(bat_per_epo):
            X_real, y_real = generate_real_samples(dataset, half_batch)
            X_fake, y_fake = generate_fake_samples(g_model, latent_dim, half_batch)
            X, y = vstack((X_real, X_fake)), vstack((y_real, y_fake))
            d_loss, _ = d_model.train_on_batch(X, y)
            X_gan = generate_latent_points(latent_dim, n_batch)
            y_gan = ones((n_batch, 1))
            g_loss = gan_model.train_on_batch(X_gan, y_gan)
            print('>%d, %d/%d, d=%.3f, g=%.3f' % (i+1, j+1, bat_per_epo, d_loss, g_loss))
        if (i == 0):
            #summarize_performance(i, g_model, d_model, dataset, latent_dim)
            g_model_save = g_model
            d_model_save = d_model
 
# size of the latent space
latent_dim = 100
# # create the discriminator
d_model = define_discriminator()
# # create the generator
g_model = define_generator(latent_dim)
# # create the gan
gan_model = define_gan(g_model, d_model)
# # load image data
dataset = load_real_samples()
# # train model
train(g_model, d_model, gan_model, dataset, latent_dim)


# In[ ]:


print(gan_model)


# In[ ]:


from keras.models import load_model
from numpy.random import randn
from matplotlib import pyplot
import os


# In[ ]:


# generate points in latent space as input for the generator
def generate_latent_points(latent_dim, n_samples):
    # generate points in the latent space
    x_input = randn(latent_dim * n_samples)
    # reshape into a batch of inputs for the network
    x_input = x_input.reshape(n_samples, latent_dim)
    return x_input

model = g_model
latent_points = generate_latent_points(100, 150)
X = model.predict(latent_points)


# In[ ]:


print(X.shape)


# In[ ]:


X = X.reshape(150,64,64)


# In[ ]:


from numpy import asarray
from numpy import savetxt


# In[ ]:


print(X[0].shape)


# In[ ]:


i=0
for i in range(X.shape[0]):
    savetxt('data' +str(i)+'.csv', X[i], delimiter=',')


# In[ ]:




