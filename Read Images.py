#!/usr/bin/env python
# coding: utf-8

# In[19]:


import numpy as np
from numpy import expand_dims
from numpy import ones
from numpy import zeros
from numpy.random import rand
from numpy.random import randint
import pandas as pd
import os
import matplotlib.pyplot as plt
import cv2
from keras.models import Sequential
from keras.optimizers import Adam
from keras.layers import Dense
from keras.layers import Conv2D
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers import LeakyReLU
from keras.utils.vis_utils import plot_model
import keras
import pydotplus
from keras.utils.vis_utils import model_to_dot
import random
from numpy import genfromtxt


# In[20]:


images_data = []
i=0
for i in range(150):
    my_data = genfromtxt('data' +str(i)+'.csv', delimiter=',')
    images_data.append(my_data)
    #savetxt('data' +str(i)+'.csv', X[i], delimiter=',')


# In[21]:


print(np.array(images_data).shape[0])


# In[33]:


for x in range(np.array(images_data).shape[0]):
    plt.imshow(images_data[x],cmap="gray")
    plt.savefig('image_' +str(x)+'.png')

