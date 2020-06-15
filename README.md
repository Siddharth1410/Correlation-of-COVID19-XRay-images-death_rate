# Correlation-of-COVID19-XRay-images-to-mortality_rate
We have established a CNN model in a labelled dataset of COVID-19 infected patients. The dataset comprised of a mortality label with chest X-ray images. A GAN network is additionally constructed to generative more training data, and hence to increase the accuracy of our model. 

The links to the datasets are as follows: <br/>
https://www.kaggle.com/tawsifurrahman/covid19-radiography-database? <br/>
https://www.kaggle.com/bachrr/covid-chest-xray/data




Running Instructions for GAN - COVID.py and Read Images.py
  1. The datasets should downloaded into a folder from kaggle, with the above two following links from kaggle. 
  2. After running the GAN - COVID.py file, the genearator will save a number artificially generated 64x64 csv files. 
  3. Then proceed to run Read Images.py. This file will convert the csv number files to gray-scale images. 
  4. The path of image folder can be updated to make use of differnt images. GAN will learn to generate the images provided in      the learning folder, and genearte the new images artifically. 
