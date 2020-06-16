# Correlation-of-COVID19-XRay-images-to-mortality_rate
We have established a CNN model in a labelled dataset of COVID-19 infected patients. The dataset comprised of a mortality label with chest X-ray images. A GAN network is additionally constructed to generative more training data, and hence to increase the accuracy of our model. 

The links to the datasets are as follows: <br/>
1. https://www.kaggle.com/bachrr/covid-chest-xray/data <br/>
2. https://www.kaggle.com/tawsifurrahman/covid19-radiography-database <br/>


## Running Instructions for CNN - COVID notebook (Primary)
  1. Download dataset from above links.
  2. Change BASEPATH variable to directory of First dataset. 
  3. Change BASEPATH_2 variable to directory of Second dataset. 
  4. Run the notebook
  5. To view the current results, simply click on Covid - CNN.ipynb
  6. (Next Steps) To test the CNN with artificially created images, set COVID_PATH variable as the learning folder produced by GAN.

## Running Instructions for GAN - COVID.py and Read Images.py
  1. The datasets should downloaded into a folder from kaggle, with the above two following links from kaggle. 
  2. After running the GAN - COVID.py file, the genearator will save a number artificially generated 64x64 csv files. 
  3. Then proceed to run Read Images.py. This file will convert the csv number files to gray-scale images. 
  4. The path of image folder can be updated to make use of differnt images. GAN will learn to generate the images provided in      the learning folder, and genearte the new images artifically. 
## Document 
  The findings of the dataset and model can be found on the pdf. 
