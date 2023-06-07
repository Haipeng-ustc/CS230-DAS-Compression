## CS230 Project
### Convolutional Autoencoder for Compressing Distributed Acoustic Sensing Data from Urban Environments

The Autoencoder model is based on Lossy Data Compression from Tensorflow's official tutorial: https://www.tensorflow.org/tutorials/generative/data_compression

## Project Description
This project focuses on applying the autoencoder convolutional neural network to compress vast amounts of Distributed Acoustic Sensing (DAS) data. The primary objective is to achieve data compression with minimal loss, effectively reducing the size of DAS data while maintaining its usability for traffic monitoring and geophysical analysis. 

## Setup and Dependencies
We run the code on Google Colab for now. The following dependencies are required:

1. Python (version 3.10.11)
2. TensorFlow (version 2.12.0)
3. NumPy (version 1.22.4)
4. Matplotlib (version 3.7.1)

## Dataset
The data used in this project can be obtained by contacting one of the authors. Please refer to the the Readme file in the data foler. 

## Future Enhancements
Explore potential integration of preprocessing steps into the autoencoder's workflow, i.e., replacing some of the time-consuming preprocessing steps, such as specified noise or band-pass filtering within a CAE.

## Authors

Hassan Almomin (almomiha@stanford.edu)

Thomas Cullison (tculliso@stanford.edu)

Haipeng Li (haipeng@stanford.edu)

Department of Geophysics, Stanford University

