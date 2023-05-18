## CS230 Project
Efficient Compression of Preprocessed High-Frequency DAS Data Using Autoencoders


## Project Discribtion
This project focuses on applying the autoencoder convolutional neural network to compress vast amounts of Distributed Acoustic Sensing (DAS) data. The primary objective is to achieve data compression with minimal loss, effectively reducing the size of DAS data while maintaining its usability for traffic monitoring and geophysical analysis. The approach utilizes a patch size of 32x32 and a latent space of 50 for each data patch, resulting in an impressive compression rate of 512. However, it should be noted that a few car signals may experience less optimal reconstruction.

In addition to data compression, this approach also exhibits the capability to filter out undesired noise from the DAS data using the autoencoder. As part of the project's stretch goal, we aim to implement preprocessing steps such as de-medianing and de-trending with the autoencoder.


## Setup and Dependencies
We run the code on Google Colab for now. The following dependencies are required:

Python (version 3.10.11)
TensorFlow (version 2.12.0)
NumPy (version 1.22.4)
Matplotlib (version 3.7.1)

## Dataset
The preproceesed DAS data used in this project is stored on **Google Cloud**. The preprocessing steps include performing de-medianing, de-trending, and bandpass filtering. 

## Future Enhancements
The following improvements and enhancements can be considered for future iterations of the project:
1. Expand the training dataset to include more signal features.
2. Fine-tune hyperparameters on the enlarged dataset for improved model performance.
3. Investigate latent space size for a suitable trade-off between compression rate and reconstruction accuracy.
4. Refine model performance by adjusting the weight between rate and distortion in the loss function.
5.  Explore potential integration of preprocessing steps into the autoencoder's workflow.
6. Validate the model against an older DAS dataset collected on-campus or publicly available through PubDAS.

This project is licensed under the MIT License.
