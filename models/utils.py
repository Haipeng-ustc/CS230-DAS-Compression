import random

import numpy as np
import tensorflow as tf


def add_noise_per_patch(image_patch, snr):
    """ Add noise to a patch of an image.

    Parameters
    ----------
    image_patch : numpy array
        A patch of an image.
    snr : float
        Signal to noise ratio.

    Returns
    -------
    noisy_patch : numpy array
        A patch of an image with added noise.
    """

    # Compute the power
    signal_power = np.mean(image_patch ** 2)
    noise_power = signal_power / snr

    # Generate Gaussian noise with zero mean and noise_power standard deviation
    noise = np.random.normal(loc=0, scale=np.sqrt(noise_power), size=image_patch.shape)
    noisy_patch = np.zeros_like(image_patch)
    noisy_patch = noisy_patch + noise

    return noisy_patch


def extract_image_patches(DAS_data, patch_size,  overlap = False, 
                          add_noise = False, snr = 0.5):
    """ Divide the DAS data (channels by time samples) into small
      patches with the size of patch_size

        Parameters
        ----------
        DAS_data : numpy
            DAS data (channels by time samples)
        patch_size : int
            patch size
        overlap : boolean
            overlapped patches or not
        add_noise : boolean
            add noise or not
        snr : float
            signal to noise ratio
        
        Returns
        -------
        input_patches : list
            list of input patches
        ouput_patches : list
            list of output patches
        factors : list
            list of normalization factors for each patch, which is used to 
            recover the original scale of the data
    """

    # get the shape
    height, width, = DAS_data.shape

    # create empty list
    input_patches = []
    ouput_patches = []
    factors = []

    # set overlapped patches or not
    if overlap:
        skip_patch_size = patch_size//2
    else:
        skip_patch_size = patch_size

    # extract the patch by looping the entire DAS section
    for y in range(0, height - patch_size + 1, skip_patch_size):
        for x in range(0, width - patch_size + 1, skip_patch_size):

            # extract the patch
            patch = DAS_data[y:y+patch_size, x:x+patch_size]

            # post-process the patch
            patch = patch.reshape(patch_size, patch_size, 1)
            patch.astype(np.float32)

            # normalize the patch to the scale of 255
            norm_factor = 1.0 / np.max(abs(patch)) * 255
            patch = patch * norm_factor
            patch[patch == np.nan] = 0.0

            # add noise to the input image patches if add_noise is True
            if add_noise:
                input_patch = patch + add_noise_per_patch(patch, snr)
            else:
                input_patch = np.copy(patch)

            # append
            input_patches.append(input_patch)
            ouput_patches.append(patch)
            factors.append(norm_factor)

    return input_patches, ouput_patches, factors


def prepare_das_dataset(input_patches, ouput_patches, train_ratio = 0.8, shuffle = True):
    """Prepare the DAS dataset for training and validation

        Parameters
        ----------
        input_patches : list
            list of input patches
        ouput_patches : list
            list of output patches
        train_ratio : float
            ratio of training data
        shuffle : boolean
            shuffle the data or not
        
        Returns
        -------
        train_dataset : tf.tensor
            training data
        valid_dataset : tf.tensor
            validation data
    """
    
    # Shuffle the data randomly (we can also do this using TensorFlow)
    if shuffle == True:
        # Create a combined list of tuples
        combined_list = list(zip(input_patches, ouput_patches))
        # Shuffle the combined list
        random.shuffle(combined_list)
        # Unzip the shuffled list to retrieve the shuffled lists
        input_patches, ouput_patches = zip(*combined_list)

    # Calculate the split index based on the train_ratio
    split_index = int(len(input_patches) * train_ratio)

    # Split the data into training and validation sets
    train_data_input = input_patches[:split_index]
    train_data_ouput = ouput_patches[:split_index]
    valid_data_input = input_patches[split_index:]
    valid_data_ouput = ouput_patches[split_index:]

    # Convert the Python lists to TensorFlow tensors
    train_data_input = tf.convert_to_tensor(train_data_input, dtype=tf.float32)
    train_data_ouput = tf.convert_to_tensor(train_data_ouput, dtype=tf.float32)
    valid_data_input = tf.convert_to_tensor(valid_data_input, dtype=tf.float32)
    valid_data_ouput = tf.convert_to_tensor(valid_data_ouput, dtype=tf.float32)

    # Create TensorFlow datasets. The output is the same as the input if no 
    # noise are added. Otherwise their differences are the added random noises
    train_dataset = tf.data.Dataset.from_tensor_slices((train_data_input, train_data_ouput))
    valid_dataset = tf.data.Dataset.from_tensor_slices((valid_data_input, valid_data_ouput))

    return train_dataset, valid_dataset


def reconstruct_image_from_patches(patches, factors, original_shape, patch_size, overlap = False):
    """ Reconstruct the original image from patches.

        Parameters
        ----------
        patches : list
            list of patches 
        factors : list
            list of normalization factors for each patch, which is used to
            recover the original scale of the data
        original_shape : tuple
            original shape of the image
        patch_size : int
            patch size
        overlap : boolean
            overlapped patches or not

        Returns
        -------
        reconstructed_image : numpy
            reconstructed image
    """

    # get the shape
    height, width = original_shape
    reconstructed_image = np.zeros(original_shape)
    patch_index = 0

    # set overlapped patches or not
    if overlap:
      skip_patch_size = patch_size//2
    else:
      skip_patch_size = patch_size

    # reconstruct the image by looping the entire DAS section
    for y in range(0, height - patch_size + 1, skip_patch_size):
        for x in range(0, width - patch_size + 1, skip_patch_size):

            # get the patch
            patch = patches[patch_index]
            patch = np.reshape(patch, (patch_size, patch_size))

            # restore the normalization
            patch = patch / factors[patch_index]
            patch[patch==np.nan] = 0.0

            # put the patch back to where it belongs to
            reconstructed_image[y:y + patch_size, x:x + patch_size] = patch
            patch_index += 1

    return reconstructed_image