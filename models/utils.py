import random

import numpy as np
import tensorflow as tf

from numba import jit, prange
from skimage.util import view_as_blocks
from time import time as cur_time

#TAC: added this for channel normailzation by |energy|
@jit(nopython=True, parallel=True)
def norm_channel(data):
  """
    Normalize channels to attenuate noise and acount for
    laser energy loss over distance.

    This is a channel by channel normalization.

    Function is JIT'ed for performance

    
    Paramters
    -----------
    data: DAS data

    Returns
    -----------
    data = data/mean(abs(data))
  """
  cnorm_data = data.copy()
  nchan = cnorm_data.shape[0]
  schan = 1./nchan
  for ic in prange(nchan):
    icnorm = schan*np.sum(np.abs(cnorm_data[ic,:]))
    cnorm = 1./icnorm
    cnorm_data[ic,:] *= cnorm

  return cnorm_data

@jit(forceobj=True, parallel=True)
def chan_xcorr(d0,d1,which_axis=-1):
  """
    Channel by Channel cross-correlation in parallel

    Function is JIT'ed for parallelization

    
    Paramters
    -----------
    d0: 2D ndarray with shape=(num_channels,num_time-samples)
    d1: 2D ndarray with shape=(num_channels,num_time-samples)
    which_axis: options are 0 or -1. 
    which_axis=0 --> correlate over channels for each time sample
    which_axis=-1 --> correlate over time for each channel

    Returns
    -----------
    xcorr_data = d0.cross_correlate_with(d1,channel_axis|time_axis)
  """
  xcorr_data = np.zeros_like(d0)

  if which_axis == 0:
    #for it in range(d0.shape[1]):
    for it in prange(d0.shape[1]):
      xcorr_data[:,it] = np.correlate(d0[:,it],d1[:,it],'same')
      enorm = 1/(np.max(xcorr_data[:,it])+1e-5)
      xcorr_data[:,it] *= enorm
  
  else:
    #for ic in range(d0.shape[0]):
    for ic in prange(d0.shape[0]):
      xcorr_data[ic] = np.correlate(d0[ic],d1[ic],'same')
      enorm = 1/(np.max(xcorr_data[ic,:])+1e-5)
      xcorr_data[ic,:] *= enorm

  return xcorr_data


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
    return image_patch + np.random.normal(loc=0, scale=np.sqrt(noise_power), size=image_patch.shape)


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
            #TAC: address shaping in prepare_das_dataset
            patch = patch.reshape(patch_size, patch_size, 1)
            patch.astype(np.float32)

            # normalize the patch to the scale of 255
            norm_factor = 1.0 / np.max(abs(patch)) * 255
            patch = patch * norm_factor
            patch[patch == np.nan] = 0.0

            # add noise to the input image patches if add_noise is True
            if add_noise:
                #input_patch = patch + add_noise_per_patch(patch, snr)
                #TAC: for compatibility
                input_patch = add_noise_per_patch(patch, snr)
            #else:
                #input_patch = patch

            input_patch = patch
            # append
            input_patches.append(input_patch)
            ouput_patches.append(patch)
            factors.append(norm_factor)

    return input_patches, ouput_patches, factors


# TAC: I had to modify this function for more data. It is now 30X faster
#      Most of the gain comes from vectorization
def extract_image_patches_fast(DAS_data, patch_size,  overlap = False, 
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
    def pixel_norm(x):
        norm = 255./np.nanmax(np.abs(x))
        y = x*norm
        return np.where(np.isnan(y),0,y), np.float32(norm)
    
    patch_norm = lambda x: (
        pixel_norm(x)
    )
    noise_func = lambda x,snr: (
        add_noise_per_patch(x, snr)
    )

    # create empty list
    input_patches = []
    ouput_patches = []
    factors = [] 

    # set overlapped patches or not
    if overlap:
        skip_patch_size = patch_size//2
    else:
        skip_patch_size = patch_size

      
    ht = patch_size
    wt = patch_size
    h_ht = ht//2 #TAC: half height 
    h_wt = wt//2 #TAC: half width 
    nht = int(ht*((DAS_data.shape[0]-0.5*ht)//ht))
    nwt = int(wt*((DAS_data.shape[1]-0.5*wt)//wt))

    pats = view_as_blocks(DAS_data[:nht,:nwt],block_shape=(ht,wt))
    pats = pats.reshape((pats.shape[0]*pats.shape[1],pats.shape[2],pats.shape[3]))
    rpat = None #TAC: for the shifted patches
    if overlap:
      rpat = view_as_blocks(DAS_data[h_ht:nht+h_ht,h_wt:nwt+h_wt],block_shape=(ht,wt))
      rpat = rpat.reshape((rpat.shape[0]*rpat.shape[1],rpat.shape[2],rpat.shape[3]))
      pats = np.concatenate((pats,rpat),axis=0) 
      del rpat

    ouput_patches, factors = zip(*list(map(patch_norm,pats)))
    ouput_patches = list(ouput_patches)
    factors = list(factors)
    
    input_patches = ouput_patches
    if add_noise:
      input_patches = list(map(noise_func,ouput_patches,[snr]*len(ouput_patches)))

    del pats
    return input_patches, ouput_patches, factors


def load_patch_from_file(file_list, patch_size, overlap = False, add_noise = False, snr = 10.):
    """Load the patches from files

        Parameters
        ----------
        file_list : list
            list of file names
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

    # Extract image patches (we can join lists together from different files)
    input_patches_all = []
    ouput_patches_all = []
    factors_all = []

    for file in file_list:
        print('Loading: %s'%(file))

        # Load the data
        #pdata = np.load(file)['pdata']
        pdata = norm_channel(np.load(file)['pdata'])

        # Extract patches and normalzation factor
        
        #input_patches, ouput_patches, factors = extract_image_patches(pdata, 
                                                    #patch_size, 
                                                    #add_noise = add_noise, 
                                                    #snr = snr, 
                                                    #overlap = overlap)
        #TAC: vectorized
        input_patches, ouput_patches, factors = extract_image_patches_fast(pdata, 
                                                    patch_size, 
                                                    add_noise = add_noise, 
                                                    snr = snr, 
                                                    overlap = overlap)

        # union patches from different data
        input_patches_all += input_patches
        ouput_patches_all += ouput_patches
        factors_all += factors

        del pdata

    print("Number of patches in total: ", len(input_patches_all))

    return input_patches_all, ouput_patches_all, factors_all


# TAC: too slow. now it's about 7-10x faster. Can support larger datasets
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
        np.random.seed(int(cur_time())) #TAC: added seed for the reals.
        #np.random.seed(13) #TAC: added seed for testing
        # Create a combined list of tuples
        combined_list = list(zip(input_patches, ouput_patches))
        # Shuffle the combined list
        np.random.shuffle(combined_list)
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
    #train_data_input = tf.convert_to_tensor(train_data_input, dtype=tf.float32)
    #train_data_ouput = tf.convert_to_tensor(train_data_ouput, dtype=tf.float32)
    #valid_data_input = tf.convert_to_tensor(valid_data_input, dtype=tf.float32)
    #valid_data_ouput = tf.convert_to_tensor(valid_data_ouput, dtype=tf.float32)
    train_data_input = tf.data.Dataset.from_tensor_slices(list(train_data_input))
    train_data_ouput = tf.data.Dataset.from_tensor_slices(list(train_data_ouput))
    valid_data_input = tf.data.Dataset.from_tensor_slices(list(valid_data_input))
    valid_data_ouput = tf.data.Dataset.from_tensor_slices(list(valid_data_ouput))

    #TAC: reshape for tensorflow (this is NOT compatible with Haipeng's original funcs)
    # tf_tensor does not have the same indexing scheme as numpy. So memory copies
    # are invoked if we add np.newaxis at axis 2, haven't tested if we aded it axis=0
    # or use transpose then add axis zero.
    pshape = (input_patches[0].shape[0],input_patches[0].shape[1],1)

    train_data_input = train_data_input.map(lambda x: tf.reshape(x, pshape)) 
    train_data_ouput = train_data_ouput.map(lambda x: tf.reshape(x, pshape)) 
    valid_data_input = valid_data_input.map(lambda x: tf.reshape(x, pshape)) 
    valid_data_ouput = valid_data_ouput.map(lambda x: tf.reshape(x, pshape)) 
    

    # Create TensorFlow datasets. The output is the same as the input if no 
    # noise are added. Otherwise their differences are the added random noises
    #train_dataset = tf.data.Dataset.from_tensor_slices((train_data_input, train_data_ouput))
    #valid_dataset = tf.data.Dataset.from_tensor_slices((valid_data_input, valid_data_ouput))
    #TAC now we have a Dataset already, so we zip them instead
    train_dataset = tf.data.Dataset.zip((train_data_input, train_data_ouput))
    valid_dataset = tf.data.Dataset.zip((valid_data_input, valid_data_ouput))

    # Print info
    print('\n')
    print('Number of train data: ', len(train_dataset))
    print('Number of valid data: ', len(valid_dataset))
    print(train_dataset)
    print(valid_dataset)

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


