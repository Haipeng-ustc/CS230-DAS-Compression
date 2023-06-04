
import glob
import os
import numpy as np
import tensorflow as tf

import sys
sys.path.append('../models/')
from model import train_das_model
from utils import prepare_das_dataset, load_patch_from_file

os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"

# Set the data for trainning
file_train = sorted(glob.glob('../data/train/*.npz'))
file_test = sorted(glob.glob('../data/test/*.npz'))

# file_path = '../data/original_data/'
# file_train = [file_path + 'smcity_sandhill_nc3100_ch3000-6100_nt3000_sf5_bp0.08-2_D2022-10-25_T10-07-56_p10min.npz', \
#               file_path + 'smcity_sandhill_nc3100_ch3000-6100_nt3000_sf5_bp0.08-2_D2022-10-26_T00-37-56_p10min.npz', \
#               file_path + 'smcity_rwc_nc4800_ch37500-42300_nt3000_sf5_bp0.08-2_D2022-10-25_T10-07-56_p10min.npz']

# file_train = [file_path + 'smcity_sandhill_nc3100_ch3000-6100_nt3000_sf5_bp0.08-2_D2022-10-25_T10-07-56_p10min.npz']
# file_test = [file_path + 'smcity_rwc_nc4800_ch37500-42300_nt3000_sf5_bp0.08-2_D2022-10-26_T00-37-56_p10min.npz']


# Setup parameters for generating patches
patch_size = 32          # size of each patch
train_ratio = 0.8        # ratio between training and dev 
shuffle = True           # shuffle all patches or not
overlap = False          # overlap or not when retriving patches, i.e., tile for half patch_size
add_noise = False         # add random noise to each patch or not
snr = 150                # value for signal-to-noise ratio

# Setup parameters for training model
lmbda = 2000         # weight for rate–distortion Lagrangian
latent_dims = 100    # layers in latent space
epochs = 50          # epoch
batch_size = 128     # batch size
learning_rate = 1e-3 # learning rate
validation_freq = 1  # validation frequency

for add_noise in [True, False]:
    # Retrive patches from save files
    input_patches, ouput_patches, factors = load_patch_from_file(file_train, 
                                                                patch_size, 
                                                                overlap = overlap, 
                                                                add_noise = add_noise, 
                                                                snr = snr)

    for lmbda in [500, 1000, 2000, 4000]:
        for latent_dims in [50, 100, 200, 400]:
            # Prepare the training and dev dataset into tensorflow
            train_dataset, dev_dataset = prepare_das_dataset(input_patches, 
                                                            ouput_patches, 
                                                            train_ratio = train_ratio, 
                                                            shuffle = shuffle)

            # Train the model
            trainer, history = train_das_model(
                                train_dataset,                         # training dataset
                                dev_dataset,                           # dev dataset
                                lmbda=lmbda,                           # weight for rate–distortion Lagrangian
                                latent_dims=latent_dims,               # layers in latent space
                                patch_size = patch_size,               # patch size of the image, must be dividable by 4
                                epochs=epochs,                         # epoch
                                batch_size=batch_size,                 # batch size
                                learning_rate=learning_rate,           # learning rate
                                validation_freq=validation_freq,       # validation frequency
                                launch_trainer = True)                 # launch the trainer
                                

            #history.history.keys()
            loss_list = ['loss', 'val_loss', 'rate_loss', 'distortion_loss']

            # Make a new fodler
            path = f"../results/model_patch{patch_size}_lmbda{lmbda}_latent{latent_dims}_learning_rate{learning_rate}_epochs{epochs}_noise{add_noise}"
            if not os.path.exists(path):
                os.mkdir(path)

            # Save the model
            trainer.save_weights(os.path.join(path, 'model'))

            # Save loss curve
            for l in (loss_list):
                np.savetxt(os.path.join(path, l), history.history[l])











##### For training the model with different patch size and learning rate #####

# for patch_size in [32, 48, 64, 128]:  
#     # Retrive patches from save files
#     input_patches, ouput_patches, factors = load_patch_from_file(file_train, 
#                                                                 patch_size, 
#                                                                 overlap = overlap, 
#                                                                 add_noise = add_noise, 
#                                                                 snr = snr)
#     # Prepare the training and dev dataset into tensorflow
#     train_dataset, dev_dataset = prepare_das_dataset(input_patches, 
#                                                     ouput_patches, 
#                                                     train_ratio = train_ratio, 
#                                                     shuffle = shuffle)

#     for learning_rate in [5e-4, 7.5e-4]:    #[1e-3, 5e-3, 1e-2]

#         # Train the model
#         trainer, history = train_das_model(
#                             train_dataset,                         # training dataset
#                             dev_dataset,                           # dev dataset
#                             lmbda=lmbda,                           # weight for rate–distortion Lagrangian
#                             latent_dims=latent_dims,               # layers in latent space
#                             patch_size = patch_size,               # patch size of the image, must be dividable by 4
#                             epochs=epochs,                         # epoch
#                             batch_size=batch_size,                 # batch size
#                             learning_rate=learning_rate,           # learning rate
#                             validation_freq=validation_freq,       # validation frequency
#                             launch_trainer = True)                 # launch the trainer
                            

#         #history.history.keys()
#         loss_list = ['loss', 'val_loss', 'rate_loss', 'distortion_loss']

#         # Make a new fodler
#         path = f"../results/model_patch{patch_size}_lmbda{lmbda}_latent{latent_dims}_learning_rate{learning_rate}_epochs{epochs}_noise{add_noise}"
#         if not os.path.exists(path):
#             os.mkdir(path)

#         # Save the model
#         trainer.save_weights(os.path.join(path, 'model'))

#         # Save loss curve
#         for l in (loss_list):
#             np.savetxt(os.path.join(path, l), history.history[l])

