'''
This file contains the model for the DAS compression using neural networks (Autoencoder).
The original code is from TensorFlow Compression: 
https://github.com/tensorflow/compression

'''

import tensorflow as tf
import tensorflow_compression as tfc


def make_analysis_transform(latent_dims):
    """Creates the analysis (encoder) transform.
    
    Parameters:
    ----------
    latent_dims: int
        The dimension of the latent space representation.
    
    Returns:
    -------
    tf.keras.Sequential
        The analysis transform.
    """
    return tf.keras.Sequential([
      tf.keras.layers.Conv2D(20, 5, use_bias=True, strides=2, padding="same", activation="leaky_relu", name="conv_1"),
      tf.keras.layers.Conv2D(50, 5, use_bias=True, strides=2, padding="same", activation="leaky_relu", name="conv_2"),
      tf.keras.layers.Flatten(),
      tf.keras.layers.Dense(500, use_bias=True, activation="leaky_relu", name="fc_1"),
      tf.keras.layers.Dense(latent_dims, use_bias=True, activation=None, name="fc_2"),
    ], name="analysis_transform")


def make_synthesis_transform(patch_size):
    """Creates the synthesis (decoder) transform.
    
    Parameters:
    ----------
    patch_size: int
        The size of the patches that are being compressed.
    
    Returns:
    -------
    tf.keras.Sequential
        The synthesis transform.
    """

    if (patch_size // 4) * 4 != patch_size:
        raise ValueError('patch_size must be dividable ny 4!')

    return tf.keras.Sequential([
      tf.keras.layers.Dense(500, use_bias=True, activation="leaky_relu", name="fc_1"),
      tf.keras.layers.Dense(50 * (patch_size//4)**2, use_bias=True, activation="leaky_relu", name="fc_2"),
      tf.keras.layers.Reshape((patch_size//4, patch_size//4, 50)),
      tf.keras.layers.Conv2DTranspose(20, 5, use_bias=True, strides=2, padding="same", activation="leaky_relu", name="conv_1"),
      tf.keras.layers.Conv2DTranspose(1, 5, use_bias=True, strides=2, padding="same", activation="leaky_relu", name="conv_2"),
    ], name="synthesis_transform")


class DASCompressionTrainer(tf.keras.Model):
    """Model that trains a compressor/decompressor for DAS data."""

    def __init__(self, latent_dims, patch_size):
        super().__init__()
        self.analysis_transform = make_analysis_transform(latent_dims)
        self.synthesis_transform = make_synthesis_transform(patch_size)
        self.prior_log_scales = tf.Variable(tf.zeros((latent_dims,)))

    @property
    def prior(self):
        return tfc.NoisyLogistic(loc=0., scale=tf.exp(self.prior_log_scales))

    def call(self, x, training):
        """Computes rate and distortion losses."""

        # Note the scale applied to the data
        x = tf.cast(x, self.compute_dtype) / 255.

        # Compute latent space representation y, perturb it and model its entropy,
        y = self.analysis_transform(x)
        entropy_model = tfc.ContinuousBatchedEntropyModel(self.prior, coding_rank=1, compression=False)
        y_tilde, rate = entropy_model(y, training=training)

        # Compute the reconstructed pixel-level representation x_hat.
        x_tilde = self.synthesis_transform(y_tilde)

        # Average number of bits per image digit.
        rate = tf.reduce_mean(rate)

        # Mean absolute difference across pixels.
        distortion = tf.reduce_mean(abs(x - x_tilde))

        return dict(rate=rate, distortion=distortion)


def pass_through_loss(_, x):
    '''Dummy loss function that just passes through its input.'''

    return x


def add_rd_targets(image, label):
    '''Adds dummy rate and distortion targets to the label dictionary.'''

    return image, dict(rate=0., distortion=0.)


def train_das_model(training_dataset, validation_dataset, 
                    lmbda = 2000, 
                    latent_dims = 50, 
                    patch_size = 28, 
                    epochs = 15, 
                    batch_size = 128, 
                    learning_rate = 1e-3, 
                    validation_freq = 1,):
    
    '''Trains a DAS compression model.

    Parameters:
    ----------
    training_dataset: tf.data.Dataset
        The training dataset.
    validation_dataset: tf.data.Dataset

    lmbda: float
        Weight for rate-distortion Lagrangian.
    latent_dims: int
        The dimension of the latent space representation.
    patch_size: int
        The size of the patches that are being compressed.
    epochs: int
        Number of epochs to train for.
    batch_size: int
        Batch size.
    learning_rate: float
        Learning rate.
    validation_freq: int
        Validation frequency.
    '''

    # Create the model
    trainer = DASCompressionTrainer(latent_dims, patch_size)

    # Compile the model, just pass through rate and distortion as losses/metrics.
    trainer.compile(
        optimizer = tf.keras.optimizers.Adam(learning_rate = learning_rate),
        loss = dict(rate = pass_through_loss, distortion = pass_through_loss),
        metrics = dict(rate = pass_through_loss, distortion = pass_through_loss),
        loss_weights = dict(rate = 1., distortion = lmbda),
    )

    # Train the model
    history = trainer.fit(
        training_dataset.map(add_rd_targets).batch(batch_size).prefetch(8),
        epochs = epochs,
        validation_data = validation_dataset.map(add_rd_targets).batch(batch_size).cache(),
        validation_freq = validation_freq,
        verbose=1,
    )

    return trainer, history