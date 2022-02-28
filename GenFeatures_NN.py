from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from keras.layers import Lambda, Input, Dense, Reshape, RepeatVector, Dropout
from keras.models import Model
from keras.datasets import mnist
from keras.losses import mse, binary_crossentropy
from keras.utils import plot_model
from keras import backend as K
from keras.constraints import unit_norm, max_norm

from scipy import stats
import pandas as pd
import numpy as np
import matplotlib
#import matplotlib.pyplot as plt
import argparse
import os
from sklearn.manifold import MDS
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


def sampling(args):
#    """Reparameterization trick by sampling fr an isotropic unit Gaussian.
    # Arguments:
#        args (tensor): mean and log of variance of Q(z|X)
    # Returns:
#        z (tensor): sampled latent vector
#    """

    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    # by default, random_normal has mean=0 and std=1.0
    epsilon = K.random_normal(shape=(batch, dim))
    thre = K.random_uniform(shape=(batch,1))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon
    
    
# LOAD DATA

training_feature = np.load('.npy')
X = training_feature
training_feature.shape

Y = np.load('.npy')
ground_truth_r = Y


np.random.seed(seed=0)

original_dim = training_feature.shape[1]
num_train = training_feature.shape[0]


# SCALING OF DATA

from sklearn.preprocessing import StandardScaler

PredictorScaler=StandardScaler()
TargetVarScaler=StandardScaler()

PredictorScalerFit = PredictorScaler.fit(X)
TargetVarScalerFit = TargetVarScaler.fit(Y.reshape(-1,1))

# GENERATING THE STANDARDIZED VALUES OF X AND Y

X = PredictorScalerFit.transform(X)
Y = TargetVarScalerFit.transform(Y.reshape(-1,1))

training_feature = X
ground_truth_r = Y

original_dim = training_feature.shape[1]
num_train = training_feature.shape[0]

## Build the Model

input_shape_x = (original_dim, )
input_shape_r = (1, )

intermediate_dim1 = original_dim-2
intermediate_dim2 = original_dim-4
batch_size = 10
latent_dim = original_dim-6
epochs = 1000
    
input_x = Input(shape=input_shape_x, name = 'encoder_input')
input_r = Input(shape=input_shape_r, name='ground_truth')
inputs_x_dropout = Dropout(0.25)(input_x)

# build encoder model
    
inter_x1 = Dense(intermediate_dim1, activation='tanh', name='encoder_intermediate')(inputs_x_dropout)
inter_x2 = Dense(intermediate_dim2, activation='tanh', name='encoder_intermediate_2')(inter_x1)


# posterior on Y; probablistic regressor

r_mean = Dense(1, name='r_mean')(inter_x2)
r_log_var = Dense(1, name='r_log_var')(inter_x2)
    
    
# q(z|x)

z_mean = Dense(latent_dim, name='z_mean')(inter_x2)
z_log_var = Dense(latent_dim, name='z_log_var')(inter_x2)   
    
# use reparameterization trick to push the sampling out as input

z = Lambda(sampling, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])
r = Lambda(sampling, output_shape=(1,), name='r')([r_mean, r_log_var])
    
# latent generator (simplified)

pz_mean = Dense(latent_dim, name='pz_mean',kernel_constraint=unit_norm())(r)
#pz_log_var = Dense(1, name='pz_log_var')(r)
    
# instantiate encoder model

encoder = Model([input_x,input_r], [z_mean, z_log_var, z, r_mean, r_log_var, r, pz_mean], name='encoder')
encoder.summary()


# build decoder model
    
latent_inputs = Input(shape=(latent_dim,), name='z_sampling')
inter_y1 = Dense(intermediate_dim2, activation='tanh')(latent_inputs)
inter_y2 = Dense(intermediate_dim1, activation='tanh')(inter_y1)
outputs = Dense(original_dim)(inter_y2)
           
# instantiate decoder model

decoder = Model(latent_inputs, outputs, name='decoder')
decoder.summary()


# instantiate VAE model

outputs = decoder(encoder([input_x,input_r])[2])
vae = Model([input_x,input_r], outputs, name='vae_mlp')
vae.summary()
    
##Customize Lost Function of the VAE Model

models = (encoder, decoder)

#reconstruction_loss = K.tf.divide(0.5*K.sum(K.square(inputs_x-outputs), axis=-1), K.exp(outputs_var)) + 0.5*original_dim*outputs_var
reconstruction_loss = mse(input_x,outputs)

#kl_loss = 1 + z_log_var - pz_log_var - tf.divide(K.square(z_mean-pz_mean),K.exp(pz_log_var)) - tf.divide(K.exp(z_log_var),K.exp(pz_log_var))
kl_loss = 1 + z_log_var - K.square(z_mean-pz_mean) - K.exp(z_log_var)
kl_loss = -0.5*K.sum(kl_loss, axis=-1)
label_loss = tf.divide(0.5*K.square(r_mean - input_r), K.exp(r_log_var)) +  0.5 * r_log_var

vae_loss = K.mean(reconstruction_loss+kl_loss+label_loss)

vae.add_loss(vae_loss)
vae.compile(optimizer='adam')
vae.summary()
vae.save_weights('weights.h5')


## Training the network with cross-validation

np.random.seed(0)
skf = StratifiedKFold(n_splits=5)
pred = np.zeros((ground_truth_r.shape))
fake = np.zeros((ground_truth_r.shape[0]))
fake[:114] = 1


# Run 5-fold CV

for train_idx, test_idx in skf.split(training_feature,fake):
    training_feature_sk = training_feature[train_idx,:]
    training_score = ground_truth_r[train_idx]
    
    testing_feature_sk = training_feature[test_idx,:]
    testing_score = ground_truth_r[test_idx]    
    
    vae.load_weights('weights.h5')
    vae.fit([training_feature_sk,training_score],
             epochs=epochs,
             batch_size=batch_size,
             verbose = 0)
    
    [z_mean, z_log_var, z, r_mean, r_log_var, r_vae, pz_mean] = encoder.predict([testing_feature_sk,testing_score],batch_size=batch_size)
    pred[test_idx] = r_mean[:,0].reshape(-1,1)
    np.save('.npy',pred)
        
        
## Validation
#The mean squared error
print("Mean squared error: %.3f" % mean_squared_error(ground_truth_r, pred))
# Explained variance score: 1 is perfect prediction
print('R2 Variance score: %.3f' % r2_score(ground_truth_r, pred))

