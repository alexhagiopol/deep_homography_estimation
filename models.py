from keras.models import Sequential, Model
from keras.layers import GlobalAveragePooling2D, Dense, Flatten, Dropout, ELU, BatchNormalization, Lambda, merge, MaxPooling2D, Input, Activation
from keras.layers.convolutional import Convolution2D, Cropping2D
#from keras.utils.visualize_util import plot
from keras.optimizers import Adam
from keras.callbacks import Callback, RemoteMonitor
import keras.backend as K

def euclidean_distance(y_true, y_pred):
    return K.sqrt(K.maximum(K.sum(K.square(y_pred - y_true), axis=-1, keepdims=True), K.epsilon()))


def homography_regression_model():
    input_shape = (128, 128, 2)
    input_img = Input(shape=input_shape)

    x = Convolution2D(64, 3, 3, subsample=(1, 1), border_mode='same', name='conv1', activation='relu')(input_img)
    x = Convolution2D(64, 3, 3, subsample=(1, 1), border_mode='same', name='conv2', activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='pool1')(x)

    x = Convolution2D(64, 3, 3, subsample=(1, 1), border_mode='same', name='conv3', activation='relu')(x)
    x = Convolution2D(64, 3, 3, subsample=(1, 1), border_mode='same', name='conv4', activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='pool2')(x)

    x = Convolution2D(128, 3, 3, subsample=(1, 1), border_mode='same', name='conv5', activation='relu')(x)
    x = Convolution2D(128, 3, 3, subsample=(1, 1), border_mode='same', name='conv6', activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='pool3')(x)

    x = Convolution2D(128, 3, 3, subsample=(1, 1), border_mode='same', name='conv7', activation='relu')(x)
    x = Convolution2D(128, 3, 3, subsample=(1, 1), border_mode='same', name='conv8', activation='relu')(x)

    x = Flatten()(x)
    x = Dense(1024, name='FC1')(x)
    out = Dense(8, name='loss')(x)

    model = Model(input=input_img, output=[out])
    #plot(model, to_file='HomegraphyNet_Regression.png', show_shapes=True)

    model.compile(optimizer=Adam(lr=1e-3), loss=euclidean_distance)
    return model