import numpy as np
import keras
import keras.utils
from keras import layers
from sklearn.model_selection import ShuffleSplit
from keras.optimizers import Adam
import matplotlib.pyplot as plt
import tensorflow as tf
import keras.backend as K
from keras.models import Model
from keras.callbacks import ModelCheckpoint, Callback
from keras.layers import Conv2DTranspose, Input, Concatenate, Layer
from keras.layers import Dropout, LeakyReLU, Reshape
from capsule_layers import ConvCapsuleLayer
from keras.layers import Conv2D, BatchNormalization
from datetime import datetime




def mymodel(data, gt, num_epochs=25, class_num=16,return_all=False, folds=5):

    windowSize = 11

        class PixelSoftmax(Layer):

            def __init__(self, axis=-1, **kwargs):
                self.axis = axis
                super(PixelSoftmax, self).__init__(**kwargs)

            def get_config(self):
                config = super().get_config().copy()
                return config

            def build(self, input_shape):
                pass

            def call(self, x, mask=None):
                e = K.exp(x - K.max(x, axis=self.axis, keepdims=True))
                s = K.sum(e, axis=self.axis, keepdims=True)
                return e / s

            def get_output_shape_for(self, input_shape):
                return input_shape

        class statsLogger(Callback):

            def __init__(self):
                self.logs = []

            def on_epoch_end(self, epoch, logs):
                logs['epoch'] = epoch
                self.logs.append(logs)

            def get_config(self):
                config = super().get_config().copy()
                return config

        def conv_block(inputs, filters, kernel_size=(3, 3), strides=(1, 1)):
            x = Conv2D(filters, kernel_size=kernel_size, strides=strides, padding='same',
                                       use_bias=False)(inputs)
            x=BatchNormalization()(x)
            return keras.layers.ReLU(6.0)(x)

        def depthwise_conv_block(inputs, strides=(1, 1)):
            x = keras.layers.DepthwiseConv2D((3, 3), padding='same', strides=strides, use_bias=False)(inputs)
            x = BatchNormalization()(x)
            x = keras.layers.ReLU(6.0)(x)

            return keras.layers.ReLU(6.0)(x)


        def res_block(input_tensor, filters):
            residual = layers.Conv2D(filters,
                                     kernel_size=(1, 1),
                                     strides=1)(input_tensor)
            residual = layers.BatchNormalization()(residual)
            x = depthwise_conv_block(input_tensor, filters)
            x = depthwise_conv_block(x, filters)
            output = layers.Add()([residual, x])

            return output

        def my_loss(y_true, y_pred):
            def margin_loss(y_true, y_pred):
                L = y_true * K.square(K.maximum(0., 0.9 - y_pred)) + \
                    0.5 * (1 - y_true) * K.square(K.maximum(0., y_pred - 0.1))

                return K.mean(K.sum(L, 1))

            smooth = 1.

            def dice_coef(y_true, y_pred):
                y_true_f = K.flatten(y_true)
                y_pred_f = K.flatten(y_pred)
                intersection = K.sum(y_true_f * y_pred_f)
                return (2. * intersection + smooth) / (K.sum(y_true_f * y_true_f) + K.sum(y_pred_f * y_pred_f) + smooth)

            def dice_loss(y_true, y_pred):
                return 1. - dice_coef(y_true, y_pred)

            loss=K.sum(margin_loss(y_true, y_pred)+dice_loss(y_true, y_pred))

            return loss


        class TimeHistory(tf.keras.callbacks.Callback):
            def on_train_begin(self, logs={}):
                self.train_start_time = datetime.now()

            def on_train_end(self, logs={}):
                train_duration = datetime.now() - self.train_start_time
                print('Total training time:', train_duration)

        input_shape = x_train.shape[1:]
        img = Input(shape=input_shape)
        x = conv_block(img, 32, strides=(1, 1))
        x = conv_block(x, 64, strides=(1, 1))
        residual = layers.Conv2D(filters=64, kernel_size=(1, 1), strides=1,
                                 padding='same', use_bias=False)(x)
        x = depthwise_conv_block(x, 64)
        x = BatchNormalization()(x)
        x = Dropout(0.2)(x)
        x = res_block(x, filters=64)
        skip1 = x
        x = res_block(x, filters=64)
        x = layers.Add()([residual, x])
        skip2 = x
        x = depthwise_conv_block(x, 128)
        x = BatchNormalization()(x)
        x = Dropout(0.2)(x)

        _, H, W, C = x.get_shape()
        x = Reshape((H.value, W.value, 1, C.value))(x)
        PrimaryCapslayer = ConvCapsuleLayer(kernel_size=5, num_capsule=2, num_atoms=16, strides=1, padding='same',
                                        routings=1, name='PrimaryCapslayer')(x)
        DConvCapslayer1 = ConvCapsuleLayer(kernel_size=5, num_capsule=4, num_atoms=16, strides=1, padding='same',
                                        routings=3,
                                        name='DConvCapslayer1')(PrimaryCapslayer)
        DConvCapslayer2 = ConvCapsuleLayer(kernel_size=5, num_capsule=4, num_atoms=32, strides=1, padding='same',
                                        routings=3,
                                        name='DConvCapslayer2')(DConvCapslayer1)
        DConvCapslayer3 = ConvCapsuleLayer(kernel_size=1, num_capsule=1, num_atoms=16, strides=1, padding='same',
                                    routings=3,
                                    name='DConvCapslayer3')(DConvCapslayer2)
        x = keras.layers.Lambda(lambda x: keras.backend.squeeze(DConvCapslayer3, axis=-2))(DConvCapslayer3)
        x = Conv2DTranspose(128, kernel_size=(3, 3), strides=(1, 1), padding='same', name='deconv1', use_bias=False)(
            x)
        x = BatchNormalization()(x)
        x = LeakyReLU()(x)
        x = Dropout(0.2)(x)
        x = Concatenate(axis=-1)([x, skip2])
        x = Conv2DTranspose(64, kernel_size=(3, 3), strides=(1, 1), padding='same', name='deconv2', use_bias=False)(x)
        x = BatchNormalization()(x)
        x = LeakyReLU()(x)
        x = Dropout(0.2)(x)
        x = Concatenate(axis=-1)([x, skip1])
        x = Conv2D(class_num, kernel_size=(3, 3), strides=(windowSize, windowSize), padding='same', name='conv')(x)
        x = Reshape((1, class_num))(x)
        x = PixelSoftmax(axis=-1)(x)
        model = Model(inputs=img, outputs=x)
        y_test = y_test.reshape(y_test.shape[0], 1, class_num)
        y_train = y_train.reshape(y_train.shape[0], 1, class_num)
        filepath_name = "best-model.hdf5"
        history = statsLogger()
        opt = Adam(lr=0.001, decay=1e-4)
        ckpt = ModelCheckpoint(filepath=filepath_name,
                               save_best_only=True,
                               verbose=1,
                               monitor='loss')
        model.compile(loss=my_loss,
                      optimizer=opt,
                      metrics=['accuracy'])
        print(model.summary())
        hist = model.fit(x_train,
                         y_train,
                         batch_size=256,
                         epochs=num_epochs,
                         validation_data=(x_test, y_test),callbacks=[TimeHistory()])

        plt.plot(hist.history['loss'])
        plt.plot(hist.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'val'], loc='upper left')
        plt.show()
        acc_arr[q] = max(hist.history['acc'])
        if acc_arr[q] > best_fold_acc:
            best_x_train = x_train
            best_x_test = x_test
            best_y_train = y_train
            best_y_test = y_test
            best_model = model
            best_fold_acc = acc_arr[q]

    if (return_all):
        return (np.mean(acc_arr), best_x_train, best_x_test, best_y_train, best_y_test, best_model)
    else:
        return (np.mean(acc_arr))




