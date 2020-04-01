import tensorflow as tf
import numpy as np
import tensorflow_addons as tfa
from tensorflow.keras import Sequential
from utils import pytorch_xavier_weight_factor, pytorch_kaiming_weight_factor

##################################################################################
# Initialization
##################################################################################

"""

pytorch xavier (gain)
https://pytorch.org/docs/stable/_modules/torch/nn/init.html

USE < tf.contrib.layers.variance_scaling_initializer() >
if uniform :
    factor = gain * gain
    mode = 'FAN_AVG'
else :
    factor = (gain * gain) / 1.3
    mode = 'FAN_AVG'

pytorch : trunc_stddev = gain * sqrt(2 / (fan_in + fan_out))
tensorflow  : trunc_stddev = sqrt(1.3 * factor * 2 / (fan_in + fan_out))

"""

"""
pytorch kaiming (a=0)
https://pytorch.org/docs/stable/_modules/torch/nn/init.html

if uniform :
    a = 0 -> gain = sqrt(2)
    factor = gain * gain
    mode='FAN_IN'
else :
    a = 0 -> gain = sqrt(2)
    factor = (gain * gain) / 1.3
    mode = 'FAN_OUT', # FAN_OUT is correct, but more use 'FAN_IN

pytorch : trunc_stddev = gain * sqrt(2 / fan_in)
tensorflow  : trunc_stddev = sqrt(1.3 * factor * 2 / fan_in)

"""

# Xavier : tf.initializers.GlorotUniform() or tf.initializers.GlorotNormal()
# He : tf.initializers.VarianceScaling()
# Normal : tf.initializers.RandomNormal(mean=0.0, stddev=0.02)
# Truncated_normal : tf.initializers.TruncatedNormal(mean=0.0, stddev=0.02)
# Orthogonal : tf.initializers.Orthogonal0.02)

##################################################################################
# Regularization
##################################################################################

# l2_decay : tf.keras.regularizers.l2(0.0001)
# orthogonal_regularizer : orthogonal_regularizer(0.0001) # orthogonal_regularizer_fully(0.0001)

# factor, mode, uniform = pytorch_xavier_weight_factor(gain=0.02, uniform=False)
# weight_initializer = tf.initializers.VarianceScaling(scale=factor, mode=mode, uniform=uniform)

weight_initializer = tf.initializers.RandomNormal(mean=0.0, stddev=0.02)
weight_regularizer = tf.keras.regularizers.l2(0.0001)
weight_regularizer_fully = tf.keras.regularizers.l2(0.0001)


##################################################################################
# Layers
##################################################################################

# padding='SAME' ======> pad = floor[ (kernel - stride) / 2 ]
class Conv(tf.keras.layers.Layer):
    def __init__(self, channels, kernel=3, stride=1, pad=0, pad_type='zero', use_bias=True, sn=False, name='Conv'):
        super(Conv, self).__init__(name=name)
        self.channels = channels
        self.kernel = kernel
        self.stride = stride
        self.pad = pad
        self.pad_type = pad_type
        self.use_bias = use_bias
        self.sn = sn

        if self.sn :
            self.conv = SpectralNormalization(tf.keras.layers.Conv2D(filters=self.channels, kernel_size=self.kernel,
                                                                     kernel_initializer=weight_initializer, kernel_regularizer=weight_regularizer,
                                                                     strides=self.stride, use_bias=self.use_bias), name='sn_' + self.name)
        else :
            self.conv = tf.keras.layers.Conv2D(filters=self.channels, kernel_size=self.kernel,
                                               kernel_initializer=weight_initializer, kernel_regularizer=weight_regularizer,
                                               strides=self.stride, use_bias=self.use_bias, name=self.name)

    def call(self, x, training=None, mask=None):
        if self.pad > 0:
            h = x.shape[1]
            if h % self.stride == 0:
                pad = self.pad * 2
            else:
                pad = max(self.kernel - (h % self.stride), 0)

            pad_top = pad // 2
            pad_bottom = pad - pad_top
            pad_left = pad // 2
            pad_right = pad - pad_left

            if self.pad_type == 'reflect':
                x = tf.pad(x, [[0, 0], [pad_top, pad_bottom], [pad_left, pad_right], [0, 0]], mode='REFLECT')
            else:
                x = tf.pad(x, [[0, 0], [pad_top, pad_bottom], [pad_left, pad_right], [0, 0]])

        x = self.conv(x)

        return x

class Deconv(tf.keras.layers.Layer):
    def __init__(self, channels, kernel=3, stride=2, padding='SAME', use_bias=True, sn=False, name='Deconv'):
        super(Deconv, self).__init__(name=name)
        self.channels = channels
        self.kernel = kernel
        self.stride = stride
        self.padding = padding
        self.use_bias = use_bias
        self.sn = sn

        if self.sn :
            self.deconv = SpectralNormalization(tf.keras.layers.Conv2DTranspose(filters=self.channels, kernel_size=self.kernel,
                                                                              kernel_initializer=weight_initializer, kernel_regularizer=weight_regularizer,
                                                                              strides=self.stride, padding=self.padding,
                                                                              use_bias=self.use_bias), name='sn_' + self.name)
        else :
            self.deconv = tf.keras.layers.Conv2DTranspose(filters=self.channels, kernel_size=self.kernel,
                                                        kernel_initializer=weight_initializer, kernel_regularizer=weight_regularizer,
                                                        strides=self.stride, padding=self.padding, use_bias=self.use_bias, name=self.name)

    def call(self, x, training=None, mask=None):
        x = self.deconv(x)

        return x


##################################################################################
# Blocks
##################################################################################

class ResBlock(tf.keras.layers.Layer):
    def __init__(self, channels, use_bias=True, sn=False, name='ResBlock'):
        super(ResBlock, self).__init__(name=name)
        self.channels = channels
        self.use_bias = use_bias
        self.sn = sn

        self.conv_0 = Conv(self.channels, kernel=3, stride=1, pad=1, pad_type='reflect', use_bias=self.use_bias, sn=self.sn, name='conv_0')
        self.ins_norm_0 = InstanceNorm(name='ins_norm_0')

        self.conv_1 = Conv(self.channels, kernel=3, stride=1, pad=1, pad_type='reflect', use_bias=self.use_bias, sn=self.sn,  name='conv_1')
        self.ins_norm_1 = InstanceNorm(name='ins_norm_1')

    def call(self, x_init, training=None, mask=None):
        with tf.name_scope(self.name):
            with tf.name_scope('res1'):
                x = self.conv_0(x_init)
                x = self.ins_norm_0(x, training=training)
                x = Relu(x)

            with tf.name_scope('res2'):
                x = self.conv_1(x)
                x = self.ins_norm_1(x, training=training)

            return x + x_init

##################################################################################
# Normalization
##################################################################################

class InstanceNorm(tf.keras.layers.Layer):
    def __init__(self, epsilon=1e-5, name='InstanceNorm'):
        super(InstanceNorm, self).__init__(name=name)
        self.epsilon = epsilon

    def call(self, x, training=None, mask=None):
        x = tfa.layers.normalizations.InstanceNormalization(epsilon=self.epsilon, scale=True, center=True, name=self.name)(x)

        return x

##################################################################################
# Activation Function
##################################################################################

def Leaky_Relu(x=None, alpha=0.01, name='leaky_relu'):
    # pytorch alpha is 0.01
    if x is None:
        return tf.keras.layers.LeakyReLU(alpha=alpha, name=name)
    else:
        return tf.keras.layers.LeakyReLU(alpha=alpha, name=name)(x)

def Relu(x=None, name='relu'):
    if x is None:
        return tf.keras.layers.Activation(tf.keras.activations.relu, name=name)

    else:
        return tf.keras.layers.Activation(tf.keras.activations.relu, name=name)(x)

def Tanh(x=None, name='tanh'):
    if x is None:
        return tf.keras.layers.Activation(tf.keras.activations.tanh, name=name)
    else:
        return tf.keras.layers.Activation(tf.keras.activations.tanh, name=name)(x)

##################################################################################
# Pooling & Resize
##################################################################################


##################################################################################
# Loss Function
##################################################################################

def L1_loss(x, y):
    loss = tf.reduce_mean(tf.abs(x - y))

    return loss


##################################################################################
# GAN Loss Function
##################################################################################
def discriminator_loss(gan_type, real_logit, fake_logit):

    real_loss = 0
    fake_loss = 0

    if gan_type.__contains__('wgan') :
        real_loss = -tf.reduce_mean(real_logit)
        fake_loss = tf.reduce_mean(fake_logit)

    if gan_type == 'lsgan' :
        real_loss = tf.reduce_mean(tf.math.squared_difference(real_logit, 1.0))
        fake_loss = tf.reduce_mean(tf.square(fake_logit))

    if gan_type == 'gan' or gan_type == 'dragan' :
        real_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(real_logit), logits=real_logit))
        fake_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(fake_logit), logits=fake_logit))

    if gan_type == 'hinge' :
        real_loss = tf.reduce_mean(Relu(1.0 - real_logit))
        fake_loss = tf.reduce_mean(Relu(1.0 + fake_logit))

    return real_loss + fake_loss

def generator_loss(gan_type, fake_logit):
    fake_loss = 0

    if gan_type.__contains__('wgan') :
        fake_loss = -tf.reduce_mean(fake_logit)

    if gan_type == 'lsgan' :
        fake_loss = tf.reduce_mean(tf.math.squared_difference(fake_logit, 1.0))

    if gan_type == 'gan' or gan_type == 'dragan' :
        fake_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(fake_logit), logits=fake_logit))

    if gan_type == 'hinge' :
        fake_loss = -tf.reduce_mean(fake_logit)


    return fake_loss

def regularization_loss(model):
    loss = tf.nn.scale_regularization_loss(model.losses)

    return loss

##################################################################################
# Class function
##################################################################################

class get_weight(tf.keras.layers.Layer):
    def __init__(self, w_shape, w_init, w_regular, w_trainable):
        super(get_weight, self).__init__()

        self.w_shape = w_shape
        self.w_init = w_init
        self.w_regular = w_regular
        self.w_trainable = w_trainable
        # self.w_name = w_name

    def call(self, inputs=None, training=None, mask=None):
        return self.add_weight(shape=self.w_shape, dtype=tf.float32,
                               initializer=self.w_init, regularizer=self.w_regular,
                               trainable=self.w_trainable)


class SpectralNormalization(tf.keras.layers.Wrapper):
    def __init__(self, layer, iteration=1, eps=1e-12, training=True, **kwargs):
        self.iteration = iteration
        self.eps = eps
        self.do_power_iteration = training
        if not isinstance(layer, tf.keras.layers.Layer):
            raise ValueError(
                'Please initialize `TimeDistributed` layer with a '
                '`Layer` instance. You passed: {input}'.format(input=layer))
        super(SpectralNormalization, self).__init__(layer, **kwargs)

    def build(self, input_shape=None):
        self.layer.build(input_shape)

        self.w = self.layer.kernel
        self.w_shape = self.w.shape.as_list()

        self.u = self.add_weight(shape=(1, self.w_shape[-1]),
                                 initializer=tf.initializers.TruncatedNormal(stddev=0.02),
                                 trainable=False,
                                 name=self.name + '_u',
                                 dtype=tf.float32, aggregation=tf.VariableAggregation.ONLY_FIRST_REPLICA)

        super(SpectralNormalization, self).build()

    def call(self, inputs, training=None, mask=None):
        self.update_weights()
        output = self.layer(inputs)
        # self.restore_weights()  # Restore weights because of this formula "W = W - alpha * W_SN`"
        return output

    def update_weights(self):
        w_reshaped = tf.reshape(self.w, [-1, self.w_shape[-1]])

        u_hat = self.u
        v_hat = None

        if self.do_power_iteration:
            for _ in range(self.iteration):
                v_ = tf.matmul(u_hat, tf.transpose(w_reshaped))
                v_hat = v_ / (tf.reduce_sum(v_ ** 2) ** 0.5 + self.eps)

                u_ = tf.matmul(v_hat, w_reshaped)
                u_hat = u_ / (tf.reduce_sum(u_ ** 2) ** 0.5 + self.eps)

        sigma = tf.matmul(tf.matmul(v_hat, w_reshaped), tf.transpose(u_hat))
        self.u.assign(u_hat)

        self.layer.kernel = self.w / sigma

    def restore_weights(self):

        self.layer.kernel = self.w