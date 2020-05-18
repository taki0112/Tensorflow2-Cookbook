import tensorflow as tf
import numpy as np
import tensorflow_addons as tfa
from tensorflow.keras import Sequential
from utils import pytorch_xavier_weight_factor, pytorch_kaiming_weight_factor

##################################################################################
# Initialization
##################################################################################

# Xavier : tf.initializers.GlorotUniform() or tf.initializers.GlorotNormal()
# He : tf.initializers.he_normal() or tf.initializers.he_uniform()
# Normal : tf.initializers.RandomNormal(mean=0.0, stddev=0.02)
# Truncated_normal : tf.initializers.TruncatedNormal(mean=0.0, stddev=0.02)
# Orthogonal : tf.initializers.Orthogonal0.02)

##################################################################################
# Regularization
##################################################################################

# l2_decay : tf.keras.regularizers.l2(0.0001)
# orthogonal_regularizer : orthogonal_regularizer(0.0001) # orthogonal_regularizer_fully(0.0001)

# factor, mode = pytorch_xavier_weight_factor(gain=0.02, uniform=False)
# distribution = "untruncated_normal"
# distribution in {"uniform", "truncated_normal", "untruncated_normal"}
# weight_initializer = tf.initializers.VarianceScaling(scale=factor, mode=mode, distribution=distribution)

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

        if self.sn:
            self.conv = SpectralNormalization(tf.keras.layers.Conv2D(filters=self.channels, kernel_size=self.kernel,
                                                                     kernel_initializer=weight_initializer,
                                                                     kernel_regularizer=weight_regularizer,
                                                                     strides=self.stride, use_bias=self.use_bias),
                                              name='sn_' + self.name)
        else:
            self.conv = tf.keras.layers.Conv2D(filters=self.channels, kernel_size=self.kernel,
                                               kernel_initializer=weight_initializer,
                                               kernel_regularizer=weight_regularizer,
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


class PartialConv(tf.keras.layers.Layer):
    def __init__(self, channels, kernel=4, stride=2, sn=False, name='PartialConv'):
        super(PartialConv, self).__init__(name=name)
        self.channels = channels
        self.kernel = kernel
        self.stride = stride
        self.sn = sn

        self.slide_window = kernel * kernel
        self.mask_conv = tf.keras.layers.Conv2D(filters=1,
                                                kernel_size=self.kernel,
                                                kernel_initializer=tf.constant_initializer(1.0),
                                                strides=self.stride, padding='SAME', use_bias=False, trainable=False,
                                                name='mask_conv')
        if self.sn:
            self.conv = SpectralNormalization(tf.keras.layers.Conv2D(filters=self.channels, kernel_size=self.kernel,
                                                                     kernel_initializer=weight_initializer,
                                                                     kernel_regularizer=weight_regularizer,
                                                                     strides=self.stride, padding='SAME',
                                                                     use_bias=False),
                                              name='sn_' + self.name)
        else:
            self.conv = tf.keras.layers.Conv2D(filters=self.channels, kernel_size=self.kernel,
                                               kernel_initializer=weight_initializer,
                                               kernel_regularizer=weight_regularizer,
                                               strides=self.stride, padding='SAME', use_bias=False, name=self.name)

    def build(self, input_shape):
        h, w = input_shape[1], input_shape[2]
        self.mask = tf.ones(shape=[1, h, w, 1])

    def call(self, x, training=None, mask=None):
        update_mask = self.mask_conv(self.mask)

        mask_ratio = self.slide_window / (update_mask + 1e-8)
        update_mask = tf.clip_by_value(update_mask, 0.0, 1.0)
        mask_ratio = mask_ratio * update_mask

        x = self.conv(x)
        x = x * mask_ratio

        return x


class DilateConv(tf.keras.layers.Layer):
    def __init__(self, channels, kernel=3, rate=2, padding='SAME', use_bias=True, sn=False, name='DilateConv'):
        super(DilateConv, self).__init__(name=name)
        self.channels = channels
        self.kernel = kernel
        self.rate = rate
        self.padding = padding
        self.use_bias = use_bias
        self.sn = sn

        if self.sn:
            self.conv = SpectralNormalization(tf.keras.layers.Conv2D(self.channels, kernel_size=self.kernel,
                                                                     kernel_initializer=weight_initializer,
                                                                     kernel_regularizer=weight_regularizer,
                                                                     strides=1, padding=self.padding,
                                                                     use_bias=self.use_bias,
                                                                     dilation_rate=self.rate), name='sn_' + self.name)
        else:
            self.conv = tf.keras.layers.Conv2D(self.channels, kernel_size=self.kernel,
                                               kernel_initializer=weight_initializer,
                                               kernel_regularizer=weight_regularizer,
                                               strides=1, padding=self.padding, use_bias=self.use_bias,
                                               dilation_rate=self.rate, name=self.name)

    def call(self, x, training=None, mask=None):

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

        if self.sn:
            self.deconv = SpectralNormalization(
                tf.keras.layers.Conv2DTranspose(filters=self.channels, kernel_size=self.kernel,
                                                kernel_initializer=weight_initializer,
                                                kernel_regularizer=weight_regularizer,
                                                strides=self.stride, padding=self.padding,
                                                use_bias=self.use_bias), name='sn_' + self.name)
        else:
            self.deconv = tf.keras.layers.Conv2DTranspose(filters=self.channels, kernel_size=self.kernel,
                                                          kernel_initializer=weight_initializer,
                                                          kernel_regularizer=weight_regularizer,
                                                          strides=self.stride, padding=self.padding,
                                                          use_bias=self.use_bias, name=self.name)

    def call(self, x, training=None, mask=None):
        x = self.deconv(x)

        return x


class ConvPixelShuffleUp(tf.keras.layers.Layer):
    def __init__(self, scale_factor=2, use_bias=True, sn=False, name='ConvPixelShuffleUp'):
        super(ConvPixelShuffleUp, self).__init__(name=name)
        self.scale_factor = scale_factor
        self.use_bias = use_bias
        self.sn = sn

    def build(self, input_shape):
        self.channels = input_shape[-1] * (self.scale_factor ** 2)

        if self.sn:
            self.conv = SpectralNormalization(tf.keras.layers.Conv2D(filters=self.channels, kernel_size=1,
                                                                     kernel_initializer=weight_initializer,
                                                                     kernel_regularizer=weight_regularizer,
                                                                     strides=1, use_bias=self.use_bias),
                                              name='sn_' + self.name)
        else:
            self.conv = tf.keras.layers.Conv2D(filters=self.channels, kernel_size=1,
                                               kernel_initializer=weight_initializer,
                                               kernel_regularizer=weight_regularizer,
                                               strides=1, use_bias=self.use_bias, name=self.name)

    def call(self, x, training=None, mask=None):
        x = self.conv(x)
        x = tf.nn.depth_to_space(x, block_size=self.scale_factor)

        return x


class ConvPixelShuffleDown(tf.keras.layers.Layer):
    def __init__(self, scale_factor=2, use_bias=True, sn=False, name='ConvPixelShuffleDown'):
        super(ConvPixelShuffleDown, self).__init__(name=name)
        self.scale_factor = scale_factor
        self.use_bias = use_bias
        self.sn = sn

    def build(self, input_shape):
        self.channels = input_shape[-1] // (self.scale_factor ** 2)
        assert self.channels > 0

        if self.sn:
            self.conv = SpectralNormalization(tf.keras.layers.Conv2D(filters=self.channels, kernel_size=1,
                                                                     kernel_initializer=weight_initializer,
                                                                     kernel_regularizer=weight_regularizer,
                                                                     strides=1, use_bias=self.use_bias),
                                              name='sn_' + self.name)
        else:
            self.conv = tf.keras.layers.Conv2D(filters=self.channels, kernel_size=1,
                                               kernel_initializer=weight_initializer,
                                               kernel_regularizer=weight_regularizer,
                                               strides=1, use_bias=self.use_bias, name=self.name)

    def call(self, x, training=None, mask=None):
        x = self.conv(x)
        x = tf.nn.space_to_depth(x, block_size=self.scale_factor)

        return x


class FullyConnected(tf.keras.layers.Layer):
    def __init__(self, units, use_bias=True, sn=False, name='FullyConnected'):
        super(FullyConnected, self).__init__(name=name)
        self.units = units
        self.use_bias = use_bias
        self.sn = sn

        if self.sn:
            self.fc = SpectralNormalization(tf.keras.layers.Dense(self.units,
                                                                  kernel_initializer=weight_initializer,
                                                                  kernel_regularizer=weight_regularizer_fully,
                                                                  use_bias=self.use_bias), name='sn_' + self.name)
        else:
            self.fc = tf.keras.layers.Dense(self.units,
                                            kernel_initializer=weight_initializer,
                                            kernel_regularizer=weight_regularizer_fully,
                                            use_bias=self.use_bias, name=self.name)

    def call(self, x, training=None, mask=None):
        x = Flatten(x)
        x = self.fc(x)

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

        self.conv_0 = Conv(self.channels, kernel=3, stride=1, pad=1, pad_type='reflect', use_bias=self.use_bias,
                           sn=self.sn, name='conv_0')
        self.batch_norm_0 = BatchNorm(momentum=0.9, epsilon=1e-5, name='batch_norm_0')

        self.conv_1 = Conv(self.channels, kernel=3, stride=1, pad=1, pad_type='reflect', use_bias=self.use_bias,
                           sn=self.sn, name='conv_1')
        self.batch_norm_1 = BatchNorm(momentum=0.9, epsilon=1e-5, name='batch_norm_1')

    def build(self, input_shape):

        self.skip_flag = self.channels != input_shape[-1]
        if self.skip_flag:
            self.skip_conv = Conv(self.channels, kernel=1, stride=1, use_bias=self.use_bias, sn=self.sn,
                                  name='skip_conv')

    def call(self, x_init, training=None, mask=None):
        with tf.name_scope(self.name):
            with tf.name_scope('res1'):
                x = self.conv_0(x_init)
                x = self.batch_norm_0(x, training=training)
                x = Relu(x)

            with tf.name_scope('res2'):
                x = self.conv_1(x)
                x = self.batch_norm_1(x, training=training)

            if self.skip_flag:
                x_init = self.skip_conv(x_init)
                return Relu(x + x_init)

            else:
                return x + x_init


class ResBlockUpCondition(tf.keras.layers.Layer):
    def __init__(self, channels, use_bias=True, sn=False, name='ResBlockUpCondition'):
        super(ResBlockUpCondition, self).__init__(name=name)
        self.channels = channels
        self.use_bias = use_bias
        self.sn = sn

        self.deconv_0 = Deconv(self.channels, kernel=3, stride=2, use_bias=self.use_bias, sn=self.sn, name='deconv_0')
        self.cond_bn_0 = ConditionBatchNorm(momentum=0.9, epsilon=1e-5, name='cond_bn_0')

        self.deconv_1 = Deconv(self.channels, kernel=3, stride=1, use_bias=self.use_bias, sn=self.sn, name='deconv_1')
        self.cond_bn_1 = ConditionBatchNorm(momentum=0.9, epsilon=1e-5, name='cond_bn_1')

        self.skip_deconv = Deconv(self.channels, kernel=3, stride=2, use_bias=self.use_bias, sn=self.sn,
                                  name='skip_deconv')

    def call(self, inputs, training=True, mask=None):
        # See https://github.com/taki0112/BigGAN-Tensorflow
        with tf.name_scope(self.name):
            x_init, z = inputs[0], inputs[1]
            with tf.name_scope('res1'):
                x = self.deconv_0(x_init)
                x = self.cond_bn_0([x, z], training=training)
                x = Relu(x)

            with tf.name_scope('res2'):
                x = self.deconv_1(x)
                x = self.cond_bn_1([x, z], training=training)

            with tf.name_scope('skip'):
                x_init = self.skip_deconv(x_init)

            return Relu(x + x_init)


class ResBlockUp(tf.keras.layers.Layer):
    def __init__(self, channels, use_bias=True, sn=False, name='ResBlockUp'):
        super(ResBlockUp, self).__init__(name=name)
        self.channels = channels
        self.use_bias = use_bias
        self.sn = sn

        self.deconv_0 = Deconv(self.channels, kernel=3, stride=2, use_bias=self.use_bias, sn=self.sn, name='deconv_0')
        self.batch_norm_0 = BatchNorm(momentum=0.9, epsilon=1e-5, name='batch_norm_0')

        self.deconv_1 = Deconv(self.channels, kernel=3, stride=1, use_bias=self.use_bias, sn=self.sn, name='deconv_1')
        self.batch_norm_1 = BatchNorm(momentum=0.9, epsilon=1e-5, name='batch_norm_1')

        self.skip_deconv = Deconv(self.channels, kernel=3, stride=2, use_bias=self.use_bias, sn=self.sn,
                                  name='skip_deconv')

    def call(self, x_init, training=None, mask=None):
        with tf.name_scope(self.name):
            with tf.name_scope('res1'):
                x = self.deconv_0(x_init)
                x = self.batch_norm_0(x, training=training)
                x = Relu(x)

            with tf.name_scope('res2'):
                x = self.deconv_1(x)
                x = self.batch_norm_1(x, training=training)

            with tf.name_scope('skip'):
                x_init = self.skip_deconv(x_init)

            return Relu(x + x_init)


class ResBlockDown(tf.keras.layers.Layer):
    def __init__(self, channels, use_bias=True, sn=False, name='ResBlockDown'):
        super(ResBlockDown, self).__init__(name=name)
        self.channels = channels
        self.use_bias = use_bias
        self.sn = sn

        self.conv_0 = Conv(self.channels, kernel=3, stride=2, use_bias=self.use_bias, sn=self.sn, name='conv_0')
        self.batch_norm_0 = BatchNorm(momentum=0.9, epsilon=1e-5, name='batch_norm_0')

        self.conv_1 = Conv(self.channels, kernel=3, stride=1, use_bias=self.use_bias, sn=self.sn, name='conv_1')
        self.batch_norm_1 = BatchNorm(momentum=0.9, epsilon=1e-5, name='batch_norm_1')

        self.skip_conv = Conv(self.channels, kernel=3, stride=2, use_bias=self.use_bias, sn=self.sn, name='skip_conv')

    def call(self, x_init, training=None, mask=None):
        with tf.name_scope(self.name):
            with tf.name_scope('res1'):
                x = self.conv_0(x_init)
                x = self.batch_norm_0(x, training=training)
                x = Relu(x)

            with tf.name_scope('res2'):
                x = self.conv_1(x)
                x = self.batch_norm_1(x, training=training)

            with tf.name_scope('skip'):
                x_init = self.skip_conv(x_init)

            return Relu(x + x_init)


class DenseBlock(tf.keras.layers.Layer):
    def __init__(self, channels, n_db=6, use_bias=True, sn=False, name='DenseBlock'):
        super(DenseBlock, self).__init__(name=name)
        self.channels = channels
        self.n_db = n_db
        self.use_bias = use_bias
        self.sn = sn

        self.bottle_necks = []
        for i in range(self.n_db):
            blocks = []
            blocks += [Conv(4 * self.channels, kernel=1, stride=1, use_bias=self.use_bias, sn=self.sn, name='conv_0')]
            blocks += [BatchNorm(momentum=0.9, epsilon=1e-5, name='batch_norm_0')]
            blocks += [tf.keras.layers.ReLU(name='Relu')]

            blocks += [
                Conv(self.channels, kernel=3, stride=1, pad=1, use_bias=self.use_bias, sn=self.sn, name='conv_1')]
            blocks += [BatchNorm(momentum=0.9, epsilon=1e-5, name='batch_norm_1')]
            blocks = Sequential(blocks)

            self.bottle_necks.append(blocks)

    def call(self, x_init, training=None, mask=None):
        with tf.name_scope(self.name):
            layers = []
            layers.append(x_init)

            with tf.name_scope('bottle_neck_0'):
                x = self.bottle_necks[0](x_init, training=training)

                layers.append(x)

            for i in range(1, self.n_db):
                with tf.name_scope('bottle_neck_' + str(i)):
                    x = tf.concat(layers, axis=-1)

                    x = self.bottle_necks[i](x, training=training)

                    layers.append(x)

            x = tf.concat(layers, axis=-1)

            return x


class ResDenseBlock(tf.keras.layers.Layer):
    def __init__(self, channels, n_rdb=20, n_rdb_inter=6, use_bias=True, sn=False, name='ResDenseBlock'):
        super(ResDenseBlock, self).__init__(name=name)
        self.channels = channels
        self.n_rdb = n_rdb
        self.n_rdb_inter = n_rdb_inter
        self.use_bias = use_bias
        self.sn = sn

        self.RDB_blocks = []
        self.RDB_inter_blocks = []

        for i in range(self.n_rdb):
            blocks = []
            blocks += [
                Conv(self.channels, kernel=3, stride=1, pad=1, use_bias=self.use_bias, sn=self.sn, name='conv_0')]
            blocks += [BatchNorm(momentum=0.9, epsilon=1e-5, name='batch_norm_0')]
            blocks += [Relu()]
            blocks = Sequential(blocks)

            self.RDB_blocks.append(blocks)

            for j in range(1, self.n_rdb_inter):
                blocks = []
                blocks += [Conv(self.channels, kernel=3, stride=1, pad=1, use_bias=self.use_bias, sn=self.sn,
                                name='conv_' + str(j))]
                blocks += [BatchNorm(momentum=0.9, epsilon=1e-5, name='batch_norm_' + str(j))]

                blocks = Sequential(blocks)
                self.RDB_inter_blocks.append(blocks)

        self.local_fusion_conv = Conv(self.channels, kernel=1, stride=1, use_bias=self.use_bias, sn=self.sn,
                                      name='local_fusion_conv')
        self.GFF_1x1_conv = Conv(self.channels, kernel=1, stride=1, use_bias=self.use_bias, sn=self.sn,
                                 name='GFF_1x1_conv')
        self.GFF_3x3_conv = Conv(self.channels, kernel=3, stride=1, pad=1, use_bias=self.use_bias, sn=self.sn,
                                 name='GFF_3x3_conv')

    def build(self, input_shape):
        self.skip_flag = self.channels != input_shape[-1]
        if self.skip_flag:
            self.local_skip_conv = Conv(self.channels, kernel=1, stride=1, use_bias=self.use_bias, sn=self.sn,
                                        name='local_skip_conv')
            self.global_skip_conv = Conv(self.channels, kernel=1, stride=1, use_bias=self.use_bias, sn=self.sn,
                                         name='global_skip_conv')

    def call(self, x_init, training=None, mask=None):
        with tf.name_scope(self.name):
            RDBs = []
            x_input = x_init

            """
            n_rdb = 20 ( RDB number )
            n_rdb_conv = 6 ( per RDB conv layer )
            """

            for i in range(self.n_rdb):
                with tf.name_scope('RDB_' + str(i)):
                    layers = []
                    layers.append(x_init)

                    x = self.RDB_blocks[i](x_init, training=training)

                    layers.append(x)

                    for j in range(1, self.n_rdb_inter):
                        x = tf.concat(layers, axis=-1)

                        x = self.RDB_inter_blocks[j](x, training=training)

                        layers.append(x)

                    # Local feature fusion
                    x = tf.concat(layers, axis=-1)
                    x = self.local_fusion_conv(x)

                    # Local residual learning
                    if self.skip_flag:
                        x_init = self.local_skip_conv(x_init)
                        x = Relu(x + x_init)
                    else:
                        x = x_init + x

                    RDBs.append(x)
                    x_init = x

            x = tf.concat(RDBs, axis=-1)
            x = self.GFF_1x1_conv(x)
            x = self.GFF_3x3_conv(x)

            # Global residual learning
            if self.skip_flag:
                x_input = self.global_skip_conv(x_input)
                return Relu(x + x_input)
            else:
                return x + x_input


class SelfAttention(tf.keras.layers.Layer):
    def __init__(self, use_bias=True, sn=False, name='SelfAttention'):
        super(SelfAttention, self).__init__(name=name)
        self.use_bias = use_bias
        self.sn = sn

    def build(self, input_shape):
        self.channels = input_shape[-1]
        self.gamma = tf.Variable(initial_value=tf.constant(0.0, shape=[1]), name='gamma')

        self.f_conv = Conv(self.channels // 8, kernel=1, stride=1, use_bias=self.use_bias, sn=self.sn, name='f_conv')
        self.g_conv = Conv(self.channels // 8, kernel=1, stride=1, use_bias=self.use_bias, sn=self.sn, name='g_conv')
        self.h_conv = Conv(self.channels, kernel=1, stride=1, use_bias=self.use_bias, sn=self.sn, name='h_conv')

    def call(self, x, training=None, mask=None):
        with tf.name_scope(self.name):
            f = self.f_conv(x)  # [bs, h, w, c']
            g = self.g_conv(x)  # [bs, h, w, c']
            h = self.h_conv(x)  # [bs, h, w, c]

            # N = h * w
            s = tf.matmul(hw_flatten(g), hw_flatten(f), transpose_b=True)  # [bs, N, N]

            beta = tf.nn.softmax(s)  # attention map

            o = tf.matmul(beta, hw_flatten(h))

            o = tf.reshape(o, shape=x.shape)
            x = self.gamma * o + x

            return x


class SelfAttentionWithPooling(tf.keras.layers.Layer):
    def __init__(self, use_bias=True, sn=False, name='SelfAttentionWithPooling'):
        super(SelfAttentionWithPooling, self).__init__(name=name)
        self.use_bias = use_bias
        self.sn = sn

    def build(self, input_shape):
        self.channels = input_shape[-1]
        self.gamma = tf.Variable(initial_value=tf.constant(0.0, shape=[1]), name='gamma')

        self.f_conv = Conv(self.channels // 8, kernel=1, stride=1, use_bias=self.use_bias, sn=self.sn, name='f_conv')
        self.g_conv = Conv(self.channels // 8, kernel=1, stride=1, use_bias=self.use_bias, sn=self.sn, name='g_conv')
        self.h_conv = Conv(self.channels // 2, kernel=1, stride=1, use_bias=self.use_bias, sn=self.sn, name='h_conv')

        self.attn_conv = Conv(self.channels, kernel=1, stride=1, use_bias=self.use_bias, sn=self.sn, name='attn_conv')

    def call(self, x, training=None, mask=None):
        with tf.name_scope(self.name):
            channels = x.shape[-1]

            f = self.f_conv(x)  # [bs, h, w, c']
            f = max_pooling(f)

            g = self.g_conv(x)  # [bs, h, w, c']

            h = self.h_conv(x)  # [bs, h, w, c]
            h = max_pooling(h)

            # N = h * w
            s = tf.matmul(hw_flatten(g), hw_flatten(f), transpose_b=True)  # [bs, N, N]

            beta = tf.nn.softmax(s)  # attention map

            o = tf.matmul(beta, hw_flatten(h))

            o = tf.reshape(o, shape=[x.shape[0], x.shape[1], x.shape[2], channels // 2])  # [bs, h, w, c]
            o = self.attn_conv(o)
            x = self.gamma * o + x

            return x


class SqueezeExcitation(tf.keras.layers.Layer):
    def __init__(self, ratio=16, use_bias=True, sn=False, name='SqueezeExcitation'):
        super(SqueezeExcitation, self).__init__(name=name)
        self.ratio = ratio
        self.use_bias = use_bias
        self.sn = sn

    def build(self, input_shape):
        self.channels = input_shape[-1]

        self.squeeze_fc = FullyConnected(units=self.channels // self.ratio, use_bias=self.use_bias, sn=self.sn,
                                         name='squeeze_fc')
        self.excitation_fc = FullyConnected(units=self.channels, use_bias=self.use_bias, sn=self.sn,
                                            name='excitation_fc')

    def call(self, x, training=None, mask=None):
        with tf.name_scope(self.name):
            squeeze = Global_Avg_Pooling(x)

            excitation = self.squeeze_fc(squeeze)
            excitation = Relu(excitation)
            excitation = self.excitation_fc(excitation)
            excitation = Sigmoid(excitation)

            excitation = tf.reshape(excitation, [-1, 1, 1, self.channels])

            scale = x * excitation

            return scale


class ConvBlockAttention(tf.keras.layers.Layer):
    def __init__(self, ratio=16, use_bias=True, sn=False, name='ConvBlockAttention'):
        super(ConvBlockAttention, self).__init__(name=name)
        self.ratio = ratio
        self.use_bias = use_bias
        self.sn = sn

    def build(self, input_shape):
        self.channels = input_shape[-1]

        self.fc_0 = FullyConnected(units=self.channels // self.ratio, use_bias=self.use_bias, sn=self.sn, name='fc_0')
        self.fc_1 = FullyConnected(units=self.channels, use_bias=self.use_bias, sn=self.sn, name='fc_1')

        self.scale_conv = Conv(channels=1, kernel=7, stride=1, pad=1, use_bias=False, sn=self.sn, name='scale_conv')

    def call(self, x, training=None, mask=None):
        with tf.name_scope(self.name):
            with tf.name_scope('channel_attention'):
                x_gap = Global_Avg_Pooling(x)
                x_gap = self.fc_0(x_gap)
                x_gap = Relu(x_gap)
                x_gap = self.fc_1(x_gap)

                x_gmp = Global_Avg_Pooling(x)
                x_gmp = self.fc_0(x_gmp)
                x_gmp = Relu(x_gmp)
                x_gmp = self.fc_1(x_gmp)

                scale = tf.reshape(x_gap + x_gmp, [-1, 1, 1, self.channels])
                scale = Sigmoid(scale)

                x = x * scale

            with tf.name_scope('spatial_attention'):
                x_channel_avg_pooling = tf.reduce_mean(x, axis=-1, keepdims=True)
                x_channel_max_pooling = tf.reduce_max(x, axis=-1, keepdims=True)
                scale = tf.concat([x_channel_avg_pooling, x_channel_max_pooling], axis=-1)

                scale = self.scale_conv(scale)
                scale = Sigmoid(scale)

                x = x * scale

            return x


class GlobalContextBlock(tf.keras.layers.Layer):
    def __init__(self, channels, use_bias=True, sn=False, name='GlobalContextBlock'):
        super(GlobalContextBlock, self).__init__(name=name)
        self.channels = channels
        self.use_bias = use_bias
        self.sn = sn

        self.mask_conv = Conv(channels=1, kernel=1, stride=1, use_bias=self.use_bias, sn=self.sn, name='mask_conv')

        self.transform_blocks = []

        blocks = []
        for i in range(2):
            blocks += [Conv(self.channels, kernel=1, stride=1, use_bias=self.use_bias, sn=self.sn, name='conv_0')]
            blocks += [LayerNorm(epsilon=1e-3, name='layer_norm_0')]
            blocks += [Relu()]

            blocks = Sequential(blocks)

            self.transform_blocks.append(blocks)

    def build(self, input_shape):
        self.bs, self.h, self.w, self.ch = input_shape

        self.transform_back_conv_0 = Conv(channels=self.ch, kernel=1, stride=1, use_bias=self.use_bias, sn=self.sn,
                                          name='back_conv_0')
        self.transform_back_conv_1 = Conv(channels=self.ch, kernel=1, stride=1, use_bias=self.use_bias, sn=self.sn,
                                          name='back_conv_1')

    def call(self, x, training=None, mask=None):
        with tf.name_scope(self.name):
            with tf.name_scope('context_modeling'):
                x_init = x
                x_init = hw_flatten(x_init)  # [N, H*W, C]
                x_init = tf.transpose(x_init, perm=[0, 2, 1])
                x_init = tf.expand_dims(x_init, axis=1)

                context_mask = self.mask_conv(x)
                context_mask = hw_flatten(context_mask)
                context_mask = tf.nn.softmax(context_mask, axis=1)  # [N, H*W, 1]
                context_mask = tf.transpose(context_mask, perm=[0, 2, 1])
                context_mask = tf.expand_dims(context_mask, axis=-1)

                context = tf.matmul(x_init, context_mask)
                context = tf.reshape(context, shape=[self.bs, 1, 1, self.ch])

            with tf.name_scope('transform_0'):
                context_transform = self.transform_blocks[0](context)
                context_transform = self.transform_back_conv_0(context_transform)
                context_transform = Sigmoid(context_transform)

                x = x * context_transform

            with tf.name_scope('transform_1'):
                context_transform = self.transform_blocks[1](context)
                context_transform = self.transform_back_conv_0(context_transform)
                context_transform = Sigmoid(context_transform)

                x = x + context_transform

            return x


class SrmBlock(tf.keras.layers.Layer):
    def __init__(self, use_bias=True, sn=False, name='SrmBlock'):
        super(SrmBlock, self).__init__(name=name)
        self.use_bias = use_bias
        self.sn = sn

        self.batch_norm = BatchNorm(momentum=0.9, epsilon=1e-5, name='batch_norm')

    def build(self, input_shape):
        self.bs, self.h, self.w, self.channels = input_shape
        self.conv_1d = tf.keras.layers.Conv1D(self.channels, kernel_size=2, strides=1, use_bias=self.use_bias,
                                              name='conv_1d')

    def call(self, x, training=None, mask=None):
        with tf.name_scope(self.name):
            x = tf.reshape(x, shape=[self.bs, -1, self.channels])  # [bs, h*w, c]

            x_mean, x_var = tf.nn.moments(x, axes=1, keepdims=True)  # [bs, 1, c]
            x_std = tf.sqrt(x_var + 1e-5)

            t = tf.concat([x_mean, x_std], axis=1)  # [bs, 2, c]

            z = self.conv_1d(t)
            z = self.batch_norm(z, training=training)

            g = Sigmoid(z)

            x = tf.reshape(x * g, shape=[self.bs, self.h, self.w, self.channels])

            return x


##################################################################################
# Normalization
##################################################################################

def BatchNorm(momentum=0.9, epsilon=1e-5, name='BatchNorm'):
    return tf.keras.layers.BatchNormalization(momentum=momentum, epsilon=epsilon,
                                              center=True, scale=True,
                                              name=name)

def InstanceNorm(epsilon=1e-5, name='InstanceNorm'):
    return tfa.layers.normalizations.InstanceNormalization(epsilon=epsilon, scale=True, center=True,
                                                           name=name)

def LayerNorm(epsilon=1e-3, name='LayerNorm'):
    return tf.keras.layers.LayerNormalization(epsilon=epsilon, center=True, scale=True,
                                              name=name)

def GroupNorm(groups=32, epsilon=1e-5, name='GroupNorm'):
    return tfa.layers.normalizations.GroupNormalization(groups=groups, epsilon=epsilon, scale=True, center=True,
                                                        name=name)


class AdaptiveInstanceNorm(tf.keras.layers.Layer):
    def __init__(self, gamma, beta, epsilon=1e-5, name='AdaptiveInstanceNorm'):
        super(AdaptiveInstanceNorm, self).__init__(name=name)
        self.gamma = gamma
        self.beta = beta
        self.epsilon = epsilon

    def call(self, content, training=None, mask=None):
        # gamma, beta = style_mean, style_std from MLP
        # See https://github.com/taki0112/MUNIT-Tensorflow

        c_mean, c_var = tf.nn.moments(content, axes=[1, 2], keepdims=True)
        c_std = tf.sqrt(c_var + self.epsilon)

        x = self.gamma * ((content - c_mean) / c_std) + self.beta

        return x


class AdaptiveLayerInstanceNorm(tf.keras.layers.Layer):
    def __init__(self, gamma, beta, epsilon=1e-5, smoothing=True, name='AdaptiveLayerInstanceNorm'):
        super(AdaptiveLayerInstanceNorm, self).__init__(name=name)
        self.gamma = gamma
        self.beta = beta
        self.epsilon = epsilon
        self.smoothing = smoothing

    def build(self, input_shape):
        self.ch = input_shape[-1]
        self.rho = tf.Variable(initial_value=tf.constant(1.0, shape=[self.ch]),
                               constraint=lambda x: tf.clip_by_value(x, clip_value_min=0.0, clip_value_max=1.0),
                               name='rho')
        if self.smoothing:
            self.rho = tf.clip_by_value(self.rho - tf.constant(0.1), 0.0, 1.0)

    def call(self, x, training=None, mask=None):
        ins_mean, ins_sigma = tf.nn.moments(x, axes=[1, 2], keepdims=True)
        x_ins = (x - ins_mean) / (tf.sqrt(ins_sigma + self.epsilon))

        ln_mean, ln_sigma = tf.nn.moments(x, axes=[1, 2, 3], keepdims=True)
        x_ln = (x - ln_mean) / (tf.sqrt(ln_sigma + self.epsilon))

        x_hat = self.rho * x_ins + (1 - self.rho) * x_ln
        x_hat = x_hat * self.gamma + self.beta

        return x_hat


class ConditionBatchNorm(tf.keras.layers.Layer):
    def __init__(self, momentum=0.9, epsilon=1e-5, name='ConditionBatchNorm'):
        super(ConditionBatchNorm, self).__init__(name=name)
        self.momentum = momentum
        self.epsilon = epsilon

    def build(self, input_shape):
        self.channels = input_shape[-1]
        self.test_mean = tf.Variable(initial_value=tf.constant(0.0, shape=[self.channels]), trainable=False,
                                     name='pop_mean')
        self.test_var = tf.Variable(initial_value=tf.constant(1.0, shape=[self.channels]), trainable=False,
                                    name='pop_var')

        self.beta_fc = FullyConnected(units=self.channels, name='beta_fc')
        self.gamma_fc = FullyConnected(units=self.channels, name='gamma_fc')

    def call(self, inputs, training=None, mask=None):
        x, z = inputs[0], inputs[1]

        beta = self.beta_fc(z)
        gamma = self.gamma_fc(z)

        beta = tf.reshape(beta, shape=[-1, 1, 1, self.channels])
        gamma = tf.reshape(gamma, shape=[-1, 1, 1, self.channels])

        if training:
            batch_mean, batch_var = tf.nn.moments(x, [0, 1, 2])
            ema_mean = self.test_mean.assign(self.test_mean * self.momentum + batch_mean * (1 - self.momentum))
            ema_var = self.test_var.assign(self.test_var * self.momentum + batch_var * (1 - self.momentum))

            with tf.control_dependencies([ema_mean, ema_var]):
                return tf.nn.batch_normalization(x, batch_mean, batch_var, beta, gamma, self.epsilon, name=self.name)
        else:
            return tf.nn.batch_normalization(x, self.test_mean, self.test_var, beta, gamma, self.epsilon,
                                             name=self.name)


class BatchInstanceNorm(tf.keras.layers.Layer):
    def __init__(self, epsilon=1e-5, name='BatchInstanceNorm'):
        super(BatchInstanceNorm, self).__init__(name=name)
        self.epsilon = epsilon

    def build(self, input_shape):
        self.ch = input_shape[-1]

        self.rho = tf.Variable(initial_value=tf.constant(1.0, shape=[self.ch]),
                               constraint=lambda x: tf.clip_by_value(x, clip_value_min=0.0, clip_value_max=1.0),
                               name='rho')
        self.gamma = tf.Variable(initial_value=tf.constant(1.0, shape=[self.ch]), name='gamma')
        self.beta = tf.Variable(initial_value=tf.constant(1.0, shape=[self.ch]), name='beta')

    def call(self, x, training=None, mask=None):
        batch_mean, batch_sigma = tf.nn.moments(x, axes=[0, 1, 2], keepdims=True)
        x_batch = (x - batch_mean) / (tf.sqrt(batch_sigma + self.epsilon))

        ins_mean, ins_sigma = tf.nn.moments(x, axes=[1, 2], keepdims=True)
        x_ins = (x - ins_mean) / (tf.sqrt(ins_sigma + self.epsilon))

        x_hat = self.rho * x_batch + (1 - self.rho) * x_ins
        x_hat = x_hat * self.gamma + self.beta

        return x_hat


class LayerInstaceNorm(tf.keras.layers.Layer):
    def __init__(self, epsilon=1e-5, name='LayerInstaceNorm'):
        super(LayerInstaceNorm, self).__init__(name=name)
        self.epsilon = epsilon

    def build(self, input_shape):
        self.ch = input_shape[-1]

        self.rho = tf.Variable(initial_value=tf.constant(0.0, shape=[self.ch]),
                               constraint=lambda x: tf.clip_by_value(x, clip_value_min=0.0, clip_value_max=1.0),
                               name='rho')
        self.gamma = tf.Variable(initial_value=tf.constant(1.0, shape=[self.ch]), name='gamma')
        self.beta = tf.Variable(initial_value=tf.constant(1.0, shape=[self.ch]), name='beta')

    def call(self, x, training=None, mask=None):
        ins_mean, ins_sigma = tf.nn.moments(x, axes=[1, 2], keepdims=True)
        x_ins = (x - ins_mean) / (tf.sqrt(ins_sigma + self.epsilon))

        ln_mean, ln_sigma = tf.nn.moments(x, axes=[1, 2, 3], keepdims=True)
        x_ln = (x - ln_mean) / (tf.sqrt(ln_sigma + self.epsilon))

        x_hat = self.rho * x_ins + (1 - self.rho) * x_ln

        x_hat = x_hat * self.gamma + self.beta

        return x_hat


class SwitchNorm(tf.keras.layers.Layer):
    def __init__(self, epsilon=1e-5, name='SwitchNorm'):
        super(SwitchNorm, self).__init__(name=name)
        self.epsilon = epsilon

    def build(self, input_shape):
        self.ch = input_shape[-1]

        self.gamma = tf.Variable(initial_value=tf.constant(1.0, shape=[self.ch]), name='gamma')
        self.beta = tf.Variable(initial_value=tf.constant(1.0, shape=[self.ch]), name='beta')

        self.mean_weight = tf.nn.softmax(tf.Variable(initial_value=tf.constant(1.0, shape=[3]), name='mean_weight'))
        self.var_weight = tf.nn.softmax(tf.Variable(initial_value=tf.constant(1.0, shape=[3]), name='var_weight'))

    def call(self, x, training=None, mask=None):
        batch_mean, batch_var = tf.nn.moments(x, [0, 1, 2], keepdims=True)
        ins_mean, ins_var = tf.nn.moments(x, [1, 2], keepdims=True)
        layer_mean, layer_var = tf.nn.moments(x, [1, 2, 3], keepdims=True)

        mean = self.mean_weight[0] * batch_mean + self.mean_weight[1] * ins_mean + self.mean_weight[2] * layer_mean
        var = self.var_wegiht[0] * batch_var + self.var_wegiht[1] * ins_var + self.var_wegiht[2] * layer_var

        x = (x - mean) / (tf.sqrt(var + self.epsilon))
        x = x * self.gamma + self.beta

        return x


def pixel_norm(x, epsilon=1e-8):
    return x * tf.math.rsqrt(tf.reduce_mean(tf.square(x), axis=-1, keepdims=True) + epsilon)


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


def Sigmoid(x=None, name='sigmoid'):
    if x is None:
        return tf.keras.layers.Activation(tf.keras.activations.sigmoid, name=name)
    else:
        return tf.keras.layers.Activation(tf.keras.activations.sigmoid, name=name)(x)


class Swish(tf.keras.layers.Layer):
    def __init__(self, name='Swish'):
        super(Swish, self).__init__(name=name)

    def call(self, x, training=None, mask=None):
        return x * Sigmoid(x, name=self.name)


##################################################################################
# Pooling & Resize
##################################################################################

def nearest_up_sample(x, scale_factor=2):
    _, h, w, _ = x.get_shape().as_list()
    new_size = [h * scale_factor, w * scale_factor]
    return tf.image.resize(x, size=new_size, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)


def bilinear_up_sample(x, scale_factor=2):
    _, h, w, _ = x.get_shape().as_list()
    new_size = [h * scale_factor, w * scale_factor]
    return tf.image.resize(x, size=new_size, method=tf.image.ResizeMethod.BILINEAR)


def nearest_down_sample(x, scale_factor=2):
    _, h, w, _ = x.get_shape().as_list()
    new_size = [h // scale_factor, w // scale_factor]
    return tf.image.resize(x, size=new_size, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)


def bilinear_down_sample(x, scale_factor=2):
    _, h, w, _ = x.get_shape().as_list()
    new_size = [h // scale_factor, w // scale_factor]
    return tf.image.resize(x, size=new_size, method=tf.image.ResizeMethod.BILINEAR)


def Global_Avg_Pooling(x=None, name='global_avg_pool'):
    if x is None:
        gap = tf.keras.layers.GlobalAveragePooling2D(name=name)
    else:
        gap = tf.keras.layers.GlobalAveragePooling2D(name=name)(x)
    return gap


def Global_Max_Pooling(x=None, name='global_max_pool'):
    if x is None:
        gmp = tf.keras.layers.GlobalMaxPool2D(name=name)
    else:
        gmp = tf.keras.layers.GlobalMaxPool2D(name=name)(x)
    return gmp


def max_pooling(x, pool_size=2, name='max_pool'):
    x = tf.keras.layers.MaxPool2D(pool_size=pool_size, strides=pool_size, padding='SAME', name=name)(x)
    return x


def avg_pooling(x, pool_size=2, name='avg_pool'):
    x = tf.keras.layers.AvgPool2D(pool_size=pool_size, strides=pool_size, padding='SAME', name=name)(x)
    return x


def Flatten(x=None, name='flatten'):

    if x is None:
        return tf.keras.layers.Flatten(name=name)
    else :
        return tf.keras.layers.Flatten(name=name)(x)


def hw_flatten(x):
    return tf.reshape(x, shape=[x.shape[0], -1, x.shape[-1]])


##################################################################################
# Loss Function
##################################################################################

def classification_loss(logit, label):
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=label, logits=logit))
    prediction = tf.equal(tf.argmax(logit, -1), tf.argmax(label, -1))
    accuracy = tf.reduce_mean(tf.cast(prediction, tf.float32))

    return loss, accuracy


def L1_loss(x, y):
    loss = tf.reduce_mean(tf.abs(x - y))

    return loss


def L2_loss(x, y):
    loss = tf.reduce_mean(tf.square(x - y))

    return loss


def huber_loss(x, y):
    loss = tf.compat.v1.losses.huber_loss(x, y)

    return loss


def regularization_loss(model):
    loss = tf.nn.scale_regularization_loss(model.losses)

    return loss


def histogram_loss(x, y):
    histogram_x = get_histogram(x)
    histogram_y = get_histogram(y)

    hist_loss = L1_loss(histogram_x, histogram_y)

    return hist_loss


def get_histogram(img, bin_size=0.2):
    hist_entries = []

    img_r, img_g, img_b = tf.split(img, num_or_size_splits=3, axis=-1)

    for img_chan in [img_r, img_g, img_b]:
        for i in np.arange(-1, 1, bin_size):
            gt = tf.greater(img_chan, i)
            leq = tf.less_equal(img_chan, i + bin_size)
            condition = tf.cast(tf.logical_and(gt, leq), tf.float32)
            hist_entries.append(tf.reduce_sum(condition))

    hist = normalization(hist_entries)

    return hist


def normalization(x):
    x = (x - tf.reduce_min(x)) / (tf.reduce_max(x) - tf.reduce_min(x))
    return x


def gram_matrix(x):
    b, h, w, c = x.get_shape().as_list()

    x = tf.reshape(x, shape=[b, -1, c])

    x = tf.matmul(tf.transpose(x, perm=[0, 2, 1]), x)
    x = x / (h * w * c)

    return x

def gram_matrix_2(x):
  result = tf.linalg.einsum('bijc,bijd->bcd', x, x)
  input_shape = tf.shape(x)
  num_locations = tf.cast(input_shape[1]*input_shape[2], tf.float32)

  return result/(num_locations)

def gram_style_loss(x, y):
    _, height, width, channels = x.get_shape().as_list()

    x = gram_matrix(x)
    y = gram_matrix(y)

    loss = L2_loss(x, y)  # simple version

    # Original eqn as a constant to divide i.e 1/(4. * (channels ** 2) * (width * height) ** 2)
    # loss = tf.reduce_mean(tf.square(x - y)) / (channels ** 2 * width * height)  # (4.0 * (channels ** 2) * (width * height) ** 2)

    return loss


def color_consistency_loss(x, y):
    x_mu, x_var = tf.nn.moments(x, axes=[1, 2], keepdims=True)
    y_mu, y_var = tf.nn.moments(y, axes=[1, 2], keepdims=True)

    loss = L2_loss(x_mu, y_mu) + 5.0 * L2_loss(x_var, y_var)

    return loss


def dice_loss(n_classes, logits, labels):
    """
    :param n_classes: number of classes
    :param logits: [batch_size, m, n, n_classes] float32, output logits
    :param labels: [batch_size, m, n, 1] int32, class label
    :return:
    """

    # https://github.com/keras-team/keras/issues/9395

    smooth = 1e-7
    dtype = tf.float32

    # alpha=beta=0.5 : dice coefficient
    # alpha=beta=1   : tanimoto coefficient (also known as jaccard)
    # alpha+beta=1   : produces set of F*-scores
    alpha, beta = 0.5, 0.5

    # make onehot label [batch_size, m, n, n_classes]
    # tf.one_hot() will ignore (creates zero vector) labels larger than n_class and less then 0
    onehot_labels = tf.one_hot(tf.squeeze(labels, axis=-1), depth=n_classes, dtype=dtype)

    ones = tf.ones_like(onehot_labels, dtype=dtype)
    predicted = tf.nn.softmax(logits)
    p0 = predicted
    p1 = ones - predicted
    g0 = onehot_labels
    g1 = ones - onehot_labels

    num = tf.reduce_sum(p0 * g0, axis=[0, 1, 2])
    den = num + alpha * tf.reduce_sum(p0 * g1, axis=[0, 1, 2]) + beta * tf.reduce_sum(p1 * g0, axis=[0, 1, 2])

    loss = tf.cast(n_classes, dtype=dtype) - tf.reduce_sum((num + smooth) / (den + smooth))
    return loss


##################################################################################
# GAN Loss Function
##################################################################################
def discriminator_loss(Ra, gan_type, real_logit, fake_logit):
    # Ra = Relativistic
    real_loss = 0
    fake_loss = 0

    if Ra and (gan_type.__contains__('wgan') or gan_type == 'sphere'):
        print("No exist [Ra + WGAN or Ra + Sphere], so use the {} loss function".format(gan_type))
        Ra = False

    if Ra:
        Ra_real_logit = (real_logit - tf.reduce_mean(fake_logit))
        Ra_fake_logit = (fake_logit - tf.reduce_mean(real_logit))

        if gan_type == 'lsgan':
            real_loss = tf.reduce_mean(tf.square(Ra_real_logit - 1.0))
            fake_loss = tf.reduce_mean(tf.square(Ra_fake_logit + 1.0))

        if gan_type == 'gan' or gan_type == 'gan-gp' or gan_type == 'dragan':
            real_loss = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(Ra_real_logit), logits=Ra_real_logit))
            fake_loss = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(Ra_fake_logit), logits=Ra_fake_logit))

        if gan_type == 'hinge':
            real_loss = tf.reduce_mean(Relu(1.0 - Ra_real_logit))
            fake_loss = tf.reduce_mean(Relu(1.0 + Ra_fake_logit))

        if gan_type == 'realness':
            fake_logit = tf.exp(tf.nn.log_softmax(fake_logit, axis=-1))
            real_logit = tf.exp(tf.nn.log_softmax(real_logit, axis=-1))

            num_outcomes = real_logit.shape[-1]

            gauss = np.random.normal(0, 0.1, 1000)
            count, bins = np.histogram(gauss, num_outcomes)
            anchor0 = count / sum(count)  # anchor_fake

            unif = np.random.uniform(-1, 1, 1000)
            count, bins = np.histogram(unif, num_outcomes)
            anchor1 = count / sum(count)  # anchor_real

            anchor_real = tf.zeros([real_logit.shape[0], num_outcomes]) + tf.cast(anchor1, tf.float32)
            anchor_fake = tf.zeros([real_logit.shape[0], num_outcomes]) + tf.cast(anchor0, tf.float32)

            real_loss = realness_loss(anchor_real, real_logit, skewness=10.0)
            fake_loss = realness_loss(anchor_fake, fake_logit, skewness=-10.0)

    else:
        if gan_type.__contains__('wgan'):
            real_loss = -tf.reduce_mean(real_logit)
            fake_loss = tf.reduce_mean(fake_logit)

        if gan_type == 'lsgan':
            real_loss = tf.reduce_mean(tf.square(real_logit - 1.0))
            fake_loss = tf.reduce_mean(tf.square(fake_logit))

        if gan_type == 'gan' or gan_type == 'gan-gp' or gan_type == 'dragan':
            real_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(real_logit), logits=real_logit))
            fake_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(fake_logit), logits=fake_logit))

        if gan_type == 'hinge':
            real_loss = tf.reduce_mean(Relu(1.0 - real_logit))
            fake_loss = tf.reduce_mean(Relu(1.0 + fake_logit))

        if gan_type == 'sphere':
            bs, c = real_logit.get_shape().as_list()
            moment = 3
            north_pole = tf.one_hot(tf.tile([c], multiples=[bs]), depth=c + 1)  # [bs, c+1] -> [0, 0, 0, ... , 1]

            real_projection = inverse_stereographic_projection(real_logit)
            fake_projection = inverse_stereographic_projection(fake_logit)

            for i in range(1, moment + 1):
                real_loss += -tf.reduce_mean(tf.pow(sphere_loss(real_projection, north_pole), i))
                fake_loss += tf.reduce_mean(tf.pow(sphere_loss(fake_projection, north_pole), i))

        if gan_type == 'realness':
            fake_logit = tf.exp(tf.nn.log_softmax(fake_logit, axis=-1))
            real_logit = tf.exp(tf.nn.log_softmax(real_logit, axis=-1))

            num_outcomes = real_logit.shape[-1]

            gauss = np.random.normal(0, 0.1, 1000)
            count, bins = np.histogram(gauss, num_outcomes)
            anchor0 = count / sum(count)  # anchor_fake

            unif = np.random.uniform(-1, 1, 1000)
            count, bins = np.histogram(unif, num_outcomes)
            anchor1 = count / sum(count)  # anchor_real

            anchor_real = tf.zeros([real_logit.shape[0], num_outcomes]) + tf.cast(anchor1, tf.float32)
            anchor_fake = tf.zeros([real_logit.shape[0], num_outcomes]) + tf.cast(anchor0, tf.float32)

            real_loss = realness_loss(anchor_real, real_logit, skewness=10.0)
            fake_loss = realness_loss(anchor_fake, fake_logit, skewness=-10.0)

    loss = real_loss + fake_loss

    return loss


def generator_loss(Ra, gan_type, real_logit, fake_logit):
    # Ra = Relativistic
    fake_loss = 0
    real_loss = 0

    if Ra and (gan_type.__contains__('wgan') or gan_type == 'sphere'):
        print("No exist [Ra + WGAN or Ra + Sphere], so use the {} loss function".format(gan_type))
        Ra = False

    if Ra:
        Ra_fake_logit = (fake_logit - tf.reduce_mean(real_logit))
        Ra_real_logit = (real_logit - tf.reduce_mean(fake_logit))

        if gan_type == 'lsgan':
            fake_loss = tf.reduce_mean(tf.square(Ra_fake_logit - 1.0))
            real_loss = tf.reduce_mean(tf.square(Ra_real_logit + 1.0))

        if gan_type == 'gan' or gan_type == 'gan-gp' or gan_type == 'dragan':
            fake_loss = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(Ra_fake_logit), logits=Ra_fake_logit))
            real_loss = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(Ra_real_logit), logits=Ra_real_logit))

        if gan_type == 'hinge':
            fake_loss = tf.reduce_mean(Relu(1.0 - Ra_fake_logit))
            real_loss = tf.reduce_mean(Relu(1.0 + Ra_real_logit))

        if gan_type == 'realness':
            fake_logit = tf.exp(tf.nn.log_softmax(fake_logit, axis=-1))
            real_logit = tf.exp(tf.nn.log_softmax(real_logit, axis=-1))

            fake_loss = realness_loss(real_logit, fake_logit)

    else:
        if gan_type.__contains__('wgan'):
            fake_loss = -tf.reduce_mean(fake_logit)

        if gan_type == 'lsgan':
            fake_loss = tf.reduce_mean(tf.square(fake_logit - 1.0))

        if gan_type == 'gan' or gan_type == 'gan-gp' or gan_type == 'dragan':
            fake_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(fake_logit), logits=fake_logit))

        if gan_type == 'hinge':
            fake_loss = -tf.reduce_mean(fake_logit)

        if gan_type == 'sphere':
            bs, c = real_logit.get_shape().as_list()
            moment = 3
            north_pole = tf.one_hot(tf.tile([c], multiples=[bs]), depth=c + 1)  # [bs, c+1] -> [0, 0, 0, ... , 1]

            fake_projection = inverse_stereographic_projection(fake_logit)

            for i in range(1, moment + 1):
                fake_loss += -tf.reduce_mean(tf.pow(sphere_loss(fake_projection, north_pole), i))

        if gan_type == 'realness':
            num_outcomes = real_logit.shape[-1]
            unif = np.random.uniform(-1, 1, 1000)
            count, bins = np.histogram(unif, num_outcomes)
            anchor1 = count / sum(count)  # anchor_real
            anchor_real = tf.zeros([real_logit.shape[0], num_outcomes]) + tf.cast(anchor1, tf.float32)

            fake_logit = tf.exp(tf.nn.log_softmax(fake_logit, axis=-1))
            fake_loss = realness_loss(anchor_real, fake_logit, skewness=10.0)

    loss = fake_loss + real_loss

    return loss


def vdb_loss(mu, logvar, i_c=0.1):
    # variational discriminator bottleneck loss
    kl_divergence = 0.5 * tf.reduce_sum(tf.square(mu) + tf.exp(logvar) - 1 - logvar, axis=-1)

    loss = tf.reduce_mean(kl_divergence - i_c)

    return loss

def gradient_penalty(discriminator, real_images, fake_images, gan_type='wgan-gp', gamma=10.0):
    if gan_type.__contains__('dragan'):
        eps = tf.random.uniform(shape=real_images.shape, minval=0.0, maxval=1.0)
        _, x_var = tf.nn.moments(real_images, axes=[0, 1, 2, 3])
        x_std = tf.sqrt(x_var)

        fake_images = real_images + 0.5 * x_std * eps

    alpha = tf.random.uniform([real_images.shape[0], 1, 1, 1], minval=0.0, maxval=1.0)

    interpolated = real_images + alpha * (fake_images - real_images)

    with tf.GradientTape() as tape:
        tape.watch(interpolated)
        inter_logit = discriminator(interpolated)

    grad = tape.gradient(inter_logit, interpolated)
    grad_norm = tf.norm(Flatten(grad), axis=-1)

    if gan_type == 'wgan-lp':
        gp = tf.reduce_mean(tf.square(tf.maximum(0.0, grad_norm - 1.0))) * gamma

    else :
        gp = tf.reduce_mean(tf.square(grad_norm - 1.0)) * gamma

    return gp

def simple_gp(discriminator, real_images, fake_images, r1_gamma=10, r2_gamma=0):
    # Used in StyleGAN

    r1_penalty = 0
    r2_penalty = 0

    if r1_gamma != 0:
        with tf.GradientTape() as p_tape:
            p_tape.watch(real_images)
            real_loss = tf.reduce_sum(discriminator(real_images))  # In some cases, you may use reduce_mean

        real_grads = p_tape.gradient(real_loss, real_images)
        r1_penalty = 0.5 * r1_gamma * tf.reduce_mean(tf.reduce_sum(tf.square(real_grads), axis=[1, 2, 3]))

    if r2_gamma != 0:
        with tf.GradientTape() as p_tape:
            p_tape.watch(fake_images)
            fake_loss = tf.reduce_sum(discriminator(fake_images))  # In some cases, you may use reduce_mean

        fake_grads = p_tape.gradient(fake_loss, fake_images)
        r2_penalty = 0.5 * r2_gamma * tf.reduce_mean(tf.reduce_sum(tf.square(fake_grads), axis=[1, 2, 3]))

    return r1_penalty + r2_penalty


def inverse_stereographic_projection(x):
    x_u = tf.transpose(2 * x) / (tf.pow(tf.norm(x, axis=-1), 2) + 1.0)
    x_v = (tf.pow(tf.norm(x, axis=-1), 2) - 1.0) / (tf.pow(tf.norm(x, axis=-1), 2) + 1.0)

    x_projection = tf.transpose(tf.concat([x_u, [x_v]], axis=0))

    return x_projection


def sphere_loss(x, y):
    loss = tf.math.acos(tf.matmul(x, tf.transpose(y)))

    return loss

def realness_loss(anchor, feature, skewness=0.0, positive_skew=10.0, negative_skew=-10.0):
    """
    num_outcomes = anchor.shape[-1]
    positive_skew = 10.0
    negative_skew = -10.0
    # [num_outcomes, positive_skew, negative_skew]
    # [51, 10.0, -10.0]
    # [21, 1.0, -1.0]

    gauss = np.random.normal(0, 0.1, 1000)
    count, bins = np.histogram(gauss, num_outcomes)
    anchor0 = count / sum(count) # anchor_fake

    unif = np.random.uniform(-1, 1, 1000)
    count, bins = np.histogram(unif, num_outcomes)
    anchor1 = count / sum(count) # anchor_real
    """

    batch_size = feature.shape[0]
    num_outcomes = feature.shape[-1]

    supports = tf.linspace(start=negative_skew, stop=positive_skew, num=num_outcomes)
    delta = (positive_skew - negative_skew) / (num_outcomes - 1)

    skew = tf.fill(dims=[batch_size, num_outcomes], value=skewness)

    # experiment to adjust KL divergence between positive/negative anchors
    Tz = skew + tf.reshape(supports, shape=[1, -1]) * tf.ones(shape=[batch_size, 1])
    Tz = tf.clip_by_value(Tz, negative_skew, positive_skew)

    b = (Tz - negative_skew) / delta
    lower_b = tf.cast(tf.math.floor(b), tf.int32).numpy()
    upper_b = tf.cast(tf.math.ceil(b), tf.int32).numpy()

    lower_b[(upper_b > 0) * (lower_b == upper_b)] -= 1
    upper_b[(lower_b < (num_outcomes - 1)) * (lower_b == upper_b)] += 1

    offset = tf.expand_dims(tf.linspace(start=0.0, stop=(batch_size - 1) * num_outcomes, num=batch_size), axis=1)
    offset = tf.tile(offset, multiples=[1, num_outcomes])

    skewed_anchor = tf.reshape(tf.zeros(shape=[batch_size, num_outcomes]), shape=[-1]).numpy()
    lower_idx = tf.cast(tf.reshape(lower_b + offset, shape=[-1]), tf.int32).numpy()
    lower_updates = tf.reshape(anchor * (tf.cast(upper_b, tf.float32) - b), shape=[-1]).numpy()
    skewed_anchor[lower_idx] += lower_updates

    upper_idx = tf.cast(tf.reshape(upper_b + offset, shape=[-1]), tf.int32).numpy()
    upper_updates = tf.reshape(anchor * (b - tf.cast(lower_b, tf.float32)), shape=[-1])
    skewed_anchor[upper_idx] += upper_updates

    loss = -(skewed_anchor * tf.reduce_mean(tf.reduce_sum(tf.math.log((feature + 1e-16)), axis=-1)))

    return loss

##################################################################################
# KL-Divergence Loss Function
##################################################################################

# typical version
def z_sample(mean, logvar):
    eps = tf.random.normal(tf.shape(mean), mean=0.0, stddev=1.0, dtype=tf.float32)

    return mean + tf.exp(logvar * 0.5) * eps


def kl_loss(mean, logvar):
    # shape : [batch_size, channel]
    loss = 0.5 * tf.reduce_sum(tf.square(mean) + tf.exp(logvar) - 1 - logvar, axis=-1)
    loss = tf.reduce_mean(loss)

    return loss


# version 2
def z_sample_2(mean, sigma):
    eps = tf.random.normal(tf.shape(mean), mean=0.0, stddev=1.0, dtype=tf.float32)

    return mean + sigma * eps


def kl_loss_2(mean, sigma):
    # shape : [batch_size, channel]
    loss = 0.5 * tf.reduce_sum(tf.square(mean) + tf.square(sigma) - tf.math.log(1e-8 + tf.square(sigma)) - 1, axis=-1)
    loss = tf.reduce_mean(loss)

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


##################################################################################
# Natural Language Processing
##################################################################################

class VariousRNN(tf.keras.layers.Layer):
    def __init__(self, n_hidden=128, n_layer=1, dropout_rate=0.5, bidirectional=True, return_state=True,
                 rnn_type='lstm', name='VariousRNN'):
        super(VariousRNN, self).__init__(name=name)
        self.n_hidden = n_hidden
        self.n_layer = n_layer
        self.dropout_rate = dropout_rate
        self.bidirectional = bidirectional
        self.return_state = return_state
        self.rnn_type = rnn_type.lower()

        if self.rnn_type == 'lstm':
            self.cell_type = tf.keras.layers.LSTMCell
        elif self.rnn_type == 'gru':
            self.cell_type = tf.keras.layers.GRUCell
        else:
            raise NotImplementedError

        self.rnn = tf.keras.layers.RNN(
            [self.cell_type(units=n_hidden, dropout=self.dropout_rate) for _ in range(self.n_layer)],
            return_sequences=True, return_state=self.return_state)
        if self.bidirectional:
            self.rnn = tf.keras.layers.Bidirectional(self.rnn)
        """
        if also return_state=True, 
        whole_sequence, forward_hidden, forward_cell, backward_hidden, backward_cell (LSTM)
        whole_sequence, forward_hidden, forward_cell (GRU)
        sent_emb = tf.concat([forward_hidden, backward_hidden], axis=-1)
        """

    def call(self, x, training=None, mask=None):
        if self.return_state:
            if self.bidirectional:
                if self.rnn_type == 'gru':
                    output, forward_h, backward_h = self.rnn(x, training=training)
                else:  # LSTM
                    output, forward_state, backward_state = self.rnn(x, training=training)
                    forward_h, backward_h = forward_state[0], backward_state[0]
                    forward_c, backward_c = forward_state[1], backward_state[1]

                sent_emb = tf.concat([forward_h, backward_h], axis=-1)
            else:
                if self.rnn_type == 'gru':
                    output, forward_h = self.rnn(x, training=training)
                else:
                    output, forward_state = self.rnn(x, training=training)
                    forward_h, forward_c = forward_state

                sent_emb = forward_h

        else:
            output = self.rnn(x, training=training)
            sent_emb = output[:, -1, :]

        word_emb = output

        return word_emb, sent_emb

def embed_sequence(n_words, embed_dim, init_range=0.1, trainable=True, name='embed_layer'):
    emeddings = tf.keras.layers.Embedding(input_dim=n_words, output_dim=embed_dim,
                                          embeddings_initializer=tf.random_uniform_initializer(minval=-init_range,
                                                                                               maxval=init_range),
                                          trainable=trainable, name=name)
    return emeddings