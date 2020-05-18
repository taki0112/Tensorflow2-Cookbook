import tensorflow as tf
import numpy as np
import tensorflow_addons as tfa
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
def conv(x, channels, kernel=4, stride=2, pad=0, pad_type='zero', use_bias=True, sn=False, name='conv'):
    if pad > 0:
        h = x.get_shape().as_list()[1]
        if h % stride == 0:
            pad = pad * 2
        else:
            pad = max(kernel - (h % stride), 0)

        pad_top = pad // 2
        pad_bottom = pad - pad_top
        pad_left = pad // 2
        pad_right = pad - pad_left

        if pad_type == 'zero':
            x = tf.pad(x, [[0, 0], [pad_top, pad_bottom], [pad_left, pad_right], [0, 0]])
        if pad_type == 'reflect':
            x = tf.pad(x, [[0, 0], [pad_top, pad_bottom], [pad_left, pad_right], [0, 0]], mode='REFLECT')

    if sn:
        x = SpectralNormalization(tf.keras.layers.Conv2D(filters=channels, kernel_size=kernel,
                                                         kernel_initializer=weight_initializer, kernel_regularizer=weight_regularizer,
                                                         strides=stride, use_bias=use_bias), name='sn_' + name)(x)

    else:
        x = tf.keras.layers.Conv2D(filters=channels, kernel_size=kernel,
                                   kernel_initializer=weight_initializer, kernel_regularizer=weight_regularizer,
                                   strides=stride, use_bias=use_bias, name=name)(x)

    return x


def partial_conv(x, channels, kernel=4, stride=2, use_bias=True, padding='SAME', sn=False, name='conv_0'):
    with tf.name_scope(name) :
        _, h, w, _ = x.get_shape().as_list()

        slide_window = kernel * kernel
        mask = tf.ones(shape=[1, h, w, 1])

        update_mask = tf.keras.layers.Conv2D(filters=1,
                                             kernel_size=kernel, kernel_initializer=tf.constant_initializer(1.0),
                                             strides=stride, padding=padding, use_bias=False, trainable=False,
                                             name='mask_conv')(mask)
        mask_ratio = slide_window / (update_mask + 1e-8)
        update_mask = tf.clip_by_value(update_mask, 0.0, 1.0)
        mask_ratio = mask_ratio * update_mask

        if sn :
            x = SpectralNormalization(tf.keras.layers.Conv2D(channels, kernel_size=kernel,
                                                             kernel_initializer=weight_initializer, kernel_regularizer=weight_regularizer,
                                                             strides=stride, padding=padding, use_bias=False), name='sn_conv_x')(x)
        else :
            x = tf.keras.layers.Conv2D(channels, kernel_size=kernel,
                                       kernel_initializer=weight_initializer, kernel_regularizer=weight_regularizer,
                                       strides=stride, padding=padding, use_bias=False, name='conv_x')(x)
        x = x * mask_ratio

        if use_bias:
            bias = tf.Variable(initial_value=tf.constant(0.0, shape=[channels]), name='bias')
            x = tf.nn.bias_add(x, bias)
            x = x * update_mask

        return x


def dilate_conv(x, channels, kernel=4, rate=2, use_bias=True, padding='SAME', sn=False, name='dilate_conv'):
    if sn :
        x = SpectralNormalization(tf.keras.layers.Conv2D(channels, kernel_size=kernel,
                                                         kernel_initializer=weight_initializer,
                                                         kernel_regularizer=weight_regularizer,
                                                         strides=1, padding=padding, use_bias=use_bias,
                                                         dilation_rate=rate), name='sn_' + name)(x)
    else :
        x = tf.keras.layers.Conv2D(channels, kernel_size=kernel,
                                                         kernel_initializer=weight_initializer, kernel_regularizer=weight_regularizer,
                                                         strides=1, padding=padding, use_bias=use_bias,
                                                         dilation_rate=rate, name=name)(x)

    return x



def deconv(x, channels, kernel=4, stride=2, padding='SAME', use_bias=True, sn=False, name='deconv'):
    if sn :
        x = SpectralNormalization(tf.keras.layers.Conv2DTranspose(filters=channels, kernel_size=kernel,
                                                                  kernel_initializer=weight_initializer, kernel_regularizer=weight_regularizer,
                                                                  strides=stride, use_bias=use_bias,
                                                                  padding=padding), name='sn_' + name)(x)
    else :
        x = tf.keras.layers.Conv2DTranspose(filters=channels, kernel_size=kernel, kernel_initializer=weight_initializer,
                                            kernel_regularizer=weight_regularizer,
                                            strides=stride, use_bias=use_bias,
                                            padding=padding, name=name)(x)

    return x


def conv_pixel_shuffle_up(x, scale_factor=2, use_bias=True, sn=False, name='pixel_shuffle'):
    channel = x.get_shape()[-1] * (scale_factor ** 2)
    x = conv(x, channel, kernel=1, stride=1, use_bias=use_bias, sn=sn, name=name)
    x = tf.nn.depth_to_space(x, block_size=scale_factor)

    return x


def conv_pixel_shuffle_down(x, scale_factor=2, use_bias=True, sn=False, name='pixel_shuffle'):
    channel = x.get_shape()[-1] // (scale_factor ** 2)
    assert channel > 0
    x = conv(x, channel, kernel=1, stride=1, use_bias=use_bias, sn=sn, name=name)
    x = tf.nn.space_to_depth(x, block_size=scale_factor)

    return x


def fully_connected(x, units, use_bias=True, sn=False, name='linear'):
    x = flatten(x)
    if sn :
        x = SpectralNormalization(tf.keras.layers.Dense(units,
                                                        kernel_initializer=weight_initializer, kernel_regularizer=weight_regularizer_fully,
                                                        use_bias=use_bias), name='sn_' + name)(x)
    else :
        x = tf.keras.layers.Dense(units,
                                  kernel_initializer=weight_initializer, kernel_regularizer=weight_regularizer_fully,
                                  use_bias=use_bias, name=name)(x)

    return x


##################################################################################
# Blocks
##################################################################################

def resblock(x_init, channels, use_bias=True, training=True, sn=False, name='resblock'):
    with tf.name_scope(name) :
        with tf.name_scope('res1'):
            x = conv(x_init, channels, kernel=3, stride=1, pad=1, pad_type='reflect', use_bias=use_bias, sn=sn, name='conv_0')
            x = batch_norm(x, training, name='batch_norm_0')
            x = relu(x)

        with tf.name_scope('res2'):
            x = conv(x, channels, kernel=3, stride=1, pad=1, pad_type='reflect', use_bias=use_bias, sn=sn, name='conv_1')
            x = batch_norm(x, training, name='batch_norm_2')

        if channels != x_init.shape[-1] :
            x_init = conv(x_init, channels, kernel=1, stride=1, use_bias=use_bias, sn=sn, name='skip_conv')
            return relu(x + x_init)
        else :
            return x + x_init


def resblock_up(x_init, channels, use_bias=True, training=True, sn=False, name='resblock_up'):
    with tf.name_scope(name):
        with tf.name_scope('res1'):
            x = deconv(x_init, channels, kernel=3, stride=2, use_bias=use_bias, sn=sn, name='deconv_0')
            x = batch_norm(x, training, name='batch_norm_0')
            x = relu(x)

        with tf.name_scope('res2'):
            x = deconv(x, channels, kernel=3, stride=1, use_bias=use_bias, sn=sn, name='deconv_1')
            x = batch_norm(x, training, name='batch_norm_1')

        with tf.name_scope('skip'):
            x_init = deconv(x_init, channels, kernel=3, stride=2, use_bias=use_bias, sn=sn, name='skip_deconv')

        return relu(x + x_init)


def resblock_up_condition(x_init, z, channels, use_bias=True, training=True, sn=False, name='resblock_condition'):
    # See https://github.com/taki0112/BigGAN-Tensorflow
    with tf.name_scope(name):
        with tf.name_scope('res1'):
            x = deconv(x_init, channels, kernel=3, stride=2, use_bias=use_bias, sn=sn, name='deconv_0')
            x = condition_batch_norm(x, z, training, name='cond_bn_0')
            x = relu(x)

        with tf.name_scope('res2'):
            x = deconv(x, channels, kernel=3, stride=1, use_bias=use_bias, sn=sn, name='deconv_1')
            x = condition_batch_norm(x, z, training, name='cond_bn_1')

        with tf.name_scope('skip'):
            x_init = deconv(x_init, channels, kernel=3, stride=2, use_bias=use_bias, sn=sn, name='skip_deconv')

        return relu(x + x_init)


def resblock_down(x_init, channels, use_bias=True, training=True, sn=False, name='resblock_down'):
    with tf.name_scope(name):
        with tf.name_scope('res1'):
            x = conv(x_init, channels, kernel=3, stride=2, pad=1, use_bias=use_bias, sn=sn, name='conv_0')
            x = batch_norm(x, training, name='batch_norm_0')
            x = relu(x)

        with tf.name_scope('res2'):
            x = conv(x, channels, kernel=3, stride=1, pad=1, use_bias=use_bias, sn=sn, name='conv_1')
            x = batch_norm(x, training, name='batch_norm_1')

        with tf.name_scope('skip'):
            x_init = conv(x_init, channels, kernel=3, stride=2, pad=1, use_bias=use_bias, sn=sn, name='skip_conv')

        return relu(x + x_init)

def denseblock(x_init, channels, n_db=6, use_bias=True, training=True, sn=False, name='denseblock'):
    with tf.name_scope(name):
        layers = []
        layers.append(x_init)

        with tf.name_scope('bottle_neck_0'):
            x = conv(x_init, 4 * channels, kernel=1, stride=1, use_bias=use_bias, sn=sn, name='conv_0')
            x = batch_norm(x, training, name='batch_norm_0')
            x = relu(x)

            x = conv(x, channels, kernel=3, stride=1, pad=1, use_bias=use_bias, sn=sn, name='conv_1')
            x = batch_norm(x, training, name='batch_norm_1')
            x = relu(x)

            layers.append(x)

        for i in range(1, n_db):
            with tf.name_scope('bottle_neck_' + str(i)):
                x = tf.concat(layers, axis=-1)

                x = conv(x, 4 * channels, kernel=1, stride=1, use_bias=use_bias, sn=sn, name='conv_0')
                x = batch_norm(x, training, name='batch_norm_0')
                x = relu(x)

                x = conv(x, channels, kernel=3, stride=1, pad=1, use_bias=use_bias, sn=sn, name='conv_1')
                x = batch_norm(x, training, name='batch_norm_1')
                x = relu(x)

                layers.append(x)

        x = tf.concat(layers, axis=-1)

        return x


def res_denseblock(x_init, channels, n_rdb=20, n_rdb_conv=6, use_bias=True, training=True, sn=False, name='res_denseblock'):
    with tf.name_scope(name):
        RDBs = []
        x_input = x_init

        """
        n_rdb = 20 ( RDB number )
        n_rdb_conv = 6 ( per RDB conv layer )
        """

        for k in range(n_rdb):
            with tf.name_scope('RDB_' + str(k)):
                layers = []
                layers.append(x_init)

                x = conv(x_init, channels, kernel=3, stride=1, pad=1, use_bias=use_bias, sn=sn, name='conv_0')
                x = batch_norm(x, training, name='batch_norm_0')
                x = relu(x)

                layers.append(x)

                for i in range(1, n_rdb_conv):
                    x = tf.concat(layers, axis=-1)

                    x = conv(x, channels, kernel=3, stride=1, pad=1, use_bias=use_bias, sn=sn, name='conv_' + str(i))
                    x = batch_norm(x, training, name='batch_norm_' + str(i))
                    x = relu(x)

                    layers.append(x)

                # Local feature fusion
                x = tf.concat(layers, axis=-1)
                x = conv(x, channels, kernel=1, stride=1, use_bias=use_bias, sn=sn, name='conv_last')

                # Local residual learning
                if channels != x_init.shape[-1]:
                    x_init = conv(x_init, channels, kernel=1, stride=1, use_bias=use_bias, sn=sn, name='skip_conv')
                    x = relu(x + x_init)
                else:
                    x = x_init + x

                RDBs.append(x)
                x_init = x

        with tf.name_scope('GFF_1x1'):
            x = tf.concat(RDBs, axis=-1)
            x = conv(x, channels, kernel=1, stride=1, use_bias=use_bias, sn=sn, name='conv')

        with tf.name_scope('GFF_3x3'):
            x = conv(x, channels, kernel=3, stride=1, pad=1, use_bias=use_bias, sn=sn, name='conv')

        # Global residual learning
        if channels != x_input.shape[-1]:
            x_input = conv(x_input, channels, kernel=1, stride=1, use_bias=use_bias, sn=sn, name='skip_conv')
            return relu(x + x_input)
        else:
            return x_input + x

def self_attention(x, use_bias=True, sn=False, name='self_attention'):
    with tf.name_scope(name):
        channels = x.shape[-1]
        f = conv(x, channels // 8, kernel=1, stride=1, use_bias=use_bias, sn=sn, name='f_conv') # [bs, h, w, c']
        g = conv(x, channels // 8, kernel=1, stride=1, use_bias=use_bias, sn=sn, name='g_conv') # [bs, h, w, c']
        h = conv(x, channels, kernel=1, stride=1, use_bias=use_bias, sn=sn, name='h_conv') # [bs, h, w, c]

        # N = h * w
        s = tf.matmul(hw_flatten(g), hw_flatten(f), transpose_b=True) # [bs, N, N]

        beta = tf.nn.softmax(s) # attention map

        o = tf.matmul(beta, hw_flatten(h))
        gamma = tf.Variable(initial_value=tf.constant(0.0, shape=[1]), name='gamma')

        o = tf.reshape(o, shape=x.shape)
        x = gamma * o + x

        return x

def self_attention_with_pooling(x, use_bias=True, sn=False, name='self_attention_pooling'):
    with tf.name_scope(name):
        channels = x.shape[-1]
        f = conv(x, channels // 8, kernel=1, stride=1, use_bias=use_bias, sn=sn, name='f_conv') # [bs, h, w, c']
        f = max_pooling(f, pool_size=2)

        g = conv(x, channels // 8, kernel=1, stride=1, use_bias=use_bias, sn=sn, name='g_conv') # [bs, h, w, c']

        h = conv(x, channels // 2, kernel=1, stride=1, use_bias=use_bias, sn=sn, name='h_conv') # [bs, h, w, c]
        h = max_pooling(h, pool_size=2)

        # N = h * w
        s = tf.matmul(hw_flatten(g), hw_flatten(f), transpose_b=True) # [bs, N, N]

        beta = tf.nn.softmax(s) # attention map

        o = tf.matmul(beta, hw_flatten(h))
        gamma = tf.Variable(initial_value=tf.constant(0.0, shape=[1]), name='gamma')

        o = tf.reshape(o, shape=[x.shape[0], x.shape[1], x.shape[2], channels // 2]) # [bs, h, w, c]
        o = conv(o, channels, kernel=1, stride=1, use_bias=use_bias, sn=sn, name='attn_conv')

        x = gamma * o + x

        return x

def squeeze_excitation(x, ratio=16, use_bias=True, sn=False, name='senet'):
    with tf.name_scope(name):
        channels = x.shape[-1]
        squeeze = global_avg_pooling(x)

        excitation = fully_connected(squeeze, units=channels // ratio, use_bias=use_bias, sn=sn, name='fc_0')
        excitation = relu(excitation)
        excitation = fully_connected(excitation, units=channels, use_bias=use_bias, sn=sn, name='fc_1')
        excitation = sigmoid(excitation)

        excitation = tf.reshape(excitation, [-1, 1, 1, channels])

        scale = x * excitation

        return scale

def convolution_block_attention(x, ratio=16, use_bias=True, sn=False, name='cbam'):
    with tf.name_scope(name):
        channels = x.shape[-1]
        # for reuse
        fully_connect_layer_1 = tf.keras.layers.Dense(channels // ratio, use_bias=use_bias, name='fc_0')
        fully_connect_layer_2 = tf.keras.layers.Dense(channels, use_bias=use_bias, name='fc_1')

        if sn:
            fully_connect_layer_1 = SpectralNormalization(fully_connect_layer_1)
            fully_connect_layer_2 = SpectralNormalization(fully_connect_layer_2)

        with tf.name_scope('channel_attention'):
            x_gap = global_avg_pooling(x)
            x_gap = fully_connect_layer_1(x_gap)
            x_gap = relu(x_gap)
            x_gap = fully_connect_layer_2(x_gap)

            x_gmp = global_max_pooling(x)
            x_gmp = fully_connect_layer_1(x_gmp)
            x_gmp = relu(x_gmp)
            x_gmp = fully_connect_layer_2(x_gmp)

            scale = tf.reshape(x_gap + x_gmp, [-1, 1, 1, channels])
            scale = sigmoid(scale)

            x = x * scale

        with tf.name_scope('spatial_attention'):
            x_channel_avg_pooling = tf.reduce_mean(x, axis=-1, keepdims=True)
            x_channel_max_pooling = tf.reduce_max(x, axis=-1, keepdims=True)
            scale = tf.concat([x_channel_avg_pooling, x_channel_max_pooling], axis=-1)

            scale = conv(scale, channels=1, kernel=7, stride=1, pad=3, pad_type='reflect', use_bias=False, sn=sn, name='conv')
            scale = sigmoid(scale)

            x = x * scale

        return x

def global_context_block(x, channels, use_bias=True, sn=False, name='gc_block'):
    with tf.name_scope(name):
        with tf.name_scope('context_modeling'):
            bs, h, w, c =x.get_shape().as_list()
            x_init = x
            x_init = hw_flatten(x_init)  # [N, H*W, C]
            x_init = tf.transpose(x_init, perm=[0, 2, 1])
            x_init = tf.expand_dims(x_init, axis=1)

            context_mask = conv(x, channels=1, kernel=1, stride=1, use_bias=use_bias, sn=sn, name='conv')
            context_mask = hw_flatten(context_mask)
            context_mask = tf.nn.softmax(context_mask, axis=1)  # [N, H*W, 1]
            context_mask = tf.transpose(context_mask, perm=[0, 2, 1])
            context_mask = tf.expand_dims(context_mask, axis=-1)

            context = tf.matmul(x_init, context_mask)
            context = tf.reshape(context, shape=[bs, 1, 1, c])

        with tf.name_scope('transform_0'):
            context_transform = conv(context, channels, kernel=1, stride=1, use_bias=use_bias, sn=sn, name='conv_0')
            context_transform = layer_norm(context_transform, name='layer_norm')
            context_transform = relu(context_transform)
            context_transform = conv(context_transform, channels=c, kernel=1, stride=1, use_bias=use_bias, sn=sn, name='conv_1')
            context_transform = sigmoid(context_transform)

            x = x * context_transform

        with tf.name_scope('transform_1'):
            context_transform = conv(context, channels, kernel=1, stride=1, use_bias=use_bias, sn=sn, name='conv_0')
            context_transform = layer_norm(context_transform, name='layer_norm')
            context_transform = relu(context_transform)
            context_transform = conv(context_transform, channels=c, kernel=1, stride=1, use_bias=use_bias, sn=sn, name='conv_1')

            x = x + context_transform

        return x

def srm_block(x, use_bias=False, training=True, name='srm_block'):
    with tf.name_scope(name):

        bs, h, w, channels = x.get_shape().as_list()  # c = channels
        x = tf.reshape(x, shape=[bs, -1, channels])  # [bs, h*w, c]

        x_mean, x_var = tf.nn.moments(x, axes=1, keepdims=True)  # [bs, 1, c]
        x_std = tf.sqrt(x_var + 1e-5)

        t = tf.concat([x_mean, x_std], axis=1)  # [bs, 2, c]

        z = tf.keras.layers.Conv1D(channels, kernel_size=2, strides=1, use_bias=use_bias, name='1d_conv')(t)
        z = batch_norm(z, training=training, name='batch_norm')

        g = tf.sigmoid(z)

        x = tf.reshape(x * g, shape=[bs, h, w, channels])

        return x


##################################################################################
# Normalization
##################################################################################

def batch_norm(x, training=False, name='batch_norm'):
    x = tf.keras.layers.BatchNormalization(momentum=0.9, epsilon=1e-05,
                                           center=True, scale=True,
                                           name=name)(x, training=training)
    return x

def instance_norm(x, name='instance_norm'):
    x = tfa.layers.normalizations.InstanceNormalization(epsilon=1e-5,
                                                        scale=True,
                                                        center=True,
                                                        name=name)(x)
    return x


def layer_norm(x, name='layer_norm'):
    return tf.keras.layers.LayerNormalization(center=True, scale=True, name=name)(x)

def group_norm(x, groups=32, name='group_norm'):
    return tfa.layers.normalizations.GroupNormalization(groups=groups, epsilon=1e-05,
                                                        center=True, scale=True,
                                                        name=name)(x)

def adaptive_instance_norm(content, gamma, beta, epsilon=1e-5):
    # gamma, beta = style_mean, style_std from MLP
    # See https://github.com/taki0112/MUNIT-Tensorflow

    c_mean, c_var = tf.nn.moments(content, axes=[1, 2], keepdims=True)
    c_std = tf.sqrt(c_var + epsilon)

    return gamma * ((content - c_mean) / c_std) + beta

def adaptive_layer_instance_norm(x, gamma, beta, smoothing=True, name='ada_layer_instance_norm') :
    # proposed by UGATIT
    # https://github.com/taki0112/UGATIT
    with tf.name_scope(name):
        ch = x.shape[-1]
        eps = 1e-5

        ins_mean, ins_sigma = tf.nn.moments(x, axes=[1, 2], keepdims=True)
        x_ins = (x - ins_mean) / (tf.sqrt(ins_sigma + eps))

        ln_mean, ln_sigma = tf.nn.moments(x, axes=[1, 2, 3], keepdims=True)
        x_ln = (x - ln_mean) / (tf.sqrt(ln_sigma + eps))

        rho = tf.Variable(initial_value=tf.constant(1.0, shape=[ch]),
                          constraint=lambda x: tf.clip_by_value(x, clip_value_min=0.0, clip_value_max=1.0),
                          name='rho')
        if smoothing :
            rho = tf.clip_by_value(rho - tf.constant(0.1), 0.0, 1.0)

        x_hat = rho * x_ins + (1 - rho) * x_ln


        x_hat = x_hat * gamma + beta

        return x_hat


def condition_batch_norm(x, z, training=True, name='condition_bn'):
    # See https://github.com/taki0112/BigGAN-Tensorflow
    with tf.name_scope(name):
        _, _, _, c = x.get_shape().as_list()
        decay = 0.9
        epsilon = 1e-05

        test_mean = tf.Variable(initial_value=tf.constant(0.0, shape=[c]), trainable=False, name='pop_mean')
        test_var = tf.Variable(initial_value=tf.constant(1.0, shape=[c]), trainable=False, name='pop_var')


        beta = fully_connected(z, units=c, name='beta_fc')
        gamma = fully_connected(z, units=c, name='gamma_fc')

        beta = tf.reshape(beta, shape=[-1, 1, 1, c])
        gamma = tf.reshape(gamma, shape=[-1, 1, 1, c])

        if training:
            batch_mean, batch_var = tf.nn.moments(x, [0, 1, 2])
            ema_mean = test_mean.assign(test_mean * decay + batch_mean * (1 - decay))
            ema_var = test_var.assign(test_var * decay + batch_var * (1 - decay))

            with tf.control_dependencies([ema_mean, ema_var]):
                return tf.nn.batch_normalization(x, batch_mean, batch_var, beta, gamma, epsilon, name=name)
        else:
            return tf.nn.batch_normalization(x, test_mean, test_var, beta, gamma, epsilon, name=name)


def batch_instance_norm(x, name='batch_instance_norm'):
    with tf.name_scope(name):
        ch = x.shape[-1]
        eps = 1e-5

        batch_mean, batch_sigma = tf.nn.moments(x, axes=[0, 1, 2], keepdims=True)
        x_batch = (x - batch_mean) / (tf.sqrt(batch_sigma + eps))

        ins_mean, ins_sigma = tf.nn.moments(x, axes=[1, 2], keepdims=True)
        x_ins = (x - ins_mean) / (tf.sqrt(ins_sigma + eps))

        rho = tf.Variable(initial_value=tf.constant(1.0, shape=[ch]),
                          constraint=lambda x: tf.clip_by_value(x, clip_value_min=0.0, clip_value_max=1.0),
                          name='rho')

        gamma = tf.Variable(initial_value=tf.constant(1.0, shape=[ch]),
                            name='gamma')

        beta = tf.Variable(initial_value=tf.constant(1.0, shape=[ch]),
                           name='beta')

        x_hat = rho * x_batch + (1 - rho) * x_ins
        x_hat = x_hat * gamma + beta

        return x_hat

def layer_instance_norm(x, name='layer_instance_norm') :
    # proposed by UGATIT
    # https://github.com/taki0112/UGATIT
    with tf.name_scope(name):
        ch = x.shape[-1]
        eps = 1e-5

        ins_mean, ins_sigma = tf.nn.moments(x, axes=[1, 2], keepdims=True)
        x_ins = (x - ins_mean) / (tf.sqrt(ins_sigma + eps))

        ln_mean, ln_sigma = tf.nn.moments(x, axes=[1, 2, 3], keepdims=True)
        x_ln = (x - ln_mean) / (tf.sqrt(ln_sigma + eps))

        rho = tf.Variable(initial_value=tf.constant(0.0, shape=[ch]),
                          constraint=lambda x: tf.clip_by_value(x, clip_value_min=0.0, clip_value_max=1.0),
                          name='rho')
        gamma = tf.Variable(initial_value=tf.constant(1.0, shape=[ch]), name='gamma')
        beta = tf.Variable(initial_value=tf.constant(0.0, shape=[ch]), name='beta')

        x_hat = rho * x_ins + (1 - rho) * x_ln

        x_hat = x_hat * gamma + beta

        return x_hat

def pixel_norm(x, epsilon=1e-8):
    return x * tf.math.rsqrt(tf.reduce_mean(tf.square(x), axis=-1, keepdims=True) + epsilon)

def switch_norm(x, name='switch_norm'):
    with tf.name_scope(name):
        ch = x.shape[-1]
        eps = 1e-5

        batch_mean, batch_var = tf.nn.moments(x, [0, 1, 2], keepdims=True)
        ins_mean, ins_var = tf.nn.moments(x, [1, 2], keepdims=True)
        layer_mean, layer_var = tf.nn.moments(x, [1, 2, 3], keepdims=True)

        gamma = tf.Variable(initial_value=tf.constant(1.0, shape=[ch]), name='gamma')
        beta = tf.Variable(initial_value=tf.constant(0.0, shape=[ch]), name='beta')

        mean_weight = tf.nn.softmax(tf.Variable(initial_value=tf.constant(1.0, shape=[3]), name="mean_weight"))
        var_wegiht = tf.nn.softmax(tf.Variable(initial_value=tf.constant(1.0, shape=[3]), name="var_weight"))

        mean = mean_weight[0] * batch_mean + mean_weight[1] * ins_mean + mean_weight[2] * layer_mean
        var = var_wegiht[0] * batch_var + var_wegiht[1] * ins_var + var_wegiht[2] * layer_var

        x = (x - mean) / (tf.sqrt(var + eps))
        x = x * gamma + beta

        return x

##################################################################################
# Activation Function
##################################################################################

def lrelu(x, alpha=0.01):
    # pytorch alpha is 0.01
    return tf.nn.leaky_relu(x, alpha, name='lrelu')


def relu(x):
    return tf.nn.relu(x, name='relu')


def tanh(x):
    return tf.tanh(x, name='tanh')


def sigmoid(x):
    return tf.sigmoid(x, name='sigmoid')


def swish(x):
    return x * tf.sigmoid(x, name='sigmoid')


def elu(x):
    return tf.nn.elu(x, name='elu')


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

def global_avg_pooling(x, keepdims=True):
    gap = tf.reduce_mean(x, axis=[1, 2], keepdims=keepdims)
    return gap


def global_max_pooling(x, keepdims=True):
    gmp = tf.reduce_max(x, axis=[1, 2], keepdims=keepdims)
    return gmp


def max_pooling(x, pool_size=2):
    x = tf.keras.layers.MaxPool2D(pool_size=pool_size, strides=pool_size, padding='SAME')(x)
    return x

def avg_pooling(x, pool_size=2):
    x = tf.keras.layers.AvgPool2D(pool_size=pool_size, strides=pool_size, padding='SAME')(x)
    return x


def flatten(x):
    return tf.keras.layers.Flatten()(x)


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
            real_loss = tf.reduce_mean(relu(1.0 - Ra_real_logit))
            fake_loss = tf.reduce_mean(relu(1.0 + Ra_fake_logit))

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
            real_loss = tf.reduce_mean(relu(1.0 - real_logit))
            fake_loss = tf.reduce_mean(relu(1.0 + fake_logit))

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
            fake_loss = tf.reduce_mean(relu(1.0 - Ra_fake_logit))
            real_loss = tf.reduce_mean(relu(1.0 + Ra_real_logit))

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
    grad_norm = tf.norm(flatten(grad), axis=-1)

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
            real_loss = tf.reduce_sum(discriminator(real_images)) # In some cases, you may use reduce_mean

        real_grads = p_tape.gradient(real_loss, real_images)
        r1_penalty = 0.5 * r1_gamma * tf.reduce_mean(tf.reduce_sum(tf.square(real_grads), axis=[1, 2, 3]))

    if r2_gamma != 0:
        with tf.GradientTape() as p_tape:
            p_tape.watch(fake_images)
            fake_loss = tf.reduce_sum(discriminator(fake_images)) # In some cases, you may use reduce_mean

        fake_grads = p_tape.gradient(fake_loss, fake_images)
        r2_penalty = 0.5 * r2_gamma * tf.reduce_mean(tf.reduce_sum(tf.square(fake_grads), axis=[1, 2, 3]))

    return r1_penalty + r2_penalty

def inverse_stereographic_projection(x) :

    x_u = tf.transpose(2 * x) / (tf.pow(tf.norm(x, axis=-1), 2) + 1.0)
    x_v = (tf.pow(tf.norm(x, axis=-1), 2) - 1.0) / (tf.pow(tf.norm(x, axis=-1), 2) + 1.0)

    x_projection = tf.transpose(tf.concat([x_u, [x_v]], axis=0))

    return x_projection

def sphere_loss(x, y) :

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

def various_rnn(x, n_hidden=128, n_layer=1, dropout_rate=0.5, training=True, bidirectional=True, return_state=False, rnn_type='lstm', name='rnn') :

    if rnn_type.lower() == 'lstm' :
        cell_type = tf.keras.layers.LSTMCell
    elif rnn_type.lower() == 'gru' :
        cell_type = tf.keras.layers.GRUCell
    else :
        raise NotImplementedError
    rnn = tf.keras.layers.RNN([cell_type(units=n_hidden, dropout=dropout_rate) for _ in range(n_layer)], return_sequences=True, return_state=return_state, name=name)

    if bidirectional:
        rnn = tf.keras.layers.Bidirectional(rnn, name=name)


    """
    if also return_state=True, 
    whole_sequence, forward_hidden, forward_cell, backward_hidden, backward_cell (LSTM)
    whole_sequence, forward_hidden, forward_cell (GRU)
    sent_emb = tf.concat([forward_hidden, backward_hidden], axis=-1)
    """
    if return_state:
        if bidirectional:
            if rnn_type.lower() == 'gru':
                output, forward_h, backward_h = rnn(x, training=training)
            else:  # LSTM
                output, forward_state, backward_state = rnn(x, training=training)
                forward_h, backward_h = forward_state[0], backward_state[0]
                forward_c, backward_c = forward_state[1], backward_state[1]

            sent_emb = tf.concat([forward_h, backward_h], axis=-1)
        else:
            if rnn_type.lower() == 'gru':
                output, forward_h = rnn(x, training=training)
            else:
                output, forward_state = rnn(x, training=training)
                forward_h, forward_c = forward_state

            sent_emb = forward_h

    else:
        output = rnn(x, training=training)
        sent_emb = output[:, -1, :]

    word_emb = output

    return word_emb, sent_emb

def embed_sequence(x, n_words, embed_dim, init_range=0.1, trainable=True, name='embed_layer') :

    emeddings = tf.keras.layers.Embedding(input_dim=n_words, output_dim=embed_dim,
                                          embeddings_initializer=tf.random_uniform_initializer(minval=-init_range, maxval=init_range),
                                          trainable=trainable, name=name)(x)
    return emeddings