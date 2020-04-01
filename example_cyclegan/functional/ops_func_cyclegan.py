import tensorflow as tf
import tensorflow_addons as tfa

weight_initializer = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02)
weight_regularizer = tf.keras.regularizers.l2(0.0001)

##################################################################################
# Layer
##################################################################################

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
                                   strides=stride, use_bias=use_bias, bias_initializer=tf.keras.initializers.zeros()),
                                  name='sn_' + name)(x)

    else:
        x = tf.keras.layers.Conv2D(filters=channels, kernel_size=kernel,
                                   kernel_initializer=weight_initializer, kernel_regularizer=weight_regularizer,
                                   strides=stride, use_bias=use_bias, bias_initializer=tf.keras.initializers.zeros(), name=name)(x)

    return x

def deconv(x, channels, kernel=4, stride=2, padding='SAME', use_bias=True, sn=False, name='deconv'):
    if sn :
        x = SpectralNormalization(tf.keras.layers.Conv2DTranspose(filters=channels, kernel_size=kernel, kernel_initializer=weight_initializer, kernel_regularizer=weight_regularizer,
                                                                  strides=stride, use_bias=use_bias, bias_initializer=tf.keras.initializers.zeros(),
                                                                  padding=padding),
                                  name='sn_' + name)(x)
    else :
        x = tf.keras.layers.Conv2DTranspose(filters=channels, kernel_size=kernel, kernel_initializer=weight_initializer,
                                        kernel_regularizer=weight_regularizer,
                                        strides=stride, use_bias=use_bias, bias_initializer=tf.keras.initializers.zeros(),
                                        padding=padding,
                                        name=name)(x)

    return x

##################################################################################
# Residual-block
##################################################################################

def resblock(x_init, channels, use_bias=True, sn=False, name='resblock'):

    x = conv(x_init, channels, kernel=3, stride=1, pad=1, pad_type='reflect', use_bias=use_bias, sn=sn, name=name + '_conv1')
    x = instance_norm(x, name=name + '_ins_norm_1')
    x = relu(x)

    x = conv(x, channels, kernel=3, stride=1, pad=1, pad_type='reflect', use_bias=use_bias, sn=sn, name=name + '_conv2')
    x = instance_norm(x, name=name + '_ins_norm_2')

    return x + x_init

##################################################################################
# Activation function
##################################################################################

def lrelu(x, alpha=0.01):
    # pytorch alpha is 0.01
    return tf.nn.leaky_relu(x, alpha, name='lrelu')

def relu(x):
    return tf.nn.relu(x, name='relu')

def tanh(x):
    return tf.tanh(x, name='tanh')


##################################################################################
# Normalization function
##################################################################################

def instance_norm(x, name='instance_norm'):
    x = tfa.layers.normalizations.InstanceNormalization(epsilon=1e-5,
                                                        scale=True,
                                                        center=True,
                                                        name=name)(x)
    return x

def batch_norm(x, training=True, name='batch_norm'):
    x = tf.keras.layers.BatchNormalization(momentum=0.9, epsilon=1e-05,
                                           center=True, scale=True,
                                           name=name)(x, training=training)
    return x

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

        self.v = self.add_weight(shape=(1, self.w_shape[0] * self.w_shape[1] * self.w_shape[2]),
                                 initializer=tf.initializers.TruncatedNormal(stddev=0.02),
                                 trainable=False,
                                 name='sn_v',
                                 dtype=tf.float32)

        self.u = self.add_weight(shape=(1, self.w_shape[-1]),
                                 initializer=tf.initializers.TruncatedNormal(stddev=0.02),
                                 trainable=False,
                                 name='sn_u',
                                 dtype=tf.float32)

        super(SpectralNormalization, self).build()

    def call(self, inputs, training=None, mask=None):
        self.update_weights()
        output = self.layer(inputs)
        self.restore_weights()  # Restore weights because of this formula "W = W - alpha * W_SN`"
        return output

    def update_weights(self):
        w_reshaped = tf.reshape(self.w, [-1, self.w_shape[-1]])

        u_hat = self.u
        v_hat = self.v  # init v vector

        if self.do_power_iteration:
            for _ in range(self.iteration):
                v_ = tf.matmul(u_hat, tf.transpose(w_reshaped))
                v_hat = v_ / (tf.reduce_sum(v_ ** 2) ** 0.5 + self.eps)

                u_ = tf.matmul(v_hat, w_reshaped)
                u_hat = u_ / (tf.reduce_sum(u_ ** 2) ** 0.5 + self.eps)

        sigma = tf.matmul(tf.matmul(v_hat, w_reshaped), tf.transpose(u_hat))
        self.u.assign(u_hat)
        self.v.assign(v_hat)

        self.layer.kernel.assign(self.w / sigma)

    def restore_weights(self):
        self.layer.kernel.assign(self.w)

##################################################################################
# Loss function
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
        real_loss = tf.reduce_mean(relu(1.0 - real_logit))
        fake_loss = tf.reduce_mean(relu(1.0 + fake_logit))

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

def L1_loss(x, y):
    loss = tf.reduce_mean(tf.abs(x - y))

    return loss

def regularization_loss(model):
    loss = tf.nn.scale_regularization_loss(model.losses)

    return loss