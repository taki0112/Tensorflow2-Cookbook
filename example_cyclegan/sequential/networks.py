from ops_seq_cyclegan import *


class Generator(tf.keras.Model):
    def __init__(self, channels, n_res, name):
        super(Generator, self).__init__(name=name)
        self.channels = channels
        self.n_res = n_res

        self.encoder = Encoder(self.channels, name='g_encoder')
        self.bottleneck = Bottleneck(self.channels * 4, self.n_res, name='g_bottleneck')
        self.decoder = Decoder(self.channels * 4, name='g_decoder')


    def call(self, inputs, training=None, mask=None):

        x = self.encoder(inputs, training=training)
        x = self.bottleneck(x, training=training)
        x = self.decoder(x, training=training)

        return x

    def build_summary(self, input_shape, detail_summary=False):
        x_init = tf.keras.layers.Input(input_shape, name='g_input')
        enc_x = self.encoder(x_init)
        res_x = self.bottleneck(enc_x)
        x = self.decoder(res_x)

        self.build_model = tf.keras.Model(x_init, x, name=self.name)
        self.build_model.summary()

        if detail_summary :
            self.encoder.model.summary()
            self.bottleneck.model.summary()
            self.decoder.model.summary()

    def count_parameter(self):

        x = self.encoder.count_params() + self.bottleneck.count_params() + self.decoder.count_params()

        return x

class Encoder(tf.keras.layers.Layer):
    def __init__(self, channels, name):
        super(Encoder, self).__init__(name=name)
        self.channels = channels

        self.model = self.architecture()

    def architecture(self):
        model = []

        model += [Conv(self.channels, kernel=7, stride=1, pad=3, pad_type='reflect', use_bias=False, name='conv')]
        model += [InstanceNorm(name='ins_norm')]
        model += [Relu()]

        for i in range(2):
            model += [Conv(self.channels * 2, kernel=3, stride=2, pad=1, pad_type='reflect', use_bias=False, name='down_conv_' + str(i))]
            model += [InstanceNorm(name='down_ins_norm_' + str(i))]
            model += [Relu()]

            self.channels = self.channels * 2

        model = Sequential(model)

        return model

    def call(self, x_init, training=None, mask=None):
        x = self.model(x_init, training=training)

        return x

class Bottleneck(tf.keras.layers.Layer):
    def __init__(self, channels, n_res, name):
        super(Bottleneck, self).__init__(name=name)
        self.channels = channels
        self.n_res = n_res

        self.model = self.architecture()

    def architecture(self):
        model = []

        for i in range(self.n_res):
            model += [ResBlock(self.channels, use_bias=False, name='resblock_' + str(i))]

        model = Sequential(model)

        return model

    def call(self, x_init, training=None, mask=None):
        x = self.model(x_init, training=training)

        return x

class Decoder(tf.keras.layers.Layer):
    def __init__(self, channels, name):
        super(Decoder, self).__init__(name=name)
        self.channels = channels

        self.model = self.architecture()

    def architecture(self):
        model = []
        for i in range(2):
            model += [Deconv(self.channels // 2, kernel=3, stride=2, use_bias=False, name='deconv_' + str(i))]
            model += [InstanceNorm(name='up_ins_norm_' + str(i))]
            model += [Relu()]

            self.channels = self.channels // 2

        model += [Conv(channels=3, kernel=7, stride=1, pad=3, pad_type='reflect', use_bias=False, name='g_logit')]
        model += [Tanh()]

        model = Sequential(model)

        return model

    def call(self, x_init, training=None, mask=None):
        x = self.model(x_init, training=training)

        return x

class Discriminator(tf.keras.Model):
    def __init__(self, channels, n_dis, sn, name):
        super(Discriminator, self).__init__(name=name)
        self.channels = channels
        self.n_dis = n_dis
        self.sn = sn

        self.model = self.architecture()

    def architecture(self):
        channels = self.channels
        model = []

        model += [Conv(channels, kernel=4, stride=2, pad=1, pad_type='reflect', use_bias=True, sn=self.sn, name='conv_0')]
        model += [Leaky_Relu(alpha=0.2)]

        for i in range(1, self.n_dis):
            model += [Conv(channels * 2, kernel=4, stride=2, pad=1, pad_type='reflect', use_bias=False, sn=self.sn, name='conv_' + str(i))]
            model += [InstanceNorm(name='ins_norm_' + str(i))]
            model += [Leaky_Relu(alpha=0.2)]

            channels = channels * 2

        model += [Conv(channels * 2, kernel=4, stride=1, pad=1, pad_type='reflect', use_bias=False, sn=self.sn, name='last_conv')]
        model += [InstanceNorm(name='last_ins_norm')]
        model += [Leaky_Relu(alpha=0.2)]

        model += [Conv(channels=1, kernel=4, stride=1, pad=1, pad_type='reflect', use_bias=True, sn=self.sn, name='d_logit')]
        model += [Tanh()]

        model = Sequential(model, name='d_encoder')

        return model

    def call(self, x_init, training=None, mask=None):

        x = self.model(x_init, training=training)

        return x

    def build_summary(self, input_shape, detail_summary=False):
        x_init = tf.keras.layers.Input(input_shape, name='d_input')

        x = self.model(x_init)

        self.build_model = tf.keras.Model(x_init, x, name=self.name)
        self.build_model.summary()

        if detail_summary:
            self.model.summary()

    def count_parameter(self):
        x = self.model.count_params()
        return x

"""
class Generator(tf.keras.Model):
    def __init__(self, channels, n_res, name):
        super(Generator, self).__init__(name=name)
        self.channels = channels
        self.n_res = n_res

        self.model = self.architecture()

    def architecture(self):
        model = []
        model += [Conv(self.channels, kernel=7, stride=1, pad=3, pad_type='reflect', use_bias=False, name='conv')]
        model += [InstanceNorm(name='ins_norm')]
        model += [Relu()]

        for i in range(2):
            model += [Conv(self.channels * 2, kernel=3, stride=2, pad=1, pad_type='reflect', use_bias=False,
                           name='down_conv_' + str(i))]
            model += [InstanceNorm(name='down_ins_norm_' + str(i))]
            model += [Relu()]

            self.channels = self.channels * 2

        for i in range(self.n_res):
            model += [ResBlock(self.channels, use_bias=False, name='resblock_' + str(i))]

        for i in range(2):
            model += [Deconv(self.channels // 2, kernel=3, stride=2, use_bias=False, name='deconv_' + str(i))]
            model += [InstanceNorm(name='up_ins_norm_' + str(i))]
            model += [Relu()]

            self.channels = self.channels // 2

        model += [Conv(channels=3, kernel=7, stride=1, pad=3, pad_type='reflect', use_bias=False, name='g_logit')]
        model += [Tanh()]

        model = Sequential(model)

        return model

    def call(self, inputs, training=None, mask=None):

        x = self.model(inputs, training=training)

        return x

    def build_summary(self, input_shape, detail_summary=False):
        x_init = tf.keras.layers.Input(input_shape, name='d_input')

        x = self.model(x_init)

        self.build_model = tf.keras.Model(x_init, x, name=self.name)
        self.build_model.summary()

        if detail_summary:
            self.model.summary()

    def count_parameter(self):

        x = self.model.count_params()

        return x

"""