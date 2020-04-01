from ops_func_cyclegan import *

class Generator(tf.keras.Model):
    def __init__(self, input_shape, channel, n_res, name, training=True):
        super(Generator, self).__init__(name=name)

        self.encoder_ch = channel
        self.resblock_ch = channel * 4
        self.decoder_ch = channel * 4
        self.n_res = n_res
        self.network_name = name

        self.inputs = tf.keras.layers.Input(input_shape, name='g_input')

        self.encoder = Encoder(input_shape, self.encoder_ch, name='g_encoder', training=training)
        self.bottleneck = Bottleneck(self.encoder.out_shape, self.resblock_ch, self.n_res, name='g_bottleneck', training=training)
        self.decoder = Decoder(self.bottleneck.out_shape, self.decoder_ch, name='g_decoder', training=training)

    def call(self, inputs, training=None, mask=None):

        x = self.encoder(inputs)
        x = self.bottleneck(x)
        x = self.decoder(x)

        return x

    def build_summary(self, detail_summary=False):

        enc_x = self.encoder(self.inputs)
        res_x = self.bottleneck(enc_x)
        x = self.decoder(res_x)

        self.model = tf.keras.Model(self.inputs, x, name=self.network_name)
        self.model.summary()

        if detail_summary:
            self.encoder.build_summary()
            self.bottleneck.build_summary()
            self.decoder.build_summary()

    def count_parameter(self):

        enc_x = self.encoder(self.inputs)
        res_x = self.bottleneck(enc_x)
        x = self.decoder(res_x)

        self.model = tf.keras.Model(self.inputs, x, name=self.network_name)

        return self.model.count_params()

class Encoder(tf.keras.layers.Layer):
    def __init__(self, input_shape, channel, name, training=True):
        super(Encoder, self).__init__(name=name)
        self.channel = channel
        inputs = tf.keras.layers.Input(input_shape, name=name + '_input')

        self.model = self.architecture(inputs, name, training)
        self.out_shape = self.model.output_shape[1:]

    def architecture(self, x_init, name, training):
        x = conv(x_init, channels=self.channel, kernel=7, stride=1, pad=3, pad_type='reflect', use_bias=False, name='conv')
        x = instance_norm(x, name='ins_norm')
        x = relu(x)

        for i in range(2):
            x = conv(x, self.channel * 2, kernel=3, stride=2, pad=1, use_bias=False, name='down_conv_' + str(i))
            x = instance_norm(x, name='down_ins_norm_' + str(i))
            x = relu(x)

            self.channel = self.channel * 2

        return tf.keras.Model(x_init, x, name=name + '_architecture')

    def call(self, x, training=None, mask=None):
        x = self.model(x)

        return x

    def build_summary(self):
        self.model.summary()

class Bottleneck(tf.keras.layers.Layer):
    def __init__(self, input_shape, channel, n_res, name, training=True):
        super(Bottleneck, self).__init__(name=name)
        self.channel = channel
        self.n_res = n_res

        inputs = tf.keras.layers.Input(input_shape, name=name + '_input')

        self.model = self.architecture(inputs, name, training)
        self.out_shape = self.model.output_shape[1:]

    def architecture(self, x_init, name, training):
        x = x_init
        for i in range(self.n_res):
            x = resblock(x, self.channel, use_bias=False, name='resblock_' + str(i))

        return tf.keras.Model(x_init, x, name=name + '_architecture')

    def call(self, x, training=None, mask=None):

        x = self.model(x)

        return x

    def build_summary(self):
        self.model.summary()

class Decoder(tf.keras.layers.Layer):
    def __init__(self, input_shape, channel, name, training=True):
        super(Decoder, self).__init__(name=name)
        self.channel = channel

        inputs = tf.keras.layers.Input(input_shape, name=name + '_input')

        self.model = self.architecture(inputs, name, training)
        self.out_shape = self.model.output_shape[1:]

    def architecture(self, x_init, name, training):
        x = x_init
        for i in range(2):
            x = deconv(x, self.channel // 2, kernel=3, stride=2, padding='SAME', use_bias=False, name='deconv_' + str(i))
            x = instance_norm(x, name='up_ins_norm_' + str(i))
            x = relu(x)

            self.channel = self.channel // 2

        x = conv(x, channels=3, kernel=7, stride=1, pad=3, pad_type='reflect', use_bias=True, name='g_logit')
        x = tanh(x)

        return tf.keras.Model(x_init, x, name=name + '_architecture')

    def call(self, x, training=None, mask=None):

        x = self.model(x)

        return x

    def build_summary(self):

        self.model.summary()

class Discriminator(tf.keras.Model):
    def __init__(self, input_shape, channel, n_dis, sn, name, training=True):
        super(Discriminator, self).__init__(name=name)

        self.encoder_ch = channel
        self.n_dis = n_dis
        self.sn = sn
        self.network_name = name

        self.inputs = tf.keras.layers.Input(input_shape, name='d_input')

        self.encoder = Dis_encoder(input_shape, self.encoder_ch, self.n_dis, self.sn, name='d_encoder', training=training)

    def call(self, inputs, training=None, mask=None):

        x = self.encoder(inputs)

        return x

    def build_summary(self, detail_summary=False):

        x = self.encoder(self.inputs)

        self.model = tf.keras.Model(self.inputs, x, name=self.network_name)

        self.model.summary()
        if detail_summary:
            self.encoder.build_summary()

    def count_parameter(self):

        x = self.encoder(self.inputs)

        self.model = tf.keras.Model(self.inputs, x, name=self.network_name)

        return self.model.count_params()

class Dis_encoder(tf.keras.layers.Layer):
    def __init__(self, input_shape, channel, n_dis, sn, name, training):
        super(Dis_encoder, self).__init__(name=name)
        self.channel = channel
        self.n_dis = n_dis
        self.sn = sn

        inputs = tf.keras.layers.Input(input_shape, name=name + '_input')

        self.model = self.architecture(inputs, name, training)
        self.out_shape = self.model.output_shape[1:]

    def architecture(self, x_init, name, training):
        x = conv(x_init, self.channel, kernel=4, stride=2, pad=1, pad_type='reflect', use_bias=True, sn=self.sn, name='conv_0')
        x = lrelu(x, 0.2)

        for i in range(1, self.n_dis):
            x = conv(x, self.channel * 2, kernel=4, stride=2, pad=1, pad_type='reflect', use_bias=False, sn=self.sn, name='conv_' + str(i))
            x = instance_norm(x, name='ins_norm_' + str(i))
            x = lrelu(x, 0.2)

            self.channel = self.channel * 2

        x = conv(x, self.channel * 2, kernel=4, stride=1, pad=1, pad_type='reflect', use_bias=False, sn=self.sn, name='last_conv')
        x = instance_norm(x, name='last_ins_norm')
        x = lrelu(x, 0.2)

        x = conv(x, channels=1, kernel=4, stride=1, pad=1, pad_type='reflect', use_bias=True, sn=self.sn, name='d_logit')

        return tf.keras.Model(x_init, x, name=name + '_architecture')

    def call(self, x, training=None, mask=None):

        x = self.model(x)

        return x

    def build_summary(self):

        self.model.summary()
