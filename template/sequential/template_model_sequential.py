from utils import *
from ops_sequential import *
import time
from tensorflow.python.data.experimental import prefetch_to_device, shuffle_and_repeat, map_and_batch, AUTOTUNE

from glob import glob
from tqdm import tqdm

automatic_gpu_usage() # for efficient gpu use

class SubNetwork(tf.keras.Model):
    def __init__(self, channels, name):
        super(SubNetwork, self).__init__(name=name)
        self.channels = channels

        self.model = self.architecture()

    def architecture(self):
        # Implement sub network architecture

        # example
        model = []

        model += [Conv(self.channels, kernel=7, stride=1, pad=3, pad_type='reflect', use_bias=False, sn=True, name='conv')]
        model += [BatchNorm(name='batch_norm')]
        model += [Relu()]

        model += [Global_Avg_Pooling()]
        model += [FullyConnected(units=10, sn=False, name='fc')]

        model = Sequential(model, name='d_encoder')

        return model

    def call(self, x_init, training=None, mask=None):

        x = self.model(x_init, training=training)

        return x

    def build_summary(self, input_shape, detail_summary=False):
        x_init = tf.keras.layers.Input(input_shape, name='sub_network_input')

        x = self.model(x_init)

        if detail_summary:
            self.model.summary()

        self.build_model = tf.keras.Model(x_init, x, name=self.name)
        self.build_model.summary()

    def count_parameter(self):
        x = self.model.count_params()
        return x


class Network():
    def __init__(self, args):
        super(Network, self).__init__()

        self.model_name = 'Network'
        self.phase = args.phase
        self.checkpoint_dir = args.checkpoint_dir
        self.result_dir = args.result_dir
        self.log_dir = args.log_dir
        self.sample_dir = args.sample_dir

        self.save_freq = args.save_freq

        # Set parameter
        self.dataset_name = args.dataset
        self.augment_flag = args.augment_flag
        self.batch_size = args.batch_size
        self.iteration = args.iteration

        self.img_height = args.img_height
        self.img_width = args.img_width
        self.img_ch = args.img_ch

        self.init_lr = args.lr

        self.sample_dir = os.path.join(args.sample_dir, self.model_dir)
        check_folder(self.sample_dir)

        self.checkpoint_dir = os.path.join(args.checkpoint_dir, self.model_dir)
        check_folder(self.checkpoint_dir)

        self.log_dir = os.path.join(args.log_dir, self.model_dir)
        check_folder(self.log_dir)

        self.dataset_path = os.path.join('./dataset', self.dataset_name)

    ##################################################################################
    # Model
    ##################################################################################
    def build_model(self):
        if self.phase == 'train':
            """ Input Image"""
            img_class = Image_data(self.img_height, self.img_width, self.img_ch, self.dataset_path, self.augment_flag)
            img_class.preprocess()
            dataset_num = len(img_class.dataset)

            print("Dataset number : ", dataset_num)

            img_slice = tf.data.Dataset.from_tensor_slices(img_class.dataset)

            gpu_device = '/gpu:0'
            img_slice = img_slice. \
                apply(shuffle_and_repeat(dataset_num)). \
                apply(map_and_batch(img_class.image_processing, self.batch_size, num_parallel_batches=AUTOTUNE,
                                    drop_remainder=True)). \
                apply(prefetch_to_device(gpu_device, AUTOTUNE))

            self.dataset_iter = iter(img_slice)

            """ Network """
            self.classifier = SubNetwork(channels=64, name='classifier')

            """ Optimizer """
            self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.init_lr, beta_1=0.5, beta_2=0.999,
                                                      epsilon=1e-08)

            """ Summary """
            # mean metric
            self.loss_metric = tf.keras.metrics.Mean('loss',
                                                     dtype=tf.float32)  # In tensorboard, make a loss to smooth graph

            # print summary
            input_shape = [self.img_height, self.img_width, self.img_ch]
            self.classifier.build_summary(input_shape)

            """ Count parameters """
            params = self.classifier.count_parameter()
            print("Total network parameters : ", format(params, ','))

            """ Checkpoint """
            self.ckpt = tf.train.Checkpoint(classifier=self.classifier, optimizer=self.optimizer)
            self.manager = tf.train.CheckpointManager(self.ckpt, self.checkpoint_dir, max_to_keep=2)
            self.start_iteration = 0

            if self.manager.latest_checkpoint:
                self.ckpt.restore(self.manager.latest_checkpoint)
                self.start_iteration = int(self.manager.latest_checkpoint.split('-')[-1])
                print('Latest checkpoint restored!!')
                print('start iteration : ', self.start_iteration)
            else:
                print('Not restoring from saved checkpoint')

        else:
            """ Test """

            """ Network """
            self.classifier = SubNetwork(channels=64, name='classifier')

            """ Summary """
            input_shape = [self.img_height, self.img_width, self.img_ch]
            self.classifier.build_summary(input_shape)

            """ Count parameters """
            params = self.classifier.count_parameter()
            print("Total network parameters : ", format(params, ','))

            """ Checkpoint """
            self.ckpt = tf.train.Checkpoint(classifier=self.classifier)
            self.manager = tf.train.CheckpointManager(self.ckpt, self.checkpoint_dir, max_to_keep=2)

            if self.manager.latest_checkpoint:
                self.ckpt.restore(self.manager.latest_checkpoint).expect_partial()
                print('Latest checkpoint restored!!')
            else:
                print('Not restoring from saved checkpoint')

    def train_step(self, real_img):
        with tf.GradientTape() as tape:
            logit = self.classifier(real_img)

            # calculate loss
            """

            if classification
            loss = cross_entroy_loss(logit, label)

            """

            your_loss = 0
            self.loss = your_loss + regularization_loss(self.classifier)

        train_variable = self.classifier.trainable_variables
        gradient = tape.gradient(self.loss, train_variable)
        self.optimizer.apply_gradients(zip(gradient, train_variable))

        self.loss_metric(self.loss)

        return self.loss

    def train(self):

        start_time = time.time()

        # setup tensorboards
        train_summary_writer = tf.summary.create_file_writer(self.log_dir)

        for idx in range(self.start_iteration, self.iteration):

            real_img = next(self.dataset_iter)

            # update network
            loss = self.train_step(real_img)

            # save to tensorboard
            with train_summary_writer.as_default():
                tf.summary.scalar('loss', self.loss, step=idx)
                tf.summary.scalar('loss_mean', self.loss_metric.result(), step=idx)

            # save every self.save_freq
            if np.mod(idx + 1, self.save_freq) == 0:
                self.manager.save(checkpoint_number=idx + 1)

            print("iter: [%6d/%6d] time: %4.4f loss: %.8f" % (
                idx, self.iteration, time.time() - start_time, loss))

        # save model for final step
        self.manager.save(checkpoint_number=self.iteration)

    @property
    def model_dir(self):

        return "{}_{}".format(self.model_name, self.dataset_name)

    def test(self):
        test_dataset = glob('./dataset/{}/{}/*.jpg'.format(self.dataset_name, 'test_img_folder')) + glob(
            './dataset/{}/{}/*.png'.format(self.dataset_name, 'test_img_folder'))
        self.result_dir = os.path.join(self.result_dir, self.model_dir)
        check_folder(self.result_dir)

        for sample_file in tqdm(test_dataset):
            sample_image = load_test_image(sample_file, self.img_width, self.img_height, self.img_ch)
            logit = self.classifier(sample_image, training=False)

            ### ...