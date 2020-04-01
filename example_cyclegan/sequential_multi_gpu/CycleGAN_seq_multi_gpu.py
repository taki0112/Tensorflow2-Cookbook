from utils import *
import time
from tensorflow.python.data.experimental import prefetch_to_device, shuffle_and_repeat, map_and_batch, AUTOTUNE

from glob import glob
from tqdm import tqdm
from networks import *
# automatic_gpu_usage()

# https://blog.paperspace.com/tensorflow-2-0-in-practice/
class CycleGAN():
    def __init__(self, args):
        super(CycleGAN, self).__init__()

        self.model_name = 'CycleGAN'
        self.phase = args.phase
        self.checkpoint_dir = args.checkpoint_dir
        self.result_dir = args.result_dir
        self.log_dir = args.log_dir
        self.sample_dir = args.sample_dir
        self.dataset_name = args.dataset
        self.augment_flag = args.augment_flag

        self.decay_flag = args.decay_flag
        self.decay_iter = args.decay_iter
        self.iteration = args.iteration
        self.gan_type = args.gan_type

        self.batch_size = args.batch_size
        self.print_freq = args.print_freq
        self.save_freq = args.save_freq

        self.img_height = args.img_size
        self.img_width = args.img_size
        self.img_ch = args.img_ch

        self.init_lr = args.lr
        self.ch = args.ch

        """ Weight """
        self.adv_weight = args.adv_weight
        self.cyc_weight = args.cyc_weight
        self.identity_weight = args.identity_weight

        """ Generator """
        self.n_res = args.n_res

        """ Discriminator """
        self.n_dis = args.n_dis
        self.sn = args.sn

        self.sample_dir = os.path.join(args.sample_dir, self.model_dir)
        check_folder(self.sample_dir)

        self.checkpoint_dir = os.path.join(args.checkpoint_dir, self.model_dir)
        check_folder(self.checkpoint_dir)

        self.log_dir = os.path.join(args.log_dir, self.model_dir)
        check_folder(self.log_dir)

        self.dataset_path = os.path.join('./dataset', self.dataset_name)

        print("##### Information #####")
        print("# gan type : ", self.gan_type)
        print("# dataset : ", self.dataset_name)
        print("# batch_size : ", self.batch_size)
        print("# max iteration : ", self.iteration)
        print("# spectral normalization : ", self.sn)

        print()

        print("##### Generator #####")
        print("# residual blocks : ", self.n_res)

        print()

        print("##### Discriminator #####")
        print("# Discriminator layer : ", self.n_dis)

    ##################################################################################
    # Model
    ##################################################################################
    def build_model(self):
        if self.phase == 'train':
            # devices = ['/device:GPU:{}'.format(i) for i in range(NUM_GPUS)]
            self.strategy = tf.distribute.MirroredStrategy()
            self.NUM_GPUS = self.strategy.num_replicas_in_sync

            self.iteration = self.iteration // self.NUM_GPUS
            self.decay_iter = self.decay_iter // self.NUM_GPUS

            with self.strategy.scope():
                """ Input Image"""
                img_class = Image_data(self.img_height, self.img_width, self.img_ch, self.dataset_path,
                                       self.augment_flag)
                img_class.preprocess()
                dataset_num = max(len(img_class.train_A_dataset), len(img_class.train_B_dataset))

                print("Dataset number : ", dataset_num)

                img_slice_A = tf.data.Dataset.from_tensor_slices(img_class.train_A_dataset)
                img_slice_B = tf.data.Dataset.from_tensor_slices(img_class.train_B_dataset)

                gpu_device = '/gpu:0'
                # gpu_device = devices[0]
                img_slice_A = img_slice_A. \
                    apply(shuffle_and_repeat(dataset_num)). \
                    apply(
                    map_and_batch(img_class.image_processing, self.batch_size * self.NUM_GPUS, num_parallel_calls=AUTOTUNE,
                                  drop_remainder=True)). \
                    apply(prefetch_to_device(gpu_device, AUTOTUNE))

                img_slice_B = img_slice_B. \
                    apply(shuffle_and_repeat(dataset_num)). \
                    apply(
                    map_and_batch(img_class.image_processing, self.batch_size * self.NUM_GPUS, num_parallel_calls=AUTOTUNE,
                                  drop_remainder=True)). \
                    apply(prefetch_to_device(gpu_device, AUTOTUNE))

                img_slice_A = self.strategy.experimental_distribute_dataset(img_slice_A)
                img_slice_B = self.strategy.experimental_distribute_dataset(img_slice_B)

                self.dataset_A_iter = iter(img_slice_A)
                self.dataset_B_iter = iter(img_slice_B)


                """ Network """
                self.source_generator = Generator(self.ch, self.n_res, name='source_generator')
                self.target_generator = Generator(self.ch, self.n_res, name='target_generator')
                self.source_discriminator = Discriminator(self.ch, self.n_dis, self.sn, name='source_discriminator')
                self.target_discriminator = Discriminator(self.ch, self.n_dis, self.sn, name='target_discriminator')

                """ Optimizer """
                self.g_optimizer = tf.keras.optimizers.Adam(learning_rate=self.init_lr, beta_1=0.5, beta_2=0.999, epsilon=1e-08)
                self.d_optimizer = tf.keras.optimizers.Adam(learning_rate=self.init_lr, beta_1=0.5, beta_2=0.999, epsilon=1e-08)

                """ Summary """
                # mean metric
                self.g_adv_loss_metric = tf.keras.metrics.Mean('g_adv_loss', dtype=tf.float32)
                self.g_cyc_loss_metric = tf.keras.metrics.Mean('g_cyc_loss', dtype=tf.float32)
                self.g_identity_loss_metric = tf.keras.metrics.Mean('g_identity_loss', dtype=tf.float32)
                self.g_loss_metric = tf.keras.metrics.Mean('g_loss', dtype=tf.float32)

                self.d_adv_loss_metric = tf.keras.metrics.Mean('d_adv_loss', dtype=tf.float32)
                self.d_loss_metric = tf.keras.metrics.Mean('d_loss', dtype=tf.float32)


                input_shape = [self.img_height, self.img_width, self.img_ch]
                self.source_generator.build_summary(input_shape)
                self.source_discriminator.build_summary(input_shape)
                self.target_generator.build_summary(input_shape)
                self.target_discriminator.build_summary(input_shape)

                """ Count parameters """
                params = self.source_generator.count_parameter() + self.target_generator.count_parameter() \
                         + self.source_discriminator.count_parameter() + self.target_discriminator.count_parameter()

                print("Total network parameters : ", format(params, ','))
                print("Total gpu numbers : ", self.NUM_GPUS)

                """ Checkpoint """
                self.ckpt = tf.train.Checkpoint(source_generator=self.source_generator,
                                                target_generator=self.target_generator,
                                                source_discriminator=self.source_discriminator,
                                                target_discriminator=self.target_discriminator,
                                                g_optimizer=self.g_optimizer, d_optimizer=self.d_optimizer)
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
            self.source_generator = Generator(self.ch, self.n_res, name='source_generator')
            self.target_generator = Generator(self.ch, self.n_res, name='target_generator')
            self.source_discriminator = Discriminator(self.ch, self.n_dis, self.sn, name='source_discriminator')
            self.target_discriminator = Discriminator(self.ch, self.n_dis, self.sn, name='target_discriminator')

            input_shape = [self.img_height, self.img_width, self.img_ch]
            self.source_generator.build_summary(input_shape)
            self.source_discriminator.build_summary(input_shape)
            self.target_generator.build_summary(input_shape)
            self.target_discriminator.build_summary(input_shape)

            """ Count parameters """
            params = self.source_generator.count_parameter() + self.target_generator.count_parameter() \
                     + self.source_discriminator.count_parameter() + self.target_discriminator.count_parameter()

            print("Total network parameters : ", format(params, ','))

            """ Checkpoint """
            self.ckpt = tf.train.Checkpoint(source_generator=self.source_generator,
                                            target_generator=self.target_generator,
                                            source_discriminator=self.source_discriminator,
                                            target_discriminator=self.target_discriminator)
            self.manager = tf.train.CheckpointManager(self.ckpt, self.checkpoint_dir, max_to_keep=2)

            if self.manager.latest_checkpoint:
                self.ckpt.restore(self.manager.latest_checkpoint).expect_partial()
                print('Latest checkpoint restored!!')
            else:
                print('Not restoring from saved checkpoint')

    def g_train_step(self, real_a, real_b):
        with tf.GradientTape() as g_tape:
            x_ab = self.target_generator(real_a)
            x_ba = self.source_generator(real_b)

            x_aa = self.source_generator(real_a)
            x_bb = self.target_generator(real_b)

            x_aba = self.source_generator(x_ab)
            x_bab = self.target_generator(x_ba)

            fake_a_logit = self.source_discriminator(x_ba)
            fake_b_logit = self.target_discriminator(x_ab)

            g_adv_loss = self.adv_weight * (generator_loss(self.gan_type, fake_a_logit) + generator_loss(self.gan_type, fake_b_logit))
            g_cyc_loss = self.cyc_weight * (L1_loss(x_aba, real_a) + L1_loss(x_bab, real_b))
            g_identity_loss = self.identity_weight * (L1_loss(x_aa, real_a) + L1_loss(x_bb, real_b))

            regular_loss = regularization_loss(self.source_generator) + regularization_loss(self.target_generator)

            g_loss = g_adv_loss + g_cyc_loss + g_identity_loss + regular_loss

        g_train_variable = self.source_generator.trainable_variables + self.target_generator.trainable_variables
        g_gradient = g_tape.gradient(g_loss, g_train_variable)
        self.g_optimizer.apply_gradients(zip(g_gradient, g_train_variable))

        # self.g_adv_loss_metric(self.g_adv_loss)
        # self.g_cyc_loss_metric(self.g_cyc_loss)
        # self.g_identity_loss_metric(self.g_identity_loss)
        # self.g_loss_metric(self.g_loss)

        return x_ab, x_ba, g_loss, g_adv_loss, g_cyc_loss, g_identity_loss

    def d_train_step(self, real_a, real_b):
        with tf.GradientTape() as d_tape:
            x_ab = self.target_generator(real_a)
            x_ba = self.source_generator(real_b)

            real_a_logit = self.source_discriminator(real_a)
            real_b_logit = self.target_discriminator(real_b)

            fake_a_logit = self.source_discriminator(x_ba)
            fake_b_logit = self.target_discriminator(x_ab)

            d_adv_loss = self.adv_weight * (discriminator_loss(self.gan_type, real_a_logit, fake_a_logit) + discriminator_loss(self.gan_type, real_b_logit, fake_b_logit))

            regular_loss = regularization_loss(self.source_discriminator) + regularization_loss(self.target_discriminator)
            d_loss = d_adv_loss + regular_loss

        d_train_variable = self.source_discriminator.trainable_variables + self.target_discriminator.trainable_variables
        d_gradient = d_tape.gradient(d_loss, d_train_variable)
        self.d_optimizer.apply_gradients(zip(d_gradient, d_train_variable))

        # self.d_adv_loss_metric(self.d_adv_loss)
        # self.d_loss_metric(self.d_loss)

        return d_loss, d_adv_loss

    def train_step(self, real_a, real_b):
        with tf.GradientTape() as d_tape, tf.GradientTape() as g_tape:
            x_ab = self.target_generator(real_a)
            x_ba = self.source_generator(real_b)

            x_aa = self.source_generator(real_a)
            x_bb = self.target_generator(real_b)

            x_aba = self.source_generator(x_ab)
            x_bab = self.target_generator(x_ba)

            real_a_logit = self.source_discriminator(real_a)
            real_b_logit = self.target_discriminator(real_b)

            fake_a_logit = self.source_discriminator(x_ba)
            fake_b_logit = self.target_discriminator(x_ab)

            d_adv_loss = self.adv_weight * (discriminator_loss(self.gan_type, real_a_logit, fake_a_logit) + discriminator_loss(self.gan_type, real_b_logit, fake_b_logit))

            regular_loss = regularization_loss(self.source_discriminator) + regularization_loss(self.target_discriminator)
            d_loss = d_adv_loss + regular_loss
            # [0.11]
            # [0.22]
            g_adv_loss = self.adv_weight * (generator_loss(self.gan_type, fake_a_logit) + generator_loss(self.gan_type, fake_b_logit))
            g_cyc_loss = self.cyc_weight * (L1_loss(x_aba, real_a) + L1_loss(x_bab, real_b))
            g_identity_loss = self.identity_weight * (L1_loss(x_aa, real_a) + L1_loss(x_bb, real_b))

            regular_loss = regularization_loss(self.source_generator) + regularization_loss(self.target_generator)

            g_loss = g_adv_loss + g_cyc_loss + g_identity_loss + regular_loss

        d_train_variable = self.source_discriminator.trainable_variables + self.target_discriminator.trainable_variables
        d_gradient = d_tape.gradient(d_loss, d_train_variable)
        self.d_optimizer.apply_gradients(zip(d_gradient, d_train_variable))

        g_train_variable = self.source_generator.trainable_variables + self.target_generator.trainable_variables
        g_gradient = g_tape.gradient(g_loss, g_train_variable)
        self.g_optimizer.apply_gradients(zip(g_gradient, g_train_variable))

        # self.d_adv_loss_metric(self.d_adv_loss)
        # self.d_loss_metric(self.d_loss)

        # self.g_adv_loss_metric(self.g_adv_loss)
        # self.g_cyc_loss_metric(self.g_cyc_loss)
        # self.g_identity_loss_metric(self.g_identity_loss)
        # self.g_loss_metric(self.g_loss)

        return x_ab, x_ba, d_loss, d_adv_loss, g_loss, g_adv_loss, g_cyc_loss, g_identity_loss


    def distributed_train(self, real_a, real_b):
        with self.strategy.scope():
            """
            d_loss, d_adv_loss = self.strategy.experimental_run_v2(self.d_train_step, args=(real_a, real_b, ))
            x_ab, x_ba, g_loss, g_adv_loss, g_cyc_loss, g_identity_loss = self.strategy.experimental_run_v2(self.g_train_step, args=(real_a, real_b))
            
            self.d_loss = self.strategy.reduce(tf.distribute.ReduceOp.MEAN, d_loss, axis=None)
            self.d_adv_loss = self.strategy.reduce(tf.distribute.ReduceOp.MEAN, d_adv_loss, axis=None)
            
            self.g_loss = self.strategy.reduce(tf.distribute.ReduceOp.MEAN, g_loss, axis=None)
            self.g_adv_loss = self.strategy.reduce(tf.distribute.ReduceOp.MEAN, g_adv_loss, axis=None)
            self.g_cyc_loss = self.strategy.reduce(tf.distribute.ReduceOp.MEAN, g_cyc_loss, axis=None)
            self.g_identity_loss = self.strategy.reduce(tf.distribute.ReduceOp.MEAN, g_identity_loss, axis=None)


            self.d_adv_loss_metric(self.d_adv_loss)
            self.d_loss_metric(self.d_loss)
            
            self.g_adv_loss_metric(self.g_adv_loss)
            self.g_cyc_loss_metric(self.g_cyc_loss)
            self.g_identity_loss_metric(self.g_identity_loss)
            self.g_loss_metric(self.g_loss)
            """
            x_ab, x_ba, d_loss, d_adv_loss, g_loss, g_adv_loss, g_cyc_loss, g_identity_loss = self.strategy.experimental_run_v2(self.train_step, args=(real_a, real_b))

            self.d_loss = self.strategy.reduce(tf.distribute.ReduceOp.MEAN, d_loss, axis=None)
            self.d_adv_loss = self.strategy.reduce(tf.distribute.ReduceOp.MEAN, d_adv_loss, axis=None)

            self.g_loss = self.strategy.reduce(tf.distribute.ReduceOp.MEAN, g_loss, axis=None)
            self.g_adv_loss = self.strategy.reduce(tf.distribute.ReduceOp.MEAN, g_adv_loss, axis=None)
            self.g_cyc_loss = self.strategy.reduce(tf.distribute.ReduceOp.MEAN, g_cyc_loss, axis=None)
            self.g_identity_loss = self.strategy.reduce(tf.distribute.ReduceOp.MEAN, g_identity_loss, axis=None)

            self.d_adv_loss_metric(self.d_adv_loss)
            self.d_loss_metric(self.d_loss)

            self.g_adv_loss_metric(self.g_adv_loss)
            self.g_cyc_loss_metric(self.g_cyc_loss)
            self.g_identity_loss_metric(self.g_identity_loss)
            self.g_loss_metric(self.g_loss)

            return x_ab, x_ba, self.d_loss, self.g_loss

    def train(self):
        start_time = time.time()

        # setup tensorboard
        train_summary_writer = tf.summary.create_file_writer(self.log_dir)

        for idx in range(self.start_iteration, self.iteration):
            current_step = idx
            if self.decay_flag:
                # total_step = self.iteration
                decay_start_step = self.decay_iter

                # if current_step >= decay_start_step :
                # lr = self.init_lr * (total_step - current_step) / (total_step - decay_start_step)
                if idx > 0 and (idx % decay_start_step) == 0:
                    lr = self.init_lr * pow(0.5, idx // decay_start_step)
                    self.g_optimizer.learning_rate = lr
                    self.d_optimizer.learning_rate = lr

            real_a = next(self.dataset_A_iter)
            real_b = next(self.dataset_B_iter)

            # update discriminator
            # d_loss = self.d_train_step(real_a, real_b)

            # update generator
            # fake_b, fake_a, g_loss = self.g_train_step(real_a, real_b)

            # update
            fake_b, fake_a, d_loss, g_loss = self.distributed_train(real_a, real_b)

            # save to tensorboard
            with train_summary_writer.as_default():
                """
                tf.summary.scalar('g_adv_loss', self.g_adv_loss_metric.result(), step=idx)
                tf.summary.scalar('g_cyc_loss', self.g_cyc_loss_metric.result(), step=idx)
                tf.summary.scalar('g_identity_loss', self.g_identity_loss_metric.result(), step=idx)
                tf.summary.scalar('g_loss', self.g_loss_metric.result(), step=idx)

                tf.summary.scalar('d_adv_loss', self.d_adv_loss_metric.result(), step=idx)
                tf.summary.scalar('d_loss', self.d_loss_metric.result(), step=idx)
                """
                tf.summary.scalar('g_adv_loss', self.g_adv_loss, step=idx)
                tf.summary.scalar('g_cyc_loss', self.g_cyc_loss, step=idx)
                tf.summary.scalar('g_identity_loss', self.g_identity_loss, step=idx)
                tf.summary.scalar('g_loss', self.g_loss, step=idx)

                tf.summary.scalar('d_adv_loss', self.d_adv_loss, step=idx)
                tf.summary.scalar('d_loss', self.d_loss, step=idx)

            # save every self.save_freq
            if np.mod(idx + 1, self.save_freq) == 0:
                self.manager.save(checkpoint_number=idx + 1)

            # save every self.print_freq
            if np.mod(idx + 1, self.print_freq) == 0:
                if self.NUM_GPUS > 1 :
                    real_a = real_a.values[0]
                    real_b = real_b.values[0]
                    fake_a = fake_a.values[0]
                    fake_b = fake_b.values[0]

                real_a = np.expand_dims(real_a[0], axis=0)
                real_b = np.expand_dims(real_b[0], axis=0)
                real_imgs = np.concatenate([real_a, real_b], axis=0)

                fake_a = np.expand_dims(fake_a[0], axis=0)
                fake_b = np.expand_dims(fake_b[0], axis=0)
                fake_imgs = np.concatenate([fake_b, fake_a], axis=0)

                merge_real_img = np.expand_dims(return_images(real_imgs, [2, 1]), axis=0)
                merge_fake_img = np.expand_dims(return_images(fake_imgs, [2, 1]), axis=0)

                merge_img = np.concatenate([merge_real_img, merge_fake_img], axis=0)

                save_images(merge_img, [1, 2],
                            './{}/merge_{:07d}.jpg'.format(self.sample_dir, idx + 1))

            print("iter: [%6d/%6d] time: %4.4f d_loss: %.8f, g_loss: %.8f" % (
            idx, self.iteration, time.time() - start_time, d_loss, g_loss))

        # save model for final step
        self.manager.save(checkpoint_number=self.iteration)

    @property
    def model_dir(self):

        if self.sn:
            sn = '_sn'
        else:
            sn = ''

        return "{}_{}_{}_{}adv_{}cyc_{}identity{}".format(self.model_name, self.dataset_name, self.gan_type,
                                                          self.adv_weight, self.cyc_weight, self.identity_weight,
                                                          sn)

    def test(self):
        test_A_dataset = glob('./dataset/{}/{}/*.jpg'.format(self.dataset_name, 'testA')) + glob(
            './dataset/{}/{}/*.png'.format(self.dataset_name, 'testA'))
        result_dir = os.path.join(self.result_dir, self.model_dir, 'testA')
        check_folder(result_dir)

        # write html for visual comparison
        index_path = os.path.join(result_dir, 'index.html')
        index = open(index_path, 'w')
        index.write("<html><body><table><tr>")
        index.write("<th>name</th><th>input</th><th>output</th></tr>")

        for sample_file in tqdm(test_A_dataset):
            sample_image = load_test_image(sample_file, self.img_width, self.img_height, self.img_ch)
            image_path = os.path.join(result_dir, '{}'.format(os.path.basename(sample_file)))

            fake_img = self.target_generator(sample_image, training=False)
            save_images(fake_img, [1, 1], image_path)

            index.write("<td>%s</td>" % os.path.basename(image_path))
            index.write(
                "<td><img src='%s' width='%d' height='%d'></td>" % (sample_file if os.path.isabs(sample_file) else (
                        '../../..' + os.path.sep + sample_file), self.img_width, self.img_height))
            index.write(
                "<td><img src='%s' width='%d' height='%d'></td>" % (image_path if os.path.isabs(image_path) else (
                        '../../..' + os.path.sep + image_path), self.img_width, self.img_height))
            index.write("</tr>")
        index.close()

        ######################################################################################
        test_B_dataset = glob('./dataset/{}/{}/*.jpg'.format(self.dataset_name, 'testB')) + glob(
            './dataset/{}/{}/*.png'.format(self.dataset_name, 'testB'))
        result_dir = os.path.join(self.result_dir, self.model_dir, 'testB')
        check_folder(result_dir)

        # write html for visual comparison
        index_path = os.path.join(result_dir, 'index.html')
        index = open(index_path, 'w')
        index.write("<html><body><table><tr>")
        index.write("<th>name</th><th>input</th><th>output</th></tr>")

        for sample_file in tqdm(test_B_dataset):
            sample_image = load_test_image(sample_file, self.img_width, self.img_height, self.img_ch)
            image_path = os.path.join(result_dir, '{}'.format(os.path.basename(sample_file)))

            fake_img = self.target_generator(sample_image, training=False)
            save_images(fake_img, [1, 1], image_path)

            index.write("<td>%s</td>" % os.path.basename(image_path))
            index.write(
                "<td><img src='%s' width='%d' height='%d'></td>" % (sample_file if os.path.isabs(sample_file) else (
                        '../../..' + os.path.sep + sample_file), self.img_width, self.img_height))
            index.write(
                "<td><img src='%s' width='%d' height='%d'></td>" % (image_path if os.path.isabs(image_path) else (
                        '../../..' + os.path.sep + image_path), self.img_width, self.img_height))
            index.write("</tr>")
        index.close()