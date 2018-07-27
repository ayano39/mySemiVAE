import tensorflow as tf

trans_activation = {
    'sigmoid': tf.nn.sigmoid,
    'relu': tf.nn.relu
}


class VAE(object):
    def __init__(self, FLAGS,
                 pure_supervised=False,
                 pure_unsupervised=False,
                 mute_class_loss=False
                 ):
        self.x = None
        self.y = None
        self.FLAGS = FLAGS
        self.pure_supervised = pure_supervised
        self.pure_unsupervised = pure_unsupervised
        self.mute_class_loss = mute_class_loss
        self.activation = trans_activation[FLAGS.activation]
        self.l_batch_size = tf.placeholder(tf.int32, shape=[])
        self.z_mean = None
        self.z_var = None
        self.z_std = None
        self.z = None
        self.x_gen = None
        self.x_gen_sigmoid = None
        self.class_logits = None
        self.prediction = None
        self.accuracy = None
        self.kld_loss = None
        self.recon_loss = None
        self.class_loss = None
        self.L = None
        self.train_op = None
        self.merged_summary = None

    def build_encoder(self):
        with tf.variable_scope("Encoder"):
            hidden = self.x
            for i in range(self.FLAGS.num_layers):
                hidden = tf.layers.dense(hidden, self.FLAGS.dim_hid,
                                         activation=self.activation)
            self.z_mean = tf.layers.dense(hidden, self.FLAGS.dim_z,
                                          activation=None)
            z_logvar = tf.layers.dense(hidden, self.FLAGS.dim_z,
                                       activation=None)
            self.z_var = tf.exp(z_logvar)
            self.z_std = tf.sqrt(self.z_var)
            eps = tf.random_normal(tf.shape(self.z_mean), mean=0.0, stddev=1.0)
            self.z = self.z_mean + self.z_std * eps

    def build_decoder(self):
        with tf.variable_scope("Decoder"):
            hidden = self.z
            for i in range(self.FLAGS.num_layers):
                hidden = tf.layers.dense(hidden, self.FLAGS.dim_hid,
                                         activation=self.activation)
            self.x_gen = tf.layers.dense(hidden, self.FLAGS.dim_x,
                                         activation=None)
            self.x_gen_sigmoid = tf.nn.sigmoid(self.x_gen)

    def build_discriminator(self):
        with tf.variable_scope("Discriminator"):
            hidden = self.z[:self.l_batch_size, :]
            for i in range(self.FLAGS.num_class_layers):
                hidden = tf.layers.dense(hidden, self.FLAGS.dim_hid,
                                         activation=self.activation)
            self.class_logits = tf.layers.dense(hidden, self.FLAGS.dim_y, activation=None)
            self.prediction = tf.argmax(self.class_logits, axis=1)
            true_y = tf.argmax(self.y, axis=1)
            self.accuracy = tf.reduce_mean(
                tf.cast(
                    tf.equal(
                        self.prediction,
                        true_y
                    ),
                    tf.float32
                )
            )

    def build_loss(self):
        with tf.variable_scope("Loss"):
            self.kld_loss = -0.5 * tf.reduce_sum(
                1 + tf.log(self.z_var) - tf.square(self.z_mean) - self.z_var,
                axis=1
            )

            self.recon_loss = tf.reduce_sum(
                tf.nn.sigmoid_cross_entropy_with_logits(
                    labels=self.x,
                    logits=self.x_gen
                ),
                axis=1
            )

            if self.pure_unsupervised or self.mute_class_loss:
                self.class_loss = tf.constant(0, dtype=tf.float32)
            else:
                self.class_loss = tf.nn.softmax_cross_entropy_with_logits(
                    logits=self.class_logits,
                    labels=self.y
                )

            self.L = tf.reduce_mean(self.kld_loss + self.recon_loss)\
                + tf.reduce_mean(tf.scalar_mul(
                    self.FLAGS.loss_lambda,
                    self.class_loss))

        with tf.variable_scope("Optimization"):
            global_step = tf.Variable(0, dtype=tf.float32)
            optimizer = tf.train.AdagradOptimizer(self.FLAGS.lr)
            self.train_op = optimizer.minimize(self.L, global_step=global_step)

        tf.summary.scalar('Loss', self.L)
        tf.summary.scalar('KLD Loss', tf.reduce_mean(self.kld_loss))
        tf.summary.scalar('Recon Loss', tf.reduce_mean(self.recon_loss))
        tf.summary.scalar('Class Loss', tf.reduce_mean(self.class_loss))
        tf.summary.scalar('Accuracy', tf.reduce_mean(self.accuracy))
        self.merged_summary = tf.summary.merge_all()

    def build_model(self, labelled_x, labelled_y, unlabelled_x):
        if self.pure_supervised:
            self.x = labelled_x
        elif self.pure_unsupervised:
            self.x = unlabelled_x
        else:
            self.x = tf.concat([labelled_x, unlabelled_x], axis=0)
        self.y = labelled_y

        with tf.variable_scope("VAE"):
            self.build_encoder()
            self.build_decoder()
            self.build_discriminator()
            self.build_loss()
