import tensorflow as tf

trans_activation = {
    'sigmoid': tf.nn.sigmoid,
    'relu': tf.nn.relu
}


class VAE(object):
    def __init__(self, FLAGS):
        self.x = None
        self.y = None
        self.FLAGS = FLAGS
        self.l_batch_size = None
        self.activation = trans_activation[FLAGS.activation]
        self.z_mean = None
        self.z_var = None
        self.z_std = None
        self.z = None
        self.kld_loss = None
        self.recon_loss = None
        self.class_loss = None
        self.class_logits = None
        self.L = None
        self.train_op = None
        self.x_gen = None
        self.merged_summary = None
        self.prediction = None
        self.accuracy = None

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

    def build_discriminator(self):
        with tf.variable_scope("Discriminator"):
            hidden = self.z
            for i in range(self.FLAGS.num_class_layers):
                hidden = tf.layers.dense(hidden, self.FLAGS.dim_hid,
                                         activation=self.activation)
            self.class_logits = tf.layers.dense(hidden, 10, activation=None)
            self.prediction = tf.argmax(self.class_logits, axis=1)
            true_y = tf.argmax(self.y, axis=1)
            self.accuracy = tf.reduce_mean(
                tf.cast(
                    tf.equal(
                        self.prediction,
                        true_y
                    ), tf.float32
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

            if self.y is None:
                self.class_loss = tf.constant(0, dtype=tf.float32)
            else:
                self.class_loss = tf.nn.softmax_cross_entropy_with_logits(
                    logits=self.class_logits[:self.l_batch_size:, :],
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
        self.merged_summary = tf.summary.merge_all()

    def build_model(self, input_x, input_y, l_batch_size):
        self.x = input_x
        self.y = input_y
        self.l_batch_size = l_batch_size
        with tf.variable_scope("VAE"):
            self.build_encoder()
            self.build_decoder()
            if self.y is not None:
                self.build_discriminator()
            self.build_loss()

    def build_generate(self, z):
        self.z = z
        with tf.variable_scope("VAE", reuse=True):
            self.build_decoder()
            self.x_gen = tf.nn.sigmoid(self.x_gen)
