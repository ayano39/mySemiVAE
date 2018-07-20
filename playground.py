import tensorflow as tf


a = [1, 0, 0]
b = [0, 0, 0]
c = tf.equal(a, b)
d = tf.reduce_mean(tf.cast(c, tf.float32))
sess = tf.InteractiveSession()

print(sess.run([c, d]))