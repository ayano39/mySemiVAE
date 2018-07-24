import tensorflow as tf
import time
import gzip
import pickle
from generate_data import save_as_images_grid

'''
# Experiments on tf.equal
a = [1, 0, 0]
b = [0, 0, 0]
c = tf.equal(a, b)
d = tf.reduce_mean(tf.cast(c, tf.float32))
sess = tf.InteractiveSession()

print(sess.run([c, d]))
'''

'''
# To confirm the correctness of dataset sampling
f_0 = gzip.open("dataset/mnist_u1000_l0.pkl.gz", "rb")
f_1000 = gzip.open("dataset/mnist_u1000_l1000.pkl.gz", "rb")
f_10000 = gzip.open("dataset/mnist_u1000_l10000.pkl.gz", "rb")
data_0 = pickle.load(f_0, encoding="bytes")
data_1000 = pickle.load(f_1000, encoding="bytes")
data_10000 = pickle.load(f_10000, encoding="bytes")

print(data_0[0][1])
print(data_1000[0][1].shape)
print(data_10000[0][1].shape)

#save_as_images_grid("0", data_0[0][0][:100], 100, 10)
#print(data_0[0][1][:100])
save_as_images_grid("1000", data_1000[0][0][:100], 100, 10)
print(data_1000[0][1][:100])
save_as_images_grid("10000", data_10000[0][0][:100], 100, 10)
print(data_10000[0][1][:100])
'''

# To confirm the efficiency of reduce_sum, reduce_max, count_nonzero
# a = [0] * 10000
# b = tf.reduce_sum(a)
# c = tf.reduce_max(a)
# d = tf.count_nonzero(a)
#
# sess = tf.InteractiveSession()
#
# print(sess.run([b, c, d]))
# start_time = time.time()
#
# for i in range(10000):
#     sess.run(d)
# stop_time = time.time()
# print("Time for count_nonzero: {}".format((stop_time-start_time) / 10000))
#
#
# start_time = time.time()
# for i in range(10000):
#     sess.run(c)
# stop_time = time.time()
# print("Time for reduce_max: {}".format((stop_time-start_time) / 10000))
#
# start_time = time.time()
# for i in range(10000):
#     sess.run(b)
# stop_time = time.time()
# print("Time for reduce_sum: {}".format((stop_time-start_time) / 10000))
#
#
#

batch_size = 0
dim_x = 5
l_batch_size = 4

a = tf.random_normal([batch_size, dim_x])
hidden = tf.layers.dense(a, 10)[:l_batch_size, :]

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    val_a, val_hidden = sess.run([a, hidden])

    print(val_a, val_a.shape)
    print(val_hidden, val_hidden.shape)