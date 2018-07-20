#import tensorflow as tf
import gzip
import pickle
from generate_data import save_as_images_grid
'''
a = [1, 0, 0]
b = [0, 0, 0]
c = tf.equal(a, b)
d = tf.reduce_mean(tf.cast(c, tf.float32))
sess = tf.InteractiveSession()

print(sess.run([c, d]))
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