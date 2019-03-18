import os
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"


"""
error: /proc/driver/nvidia/version does not exist

Setting the CUDA_VISIBLE_DEVICES variable to 0,1 
in python: os.environ["CUDA_VISIBLE_DEVICES"] = "0"

"""


##

import tensorflow as tf
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))


import tensorflow as tf
with tf.device('/gpu:0'):
    a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name='a')
    b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2], name='b')
    c = tf.matmul(a, b)

with tf.Session() as sess:
    print (sess.run(c))