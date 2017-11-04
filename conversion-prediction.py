import json
import tensorflow as tf
from datetime import datetime as dt
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
data = json.loads(open("train.json").read().strip())
data = [list(row.values())[2:] for row in data]

#print(data)
data = np.array([np.array(i, dtype=np.int64) for i in data])
minn = np.min(data, axis=0)
maxx = np.max(data, axis=0)

data = (data-minn)/(maxx-minn)
print(data)
def NN(x, weights, biases):
    layer_1 = tf.nn.relu(tf.add(tf.matmul(x, weights["1"]), biases["1"]))
    layer_2 = tf.nn.relu(tf.add(tf.matmul(layer_1, weights["2"]), biases["2"]))
    out = tf.add(tf.matmul(layer_2, weights["out"]), biases["out"])
    return out
weights = {
    "1": tf.get_variable(tf.random_normal([9, 256])),
    "2": tf.get_variable(tf.random_normal([256, 128])),
    "out": tf.get_variable(tf.random_normal([128, 2]))
}
biases = {
    "1": tf.get_variable(tf.random_normal([256])),
    "2": tf.get_variable(tf.random_normal([128])),
    "out": tf.get_variable(tf.random_normal([2]))
}
x_vals = tf.placeholder(tf.float64, [None, 9])
y_vals = tf.placeholder(tf.float64, [None, 2])
pred = NN(x_vals, weights, biases)
cost = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(logits = pred, labels = y_vals))
is_correct = tf.equal(tf.argmax(pred, 1), tf.argmax(y_vals, 1))
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
