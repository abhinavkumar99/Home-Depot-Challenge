import json
import tensorflow as tf
from datetime import datetime as dt
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
data = json.loads(open("train.json").read().strip())
train = []
for line in data:
    if list(line.values())[-1] == True:
        train.append(np.array([1,0]))
    else:
        train.append(np.array([0,1]))
print(len(train))
data = [[list(row.values())[3] - list(row.values())[2]] + list(row.values())[4:9] + [list(row.values())[10]] for row in data]
print(len(data))
#print(data)
data = np.array([np.array(i, dtype=np.int64) for i in data])
minn = np.min(data, axis=0)
maxx = np.max(data, axis=0)
x = np.array([i for i in range(len(train))])
np.random.shuffle(x)

data = data[x]
print(data.size)
train = np.array(train)[x]
data = (data-minn)/(maxx-minn)
learning_rate = .001
epochs = 450
batch_size = 815
train_len = 8965
def NN(x, weights, biases):
    layer_1 = tf.nn.relu(tf.add(tf.matmul(x, weights["1"]), biases["1"]))
    layer_2 = tf.nn.relu(tf.add(tf.matmul(layer_1, weights["2"]), biases["2"]))
    out = tf.add(tf.matmul(layer_2, weights["out"]), biases["out"])
    return out

weights = {
    "1": tf.Variable(tf.random_normal([7, 256])),
    "2": tf.Variable(tf.random_normal([256, 128])),
    "out": tf.Variable(tf.random_normal([128, 2]))
}
biases = {
    "1": tf.Variable(tf.random_normal([256])),
    "2": tf.Variable(tf.random_normal([128])),
    "out": tf.Variable(tf.random_normal([2]))
}
x_vals = tf.placeholder(tf.float32, [None, 7])
y_vals = tf.placeholder(tf.float32, [None, 2])
pred = NN(x_vals, weights, biases)
cost = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(logits = pred, labels = y_vals))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
is_correct = tf.equal(tf.argmax(pred, 1), tf.argmax(y_vals, 1))
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float64))
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    testx = data[train_len:]
    testy = train[train_len:]
    costs = 0
    accs = 0
    for epoch in range(epochs):
        
        for batch in range(train_len//batch_size):
            trainx = data[batch*batch_size: (batch + 1)*batch_size]
            trainy = train[batch*batch_size: (batch + 1)*batch_size]

            _, tcost, tacc = sess.run([optimizer, cost, accuracy], feed_dict={x_vals: trainx, y_vals: trainy})
            costs+=tcost
            accs+=tacc
        if epoch % 10 == 0:
            print(f"Train: {epoch}")
            print(costs/(10*(train_len//batch_size)))
            print(accs/(10*(train_len//batch_size)))
            costs = 0
            accs = 0
                
            tecost, teacc = sess.run([cost, accuracy], feed_dict={x_vals: testx, y_vals:testy})
            print("Test:")
            print(tecost)
            print(teacc)
            print("\n")
