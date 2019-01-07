'''
The baseline of this work is Value Iteration Network (VIN, proposed in NIPS 2016).
The novel architecture we proposed is DB-Net, which is implemented by "DB_Net(X, S1, S2, config, keep_drop)".
This file is the main function of the whole experiments.
Part of the following code refers to the original code of VIN available on github.
We mainly designed the function "DB_Net(X, S1, S2, config, keep_drop)" and made some revision to fit our dataset.
'''

import time
import numpy as np
import tensorflow as tf
from data import process_gridworld_data
from model import VIN, DB_CNN, DCNN
from utils import fmt_row
import matplotlib.pyplot as plt

# Data
tf.app.flags.DEFINE_string('input', './data/data_32.pkl', 'Path to data')
tf.app.flags.DEFINE_integer('imsize', 32, 'Size of input image')
# Parameters
tf.app.flags.DEFINE_float('lr', 0.001, 'Learning rate for RMSProp')
tf.app.flags.DEFINE_integer('epochs', 30, 'Maximum epochs to train for')
tf.app.flags.DEFINE_integer('k', 36, 'Number of value iterations')
tf.app.flags.DEFINE_integer('ch_i', 2, 'Channels in input layer')
tf.app.flags.DEFINE_integer('ch_h', 150, 'Channels in initial hidden layer')
tf.app.flags.DEFINE_integer('ch_q', 10, 'Channels in q layer (~actions)')
tf.app.flags.DEFINE_integer('batchsize', 32, 'Batch size')
tf.app.flags.DEFINE_integer(
    'statebatchsize', 10, 'Number of state inputs for each sample (real number, technically is k+1)')
tf.app.flags.DEFINE_integer('mode', 1, 'Untie weights of VI network')
tf.app.flags.DEFINE_float('belta', 0.0003, 'l2_loss')
#tf.app.flags.DEFINE_float('keep_drop', 0.8, 'dropout parameters')
# Misc.
tf.app.flags.DEFINE_integer('seed', 0, 'Random seed for numpy')
tf.app.flags.DEFINE_integer(
    'display_step', 1, 'Print summary output every n epochs')
tf.app.flags.DEFINE_boolean('log', True, 'Enable for tensorboard summary')
tf.app.flags.DEFINE_string('logdir', 'tmp/vintf/',
                           'Directory to store tensorboard summary')

config = tf.app.flags.FLAGS

np.random.seed(config.seed)

# symbolic input image tensor where typically first channel is image, second is the reward prior
X = tf.placeholder(tf.float32, name="X", shape=[
                   None, config.imsize, config.imsize, config.ch_i])
# symbolic input batches of vertical positions
S1 = tf.placeholder(tf.int32, name="S1", shape=[None, config.statebatchsize])
# symbolic input batches of horizontal positions
S2 = tf.placeholder(tf.int32, name="S2", shape=[None, config.statebatchsize])
y = tf.placeholder(tf.int32, name="y", shape=[None])
LR = tf.placeholder(tf.float32, name="LR", shape=())
keep_drop = tf.placeholder(tf.float32, name="keep_drop", shape=())

# Construct model (Value Iteration Network)
if (config.mode == 0):
    logits, nn = VIN(X, S1, S2, config)
elif (config.mode == 1):
    logits, nn, l2_loss = DB_CNN(X, S1, S2, config, keep_drop)
else:
    logits, nn = DCNN(X, S1, S2, config, keep_drop)

# Define loss and optimizer
y_ = tf.cast(y, tf.int64)
cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
    logits=logits, labels=y_, name='cross_entropy')
cross_entropy_mean = tf.reduce_mean(
    cross_entropy, name='cross_entropy_mean')  # + config.belta * l2_loss
tf.add_to_collection('losses', cross_entropy_mean)

cost = tf.add_n(tf.get_collection('losses'), name='total_loss')
optimizer = tf.train.RMSPropOptimizer(
    learning_rate=LR, epsilon=1e-6, centered=True).minimize(cost)

# Test model & calculate accuracy
cp = tf.cast(tf.argmax(nn, 1), tf.int32)
err = tf.reduce_mean(tf.cast
                     (tf.not_equal(cp, y), dtype=tf.float32))

# Initializing the variables
init = tf.global_variables_initializer()
saver = tf.train.Saver()

Xtrain, S1train, S2train, ytrain, Xtest, S1test, S2test, ytest = process_gridworld_data(
    input=config.input, imsize=config.imsize)
learning_rate = 0.003  # 0.001
Have_trained = 0
TrA = []
TeA = []
# Launch the graph
with tf.Session() as sess:
    if config.log:
        for var in tf.trainable_variables():
            tf.summary.histogram(var.op.name, var)
        summary_op = tf.summary.merge_all()
        summary_writer = tf.summary.FileWriter(config.logdir, sess.graph)
    sess.run(init)
    if Have_trained == True:
        model_file = tf.train.latest_checkpoint('ckpt/')
        saver.restore(sess, model_file)

    batch_size = config.batchsize
    print(fmt_row(10, ["Epoch", "Train Cost",
                       "Train Acc", "Epoch Time", "Test Acc"]))
    for epoch in range(int(config.epochs)):
        learning_rate = learning_rate * 0.95
        tstart = time.time()
        avg_err, avg_cost = 0.0, 0.0
        avg_err_L = []
        avg_cost_L = []
        num_batches = int(Xtrain.shape[0] / batch_size)
        # Loop over all batches
        for i in range(0, Xtrain.shape[0], batch_size):
            j = i + batch_size
            if j <= Xtrain.shape[0]:
                # Run optimization op (backprop) and cost op (to get loss value)
                fd = {X: Xtrain[i:j], S1: S1train[i:j], S2: S2train[i:j],
                      y: ytrain[i * config.statebatchsize:j * config.statebatchsize], LR: learning_rate, keep_drop: 0.5}
                _, e_, c_ = sess.run([optimizer, err, cost], feed_dict=fd)
                avg_err += e_
                avg_cost += c_
                avg_err_L.append(avg_err / num_batches)
                avg_cost_L.append(avg_cost / num_batches)
    # Display logs per epoch step
        if epoch % config.display_step == 0:
            elapsed = time.time() - tstart
            num_test_batch = int(Xtest.shape[0] / batch_size)
            acc = 0
            for k in range(0, Xtest.shape[0], batch_size):
                l = k + batch_size
                if l <= Xtest.shape[0]:
                    fd = {X: Xtest[k:l], S1: S1test[k:l], S2: S2test[k:l], y: ytest[k *
                                                                                    config.statebatchsize:l * config.statebatchsize], keep_drop: 1}
                    acc += sess.run(err, feed_dict=fd)
            acc = acc / num_test_batch
            print(fmt_row(10, [epoch, avg_cost / num_batches,
                               1 - avg_err / num_batches, elapsed, 1 - acc]))

        if config.log:
            summary = tf.Summary()
            summary.ParseFromString(sess.run(summary_op))
            summary.value.add(tag='Average error',
                              simple_value=float(avg_err / num_batches))
            summary.value.add(tag='Average cost',
                              simple_value=float(avg_cost / num_batches))
            summary_writer.add_summary(summary, epoch)
        saver.save(sess, "ckpt/", global_step=epoch)
    print("Finished training!")

    # Test model
    correct_prediction = tf.cast(tf.argmax(nn, 1), tf.int32)
    # Calculate accuracy
    accuracy = tf.reduce_mean(
        tf.cast(tf.not_equal(correct_prediction, y), dtype=tf.float32))
    acc = 0
    num_test_batch = int(Xtest.shape[0] / batch_size)
    for i in range(0, Xtest.shape[0], batch_size):
        j = i + batch_size
        if j <= Xtest.shape[0]:
            fd = {X: Xtest[i:j], S1: S1test[i:j], S2: S2test[i:j], y: ytest[i *
                                                                            config.statebatchsize:j * config.statebatchsize], keep_drop: 1}
            acc += sess.run(err, feed_dict=fd)
    acc = acc / num_test_batch
    print(f'Accuracy: {100 * (1 - acc)}%')
