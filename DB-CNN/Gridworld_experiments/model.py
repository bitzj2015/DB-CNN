import numpy as np
import tensorflow as tf
from utils import conv2d_flipkernel, conv2d_flipkernel_d

'''
The baseline of this work is Value Iteration Network (VIN, proposed in NIPS 2016) and the novel architecture we porposed is DB-Net.
This file provides
Part of the following code refers to the original code of VIN available on github.
'''


def VIN(X, S1, S2, config):
    keep_drop = 0.5
    k = config.k    # Number of value iterations performed
    ch_i = config.ch_i  # Channels in input layer
    ch_h = config.ch_h  # Channels in initial hidden layer
    ch_q = config.ch_q  # Channels in q layer (~actions)
    state_batch_size = config.statebatchsize  # k+1 state inputs for each channel

    bias = tf.Variable(np.random.randn(1, 1, 1, ch_h) * 0.01, dtype=tf.float32)
    # weights from inputs to q layer (~reward in Bellman equation)
    w0 = tf.Variable(np.random.randn(3, 3, ch_i, ch_h)
                     * 0.01, dtype=tf.float32)
    w1 = tf.Variable(np.random.randn(1, 1, ch_h, 1) * 0.01, dtype=tf.float32)
    w = tf.Variable(np.random.randn(3, 3, 1, ch_q) * 0.01, dtype=tf.float32)
    # feedback weights from v layer into q layer (~transition probabilities in Bellman equation)
    w_fb = tf.Variable(np.random.randn(3, 3, 1, ch_q) * 0.01, dtype=tf.float32)
    w_o = tf.Variable(np.random.randn(ch_q, 8) * 0.01, dtype=tf.float32)

    # initial conv layer over image+reward prior
    h = conv2d_flipkernel(X, w0, name="h0") + bias

    r = conv2d_flipkernel(h, w1, name="r")
    q = conv2d_flipkernel(r, w, name="q")
    v = tf.reduce_max(q, axis=3, keep_dims=True, name="v")

    for i in range(0, k - 1):
        rv = tf.concat([r, v], 3)  # connect two matrix
        wwfb = tf.concat([w, w_fb], 2)
        q = conv2d_flipkernel(rv, wwfb, name="q")
        v = tf.reduce_max(q, axis=3, keep_dims=True, name="v")

    # do one last convolution
    q = conv2d_flipkernel(tf.concat([r, v], 3),
                          tf.concat([w, w_fb], 2), name="q")

    # CHANGE TO THEANO ORDERING
    # Since we are selecting over channels, it becomes easier to work with
    # the tensor when it is in NCHW format vs NHWC
    q = tf.transpose(q, perm=[0, 3, 1, 2])

    # Select the conv-net channels at the state position (S1,S2).
    # This intuitively corresponds to each channel representing an action, and the convnet the Q function.
    # The tricky thing is we want to select the same (S1,S2) position *for each* channel and for each sample
    # TODO: performance can be improved here by substituting expensive
    #       transpose calls with better indexing for gather_nd
    bs = tf.shape(q)[0]
    rprn = tf.reshape(
        tf.tile(tf.reshape(tf.range(bs), [-1, 1]), [1, state_batch_size]), [-1])
    ins1 = tf.cast(tf.reshape(S1, [-1]), tf.int32)
    ins2 = tf.cast(tf.reshape(S2, [-1]), tf.int32)
    idx_in = tf.transpose(tf.stack([ins1, ins2, rprn]), [1, 0])
    q_out = tf.gather_nd(tf.transpose(q, [2, 3, 0, 1]), idx_in, name="q_out")
    logits = tf.matmul(q_out, w_o)
    output = tf.nn.softmax(logits, name="output")
    return logits, output


def DB_CNN(X, S1, S2, config, keep_drop=0.5):
    k = config.k    # Number of value iterations performed
    ch_i = config.ch_i  # Channels in input layer
    ch_h = config.ch_h  # Channels in initial hidden layer
    ch_q = config.ch_q  # Channels in q layer (~actions)
    state_batch_size = config.statebatchsize  # k+1 state inputs for each channel
    N = 20
    M = 20

    bias = tf.Variable(tf.truncated_normal(
        [1, 1, 1, N], dtype=tf.float32, stddev=0.01))
    # weights from inputs to q layer (~reward in Bellman equation)
    w0 = tf.Variable(tf.truncated_normal(
        [7, 7, ch_i, N], dtype=tf.float32, stddev=0.01))
    w1 = tf.Variable(tf.truncated_normal(
        [5, 5, N, N], dtype=tf.float32, stddev=0.01))
    w2 = tf.Variable(tf.truncated_normal(
        [5, 5, N, N], dtype=tf.float32, stddev=0.01))
    w3 = tf.Variable(tf.truncated_normal(
        [5, 5, N, N], dtype=tf.float32, stddev=0.01))
    w4 = tf.Variable(tf.truncated_normal(
        [5, 5, N, N], dtype=tf.float32, stddev=0.01))
    w = tf.Variable(tf.truncated_normal(
        [3, 3, N, ch_q], dtype=tf.float32, stddev=0.01))
    # feedback weights from v layer into q layer (~transition probabilities in Bellman equation)
    w_o = tf.Variable(tf.truncated_normal(
        [ch_q * 1, 8], dtype=tf.float32, stddev=0.01))

    #h = tf.nn.relu(conv2d_flipkernel(X, w0, name="h0") + bias)
    #pool1 = tf.nn.max_pool(h, [1, 3, 3, 1], [1, 1, 1, 1], padding='SAME')
    #r = tf.nn.relu(conv2d_flipkernel(h, w1, name="r"))
    #pool2 = tf.nn.max_pool(r, [1, 3, 3, 1], [1, 1, 1, 1], padding='SAME')
    #r1 = tf.nn.relu(conv2d_flipkernel(r, w2, name="r"))
    #pool3 = tf.nn.max_pool(r1, [1, 3, 3, 1], [1, 1, 1, 1], padding='SAME')
    # Residual Block
    h = tf.nn.relu(conv2d_flipkernel(X, w0, name="h0") + bias)
    r = tf.nn.relu(conv2d_flipkernel(h, w1, name="r"))
    r1 = tf.nn.relu(conv2d_flipkernel(r, w1, name="r1") )#+ h)
    r2 = tf.nn.relu(conv2d_flipkernel(r1, w2, name="r2"))
    r3 = tf.nn.relu(conv2d_flipkernel(r2, w2, name="r3") )#+ r1)
    r4 = tf.nn.relu(conv2d_flipkernel(r3, w3, name="r4"))
    r5 = tf.nn.relu(conv2d_flipkernel(r4, w3, name="r5") )#+ r3)
    r6 = tf.nn.relu(conv2d_flipkernel(r5, w4, name="r6"))
    r7 = tf.nn.relu(conv2d_flipkernel(r6, w4, name="r7") )#+ r5)
    # r8 = tf.nn.relu(conv2d_flipkernel(r7, w5, name="r8"))
    # r9 = tf.nn.relu(conv2d_flipkernel(r8, w5, name="r9") + r7)

    q = conv2d_flipkernel(r7, w, name="q")
    #q = tf.transpose(q, perm=[0, 3, 1, 2])

    bs = tf.shape(q)[0]
    rprn = tf.reshape(
        tf.tile(tf.reshape(tf.range(bs), [-1, 1]), [1, state_batch_size]), [-1])
    ins1 = tf.cast(tf.reshape(S1, [-1]), tf.int32)
    ins2 = tf.cast(tf.reshape(S2, [-1]), tf.int32)
    idx_in = tf.transpose(tf.stack([ins1, ins2, rprn]), [1, 0])
    q_out = tf.gather_nd(tf.transpose(q, [1, 2, 0, 3]), idx_in, name="q_out")

    w20 = tf.Variable(tf.truncated_normal(
        [5, 5, 2, M], dtype=tf.float32, stddev=0.01))
    kernel20 = tf.nn.relu(tf.nn.conv2d(X, w20, [1, 1, 1, 1], padding='SAME'))
    bias20 = tf.Variable(tf.constant(0.0, shape=[M], dtype=tf.float32))
    conv20 = tf.nn.relu(tf.nn.bias_add(kernel20, bias20))
    pool20 = tf.nn.max_pool(conv20, [1, 3, 3, 1], [1, 1, 1, 1], padding='SAME')

    w21 = tf.Variable(tf.truncated_normal(
        [3, 3, M, M], dtype=tf.float32, stddev=0.01))
    k211 = tf.nn.relu(conv2d_flipkernel(pool20, w21, name="k211"))
    k212 = tf.nn.relu(conv2d_flipkernel(k211, w21, name="k212"))
    k213 = tf.nn.relu(conv2d_flipkernel(k212, w21, name="k213") + pool20)
    pool21 = tf.nn.max_pool(k213, [1, 3, 3, 1], [1, 2, 2, 1], padding='SAME')

    w22 = tf.Variable(tf.truncated_normal(
        [3, 3, M, M], dtype=tf.float32, stddev=0.01))
    k221 = tf.nn.relu(conv2d_flipkernel(pool21, w22, name="k221"))
    k222 = tf.nn.relu(conv2d_flipkernel(k221, w22, name="k222"))
    k223 = tf.nn.relu(conv2d_flipkernel(k222, w22, name="k222") + pool21)
    pool22 = tf.nn.max_pool(k222, [1, 3, 3, 1], [1, 2, 2, 1], padding='SAME')

    w23 = tf.Variable(tf.truncated_normal(
        [3, 3, M, M], dtype=tf.float32, stddev=0.01))
    k231 = tf.nn.relu(conv2d_flipkernel(pool22, w23, name="k231"))
    k232 = tf.nn.relu(conv2d_flipkernel(k231, w23, name="k232") + pool22)
    pool23 = tf.nn.max_pool(k232, [1, 3, 3, 1], [1, 2, 2, 1], padding='SAME')

    a, b, c, d = np.shape(pool23)
    reshape = tf.reshape(pool23, [config.batch_size, b * c * d])
    reshape_drop = tf.nn.dropout(reshape, keep_drop)
    dim = reshape.get_shape()[1].value
    w24 = tf.Variable(tf.truncated_normal([dim, 192], stddev=0.01))
    bias24 = tf.Variable(tf.constant(0.0, shape=[192], dtype=tf.float32))
    local24 = tf.nn.relu(tf.matmul(reshape_drop, w24) + bias24)
    local24_drop = tf.nn.dropout(local24, keep_drop)

    w25 = tf.Variable(tf.truncated_normal([192, ch_q], stddev=0.01))
    bias25 = tf.Variable(tf.constant(0.0, shape=[ch_q], dtype=tf.float32))
    local25 = tf.nn.relu(tf.matmul(local24_drop, w25) + bias25)

    local = tf.tile(local25, [1, state_batch_size])
    local = tf.reshape(local, [config.batch_size * state_batch_size, ch_q])
    local_a = tf.tile(local25, [1, config.imsize * config.imsize])
    local_a = tf.reshape(local_a, [bs, config.imsize, config.imsize, ch_q])
    # wo = tf.tile(tf.reshape(w_o, [1, 1, ch_q * 2, 8]),
    #              [bs, config.imsize, 1, 1])
    # Q = tf.nn.relu(tf.matmul(tf.concat([q, local_a], axis=-1), wo))
    # v = tf.reduce_max(Q, axis=3, keep_dims=True, name="v") * 255
    # V = tf.reduce_max(q, axis=3, keep_dims=True, name="v") * 255
    #vv = tf.tile(v,[1,1,1,2])
    #v = tf.concat([vv,255-v],axis=-1)

    Q_out = tf.concat([q_out, local], axis=1)
    #Q_out = tf.nn.relu(q_out + local25)
    #Q_out_drop = tf.nn.dropout(Q_out, keep_drop)
    # add logits
    logits = tf.nn.relu(tf.matmul(q_out, w_o))
    # print(np.shape(logits))
    # softmax output weights
    output = tf.nn.softmax(logits, name="output")
    l2_loss = tf.nn.l2_loss(w0) + tf.nn.l2_loss(w1) + tf.nn.l2_loss(w2) + tf.nn.l2_loss(w) + tf.nn.l2_loss(
        w_o) + tf.nn.l2_loss(w20) + tf.nn.l2_loss(w21) + tf.nn.l2_loss(w22) + tf.nn.l2_loss(w24)  # + tf.nn.l2_loss(w24)
    return logits, output, l2_loss


def DCNN(X, S1, S2, config, keep_drop):
    k = config.k    # Number of value iterations performed
    ch_i = config.ch_i  # Channels in input layer
    ch_h = config.ch_h  # Channels in initial hidden layer
    ch_q = config.ch_q  # Channels in q layer (~actions)
    state_batch_size = config.statebatchsize  # k+1 state inputs for each channel

    bias = tf.Variable(np.random.randn(1, 1, 1, ch_h) * 0.01, dtype=tf.float32)
    # weights from inputs to q layer (~reward in Bellman equation)
    w0 = tf.Variable(np.random.randn(5, 5, ch_i, ch_h)
                     * 0.01, dtype=tf.float32)
    w1 = tf.Variable(np.random.randn(3, 3, ch_h, 100) * 0.01, dtype=tf.float32)
    w2 = tf.Variable(np.random.randn(3, 3, 100, 100) * 0.01, dtype=tf.float32)
    w3 = tf.Variable(np.random.randn(3, 3, 100, 50) * 0.01, dtype=tf.float32)
    w4 = tf.Variable(np.random.randn(3, 3, 50, 50) * 0.01, dtype=tf.float32)
    w = tf.Variable(np.random.randn(3, 3, 50, ch_q) * 0.01, dtype=tf.float32)
    w_o = tf.Variable(np.random.randn(ch_q, 8) * 0.01, dtype=tf.float32)

    # initial conv layer over image+reward prior
    h = tf.nn.relu(conv2d_flipkernel(X, w0, name="h0") + bias)

    r = tf.nn.relu(conv2d_flipkernel(h, w1, name="r"))
    r1 = tf.nn.relu(conv2d_flipkernel(r, w2, name="r1"))
    r2 = tf.nn.relu(conv2d_flipkernel(r1, w3, name="r2"))
    r3 = tf.nn.relu(conv2d_flipkernel(r2, w4, name="r3"))
    q = tf.nn.relu(conv2d_flipkernel(r3, w, name="q"))

    q = tf.transpose(q, perm=[0, 3, 1, 2])

    bs = tf.shape(q)[0]
    rprn = tf.reshape(
        tf.tile(tf.reshape(tf.range(bs), [-1, 1]), [1, state_batch_size]), [-1])
    ins1 = tf.cast(tf.reshape(S1, [-1]), tf.int32)
    ins2 = tf.cast(tf.reshape(S2, [-1]), tf.int32)
    idx_in = tf.transpose(tf.stack([ins1, ins2, rprn]), [1, 0])
    q_out = tf.gather_nd(tf.transpose(q, [2, 3, 0, 1]), idx_in, name="q_out")
    logits = tf.matmul(q_out, w_o)
    output = tf.nn.softmax(logits, name="output")
    return logits, output
