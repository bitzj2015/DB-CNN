import numpy as np
import tensorflow as tf
from utils import conv2d_flipkernel, conv2d_flipkernel_d


def VIN(X, S1, S2, config, length):
    keep_drop = 0.5
    k = config.k    # Number of value iterations performed
    ch_i = config.ch_i  # Channels in input layer
    ch_h = config.ch_h  # Channels in initial hidden layer
    ch_q = config.ch_q  # Channels in q layer (~actions)
    state_batch_size = length  # k+1 state inputs for each channel

    w11 = tf.Variable(np.random.randn(5, 5, 2, 6) * 0.01, dtype=tf.float32)
    w22 = tf.Variable(np.random.randn(4, 4, 6, 12) * 0.01, dtype=tf.float32)

    bias = tf.Variable(np.random.randn(1, 1, 1, ch_h) * 0.01, dtype=tf.float32)
    w0 = tf.Variable(np.random.randn(3, 3, 13, ch_h) * 0.01, dtype=tf.float32)
    w1 = tf.Variable(np.random.randn(1, 1, ch_h, 1) * 0.01, dtype=tf.float32)
    w = tf.Variable(np.random.randn(3, 3, 1, ch_q) * 0.01, dtype=tf.float32)
    w_fb = tf.Variable(np.random.randn(3, 3, 1, ch_q) * 0.01, dtype=tf.float32)
    w_o = tf.Variable(np.random.randn(ch_q, 8) * 0.01, dtype=tf.float32)

    h1 = conv2d_flipkernel(X[:, :, :, 0:2], w11, name="h1")
    pool0 = tf.nn.max_pool(h1, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')
    h2 = conv2d_flipkernel(pool0, w22, name="h2")
    pool1 = tf.nn.max_pool(h2, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')
    pool00 = tf.nn.max_pool(X[:, :, :, 2:3], [1, 4, 4, 1], [
                            1, 4, 4, 1], padding='SAME')
    r0 = tf.concat([pool1, pool00], axis=-1)

    h = conv2d_flipkernel(r0, w0, name="h0") + bias

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
    v = tf.reduce_max(q, axis=3, keep_dims=True, name="v")
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
    return logits, output, v, v


def DB_CNN(X, S1, S2, config, keep_drop, length):
    k = config.k    # Number of value iterations performed
    ch_i = config.ch_i  # Channels in input layer
    ch_h = config.ch_h  # Channels in initial hidden layer
    ch_q = config.ch_q  # Channels in q layer (~actions)
    state_batch_size = length  # k+1 state inputs for each channel
    N = 20
    M = 20

    bias = tf.Variable(tf.truncated_normal(
        [1, 1, 1, N], dtype=tf.float32, stddev=0.001))
    w11 = tf.Variable(np.random.randn(5, 5, 2, 6) * 0.001, dtype=tf.float32)
    w22 = tf.Variable(np.random.randn(4, 4, 6, 12) * 0.001, dtype=tf.float32)
    w0 = tf.Variable(tf.truncated_normal(
        [3, 3, 13, N], dtype=tf.float32, stddev=0.001))
    w1 = tf.Variable(tf.truncated_normal(
        [3, 3, N, N], dtype=tf.float32, stddev=0.001))
    w2 = tf.Variable(tf.truncated_normal(
        [3, 3, N, N], dtype=tf.float32, stddev=0.001))
    w3 = tf.Variable(tf.truncated_normal(
        [3, 3, N, N], dtype=tf.float32, stddev=0.001))
    w4 = tf.Variable(tf.truncated_normal(
        [3, 3, N, N], dtype=tf.float32, stddev=0.001))
    w5 = tf.Variable(tf.truncated_normal(
        [3, 3, N, N], dtype=tf.float32, stddev=0.001))
    w6 = tf.Variable(tf.truncated_normal(
        [3, 3, N, N], dtype=tf.float32, stddev=0.001))
    w = tf.Variable(tf.truncated_normal(
        [3, 3, N, ch_q], dtype=tf.float32, stddev=0.001))
    w_o = tf.Variable(tf.truncated_normal(
        [ch_q * 1, 8], dtype=tf.float32, stddev=0.001))

    h1 = tf.nn.relu(conv2d_flipkernel(X[:, :, :, 0:2] * 100, w11, name="h1"))
    pool0 = tf.nn.max_pool(h1, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')
    h2 = tf.nn.relu(conv2d_flipkernel(pool0, w22, name="h2"))
    pool1 = tf.nn.max_pool(h2, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')
    #h3 = conv2d_flipkernel(pool1, w33, name="h2")
    pool00 = tf.nn.max_pool(X[:, :, :, 2:3] * 10,
                            [1, 4, 4, 1], [1, 4, 4, 1], padding='SAME')
    r0 = tf.concat([pool1, pool00], axis=-1)
    h = tf.nn.relu(conv2d_flipkernel(r0, w0, name="h") + bias)
    r = tf.nn.relu(conv2d_flipkernel(h, w1, name="r"))
    r1 = tf.nn.relu(conv2d_flipkernel(r, w1, name="r1"))
    r2 = tf.nn.relu(conv2d_flipkernel(r1, w2, name="r2"))
    r3 = tf.nn.relu(conv2d_flipkernel(r2, w2, name="r3"))
    r4 = tf.nn.relu(conv2d_flipkernel(r3, w3, name="r4"))
    r5 = tf.nn.relu(conv2d_flipkernel(r4, w3, name="r5"))
    r6 = tf.nn.relu(conv2d_flipkernel(r5, w4, name="r6"))
    r7 = tf.nn.relu(conv2d_flipkernel(r6, w4, name="r7"))
    r8 = tf.nn.relu(conv2d_flipkernel(r7, w5, name="r6"))
    r9 = tf.nn.relu(conv2d_flipkernel(r8, w5, name="r7"))

    q = conv2d_flipkernel(r9, w, name="q")

    bs = tf.shape(q)[0]
    rprn = tf.reshape(
        tf.tile(tf.reshape(tf.range(bs), [-1, 1]), [1, state_batch_size]), [-1])
    ins1 = tf.cast(tf.reshape(S1, [-1]), tf.int32)
    ins2 = tf.cast(tf.reshape(S2, [-1]), tf.int32)
    idx_in = tf.transpose(tf.stack([ins1, ins2, rprn]), [1, 0])
    q_out = tf.gather_nd(tf.transpose(q, [1, 2, 0, 3]), idx_in, name="q_out")

    w20 = tf.Variable(tf.truncated_normal(
        [5, 5, 13, M], dtype=tf.float32, stddev=0.001))
    kernel20 = tf.nn.relu(tf.nn.conv2d(r0, w20, [1, 1, 1, 1], padding='SAME'))
    bias20 = tf.Variable(tf.constant(0.0, shape=[M], dtype=tf.float32))
    conv20 = tf.nn.relu(tf.nn.bias_add(kernel20, bias20))
    pool20 = tf.nn.max_pool(conv20, [1, 3, 3, 1], [1, 2, 2, 1], padding='SAME')

    w21 = tf.Variable(tf.truncated_normal(
        [3, 3, M, M], dtype=tf.float32, stddev=0.001))
    k211 = tf.nn.relu(conv2d_flipkernel(pool20, w21, name="k211"))
    #k212 = tf.nn.relu(conv2d_flipkernel(k211, w21, name="k212"))
    k213 = tf.nn.relu(conv2d_flipkernel(k211, w21, name="k213") + pool20)
    pool21 = tf.nn.max_pool(k213, [1, 3, 3, 1], [1, 2, 2, 1], padding='SAME')

    w22 = tf.Variable(tf.truncated_normal(
        [3, 3, M, M], dtype=tf.float32, stddev=0.001))
    k221 = tf.nn.relu(conv2d_flipkernel(pool21, w22, name="k221"))
    #k222 = tf.nn.relu(conv2d_flipkernel(k221, w22, name="k222"))
    k223 = tf.nn.relu(conv2d_flipkernel(k221, w22, name="k222") + pool21)
    pool22 = tf.nn.max_pool(k223, [1, 3, 3, 1], [1, 1, 1, 1], padding='SAME')

    w23 = tf.Variable(tf.truncated_normal(
        [3, 3, M, M], dtype=tf.float32, stddev=0.01))
    k231 = tf.nn.relu(conv2d_flipkernel(pool22, w23, name="k231"))
    k232 = tf.nn.relu(conv2d_flipkernel(k231, w23, name="k232") + pool22)
    pool23 = tf.nn.max_pool(k232, [1, 3, 3, 1], [1, 1, 1, 1], padding='SAME')

    a, b, c, d = np.shape(pool23)
    reshape = tf.reshape(pool23, [config.batchsize, b * c * d])
    reshape_drop = tf.nn.dropout(reshape, keep_drop)
    dim = reshape.get_shape()[1].value
    w24 = tf.Variable(tf.truncated_normal([dim, 192], stddev=0.001))
    bias24 = tf.Variable(tf.constant(0.0, shape=[192], dtype=tf.float32))
    local24 = tf.nn.relu(tf.matmul(reshape_drop, w24) + bias24)
    local24_drop = tf.nn.dropout(local24, keep_drop)

    w25 = tf.Variable(tf.truncated_normal([192, ch_q], stddev=0.001))
    bias25 = tf.Variable(tf.constant(0.0, shape=[ch_q], dtype=tf.float32))
    local25 = tf.nn.relu(tf.matmul(local24_drop, w25) + bias25)

    local = tf.tile(local25, [1, state_batch_size])
    local = tf.reshape(local, [config.batchsize * state_batch_size, ch_q])
    local_a = tf.tile(local25, [1, 32 * 32])
    local_a = tf.reshape(local_a, [bs, 32, 32, ch_q])
    wo = tf.tile(tf.reshape(w_o, [1, 1, ch_q * 1, 8]), [bs, 32, 1, 1])
    Q = tf.nn.relu(tf.matmul(tf.concat([q, local_a], axis=3), wo))
    v = tf.reduce_max(Q, axis=3, keep_dims=True, name="v") * 255
    V = tf.reduce_max(q, axis=3, keep_dims=True, name="v") * 255

    logits = tf.nn.relu(tf.matmul(q_out, w_o))
    output = tf.nn.softmax(logits, name="output")
    l2_loss = tf.nn.l2_loss(w0) + tf.nn.l2_loss(w1) + tf.nn.l2_loss(w2) + tf.nn.l2_loss(w3) + tf.nn.l2_loss(w4) + tf.nn.l2_loss(w5) + tf.nn.l2_loss(

    return logits, output, l2_loss, v, V


def DCNN(X, S1, S2, config, keep_drop, length):
    keep_drop = 0.5
    k = config.k    # Number of value iterations performed
    ch_i = config.ch_i  # Channels in input layer
    ch_h = config.ch_h  # Channels in initial hidden layer
    ch_q = config.ch_q  # Channels in q layer (~actions)
    state_batch_size = length  # k+1 state inputs for each channel

    w11 = tf.Variable(np.random.randn(5, 5, 2, 6) * 0.01, dtype=tf.float32)
    w22 = tf.Variable(np.random.randn(4, 4, 6, 12) * 0.01, dtype=tf.float32)

    bias = tf.Variable(np.random.randn(1, 1, 1, ch_h) * 0.01, dtype=tf.float32)
    # weights from inputs to q layer (~reward in Bellman equation)
    w0 = tf.Variable(np.random.randn(5, 5, 13, ch_h) * 0.01, dtype=tf.float32)
    w1 = tf.Variable(np.random.randn(3, 3, ch_h, 100) * 0.01, dtype=tf.float32)
    w2 = tf.Variable(np.random.randn(3, 3, 100, 100) * 0.01, dtype=tf.float32)
    w3 = tf.Variable(np.random.randn(3, 3, 100, 100) * 0.01, dtype=tf.float32)
    w4 = tf.Variable(np.random.randn(3, 3, 100, 50) * 0.01, dtype=tf.float32)
    w5 = tf.Variable(np.random.randn(3, 3, 50, 50) * 0.01, dtype=tf.float32)
    w6 = tf.Variable(np.random.randn(3, 3, 50, 50) * 0.01, dtype=tf.float32)
    w = tf.Variable(np.random.randn(3, 3, 50, ch_q) * 0.01, dtype=tf.float32)
    w_o = tf.Variable(np.random.randn(ch_q, 8) * 0.01, dtype=tf.float32)

    # initial conv layer over image+reward prior
    h1 = tf.nn.relu(conv2d_flipkernel(X[:, :, :, 0:2], w11, name="h1"))
    pool0 = tf.nn.max_pool(h1, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')
    h2 = tf.nn.relu(conv2d_flipkernel(pool0, w22, name="h2"))
    pool1 = tf.nn.max_pool(h2, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')
    pool00 = tf.nn.max_pool(X[:, :, :, 2:3], [1, 4, 4, 1], [
                            1, 4, 4, 1], padding='SAME')
    r0 = tf.concat([pool1, pool00], axis=-1)

    h = tf.nn.relu(conv2d_flipkernel(r0, w0, name="h0") + bias)

    r = tf.nn.relu(conv2d_flipkernel(h, w1, name="r"))
    r1 = tf.nn.relu(conv2d_flipkernel(r, w2, name="r1"))
    r2 = tf.nn.relu(conv2d_flipkernel(r1, w3, name="r2"))
    r3 = tf.nn.relu(conv2d_flipkernel(r2, w4, name="r3"))
    r4 = tf.nn.relu(conv2d_flipkernel(r3, w5, name="r4"))
    r5 = tf.nn.relu(conv2d_flipkernel(r4, w6, name="r5"))
    #r6 = tf.nn.relu(conv2d_flipkernel(r5, w7, name="r6"))
    #r7 = tf.nn.relu(conv2d_flipkernel(r6, w8, name="r7"))
    q = tf.nn.relu(conv2d_flipkernel(r5, w, name="q"))

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
