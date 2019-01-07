import time
import argparse

import numpy as np
import tensorflow as tf

import os
from Model import VIN, DB_Net, DCNN
from dataset import Dataset
import csv
from utils import *

# Parsing training parameters
parser = argparse.ArgumentParser()
parser.add_argument('--statebatchsize', type=str,
                    default=1, help='gg')
parser.add_argument('--datafile', type=str,
                    default='./data/data_64.pkl', help='Path to data file')
parser.add_argument('--imsize', type=int, default=64, help='Size of image')
parser.add_argument('--lr', type=float, default=0.001,
                    help='Learning rate, [0.01, 0.005, 0.002, 0.001]')
parser.add_argument('--epochs', type=int, default=30,
                    help='Number of epochs to train')
parser.add_argument('--k', type=int, default=80,
                    help='Number of Value Iterations')
parser.add_argument('--ch_i', type=int, default=2,
                    help='Number of channels in input layer')
parser.add_argument('--ch_h', type=int, default=150,
                    help='Number of channels in first hidden layer')
parser.add_argument('--ch_q', type=int, default=10,
                    help='Number of channels in q layer (~actions) in VI-module')
parser.add_argument('--batch_size', type=int,
                    default=1, help='Batch size')
parser.add_argument('--use_log', type=bool, default=False,
                    help='True to enable TensorBoard summary')
parser.add_argument('--logdir', type=str, default='./log/',
                    help='Directory to store TensorBoard summary')
parser.add_argument(
    '--save', type=str, default="./model/weights_res_64.ckpt", help='File to save the weights')
parser.add_argument(
    '--load', type=str, default="./model/weights_res_64.ckpt", help='File to load the weights')
parser.add_argument(
    '--load_model', type=bool, default=None, help="Whether to load model or not")
parser.add_argument(
    '--model_type', type=str, default="DB-CNN", help="Which model to choose")
args = parser.parse_args()

if not os.path.exists(os.path.dirname(args.save)):
    print("Error : file cannot be created (need folders) : " + args.save)

# Define placeholders

# Input tensor: Stack obstacle image and goal image, i.e. ch_i = 2
X = tf.placeholder(tf.float32, shape=[
                   None, args.imsize, args.imsize, args.ch_i], name='X')
# vertical positions
S1 = tf.placeholder(tf.int32, shape=[None, args.statebatchsize], name='S1')
# horizontal positions
S2 = tf.placeholder(tf.int32, shape=[None, args.statebatchsize], name='S2')
# labels : actions {0,1,2,3}
y = tf.placeholder(tf.int64, shape=[None], name='y')

# VIN model
if args.model_type == "VIN":
    logits, prob_actions = VIN(X, S1, S2, args)
elif args.model_type == "DB-CNN":
    logits, prob_actions, l2_loss = DB_Net(X, S1, S2, args)
else:
    print("No such model!")

# Training
cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
    labels=y, logits=logits, name='cross_entropy')
loss = tf.reduce_sum(cross_entropy, name='cross_entropy_mean')
train_step = tf.train.RMSPropOptimizer(
    args.lr, epsilon=1e-6, centered=True).minimize(loss)

# Testing
actions = tf.argmax(prob_actions, 1)
# Number of wrongly selected actions
num_err = tf.reduce_sum(tf.to_float(tf.not_equal(actions, y)))
print("loading trainset...")
trainset = Dataset(args.datafile, args.statebatchsize, mode='test')
print("loading testset...")
testset = Dataset(args.datafile, args.statebatchsize, mode='test')
print("loaded")
Xtest = trainset.images()
S1test = trainset.s1()
S2test=trainset.s2()
ytest = trainset.labels()
print("##################")
print(np.shape(Xtest))
print(np.shape(S1test))
# if flag:
#   Xtest, S1test, S2test, ytest, Xtest1, S1test1, S2test1, ytest1 = process_gridworld_data(input=config.input, imsize=config.imsize)
# else:
#     Xtest1, S1test1, S2test1, ytest1, Xtest, S1test, S2test, ytest = process_gridworld_data(input=config.input, imsize=config.imsize)
# environment to return rewards. penalties are proportional to distances traversed
def env_ir(x, y, goal_x, goal_y, canvas):
  if canvas[x, y] == 5:
    return 1.0, True, x, y
  elif (x <= 0 or x >= args.imsize-1 or y <= 0 or y >= args.imsize-1):
    return -1.0, True, 2, 2
  elif canvas[x, y] == 1:
    return -1.0, True, x, y
  else:
    return -0.05, False, x, y  #* dist_train_e[prev_pos,cur_pos],

# Visualize the walking
def action2cord(a):
  """
  input  : action 0~7
  output : x ,y changes
  """
  # return {'0':[0,-1],'1':[0,1],'2':[1,0],'3':[-1,0],'4':[1,-1],'5':[-1,-1],'6':[1,1],'7':[-1,1]}.get(str(a),[0,0])
  return {'0':[-1,0],'1':[1,0],'2':[0,1],'3':[0,-1],'4':[-1,1],'5':[-1,-1],'6':[1,1],'7':[1,-1]}.get(str(a),[0,0])


SessConfig = tf.ConfigProto(
        device_count = {'GPU': 0}
    )
saver = tf.train.Saver()
sess = tf.Session(config=SessConfig)
# tf.summary.image('map', m, 10)
# tf.summary.image('output', out, 10)
merged = tf.summary.merge_all()
test_writer = tf.summary.FileWriter(args.logdir + '/test')
#sess.run(init)

batch_size    = args.batch_size
print(fmt_row(10, ["Epoch", "Train Cost", "Train Err", "Epoch Time"]))
saver = tf.train.Saver()
t_variable = tf.trainable_variables()
saver.restore(sess, args.load)
# model_file=tf.train.latest_checkpoint('ckpt/')
# saver.restore(sess,model_file)

nA = 8 # number of actions


# testing

rand_idx = np.random.permutation(np.arange(len(Xtest)))
domain_index = np.random.randint(Xtest.shape[0])  # pick a domain randomly
success = 0
total = 0
tot_steps = 0
tot_rewards = 0
avg_cost = 0

for i, index in enumerate(rand_idx):
  total += 1
  obstacle = True
  start_x = S1test[index,0]
  start_y = S2test[index,0]
  '''
  while(obstacle):  # random pick a start point if there is obstacle, pick other one
    start_x = np.random.randint(1, config.imsize - 1)
    start_y = np.random.randint(1, config.imsize - 1)

    if (Xtest[index, start_x, start_y, 0] == 0):
      obstacle = False
'''
  # retrieve current position


  current_x = np.array(start_x).reshape(1,1).astype("int32")
  current_y = np.array(start_y).reshape(1,1).astype("int32")

  domain = Xtest[index, :, :, :].reshape(1, args.imsize, args.imsize, 2)

  goal_x = np.where(domain==100.)[1][0]
  goal_y = np.where(domain==100.)[2][0]

  terminate = False
  step = 0
  maximum_step = args.imsize*2

  canvas = domain[0,:,:,0].copy()
  # canvas_orig = domain[0,:,:,0].copy()
  canvas[goal_x, goal_y] = 5

  iter_ = 0
  ri = []
  b_s_t = []
  domain_in = []
  failure = 0
 # summary = sess.run(merged, {X: domain, S1: current_x, S2: current_y, keep_drop : 1, m : np.reshape(canvas,[1,args.imsize, args.imsize,1])})
  #test_writer.add_summary(summary, i)

  while (terminate==False):
    iter_ += 1
    step += 1
    tot_steps += 1

    A = np.zeros(nA, dtype=float)
    best_action = sess.run(actions, {X: domain, S1: current_x, S2: current_y})
    A[best_action] = 1.0
    action_idx = np.random.choice(np.arange(len(A)), p=A)
    current_iter = iter_ - 1
    delta_x, delta_y = action2cord(action_idx)
    next_x = current_x[0,0] + delta_x
    next_y = current_y[0,0] + delta_y
    #print(type(current_x[0,0]))
    reward, terminate, next_x, next_y = env_ir(next_x, next_y, goal_x, goal_y, canvas)

    tot_rewards += reward
    ri.append(reward)
    if reward == -1.0:
      failure = 1

    b_s_t.append(np.array([current_x[0,0], current_y[0,0]]))
    domain_in.append(np.squeeze(domain))

    if(next_x == goal_x and next_y == goal_y):
      success += 1
    if iter_ >= maximum_step:
      terminate = True
    current_x[0,0] = next_x
    current_y[0,0] = next_y
  if i % 1000 == 0:
    print("Step: " + str(i) + ", Current_Acc: " + str(success/float(total)) + ", accum rewards: " + str(tot_rewards/tot_steps) + 'total' + str(total))
print("Step: " + str(i) + ", Current_Acc: " + str(success/float(total)) + ", accum rewards: " + str(tot_rewards/tot_steps) + 'total' + str(total))

