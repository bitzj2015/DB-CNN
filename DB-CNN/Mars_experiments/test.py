import time
import numpy as np
import tensorflow as tf
import tfplot
import matplotlib.pyplot as plt
from data1  import *
# from hierarchy_model import *
from model import *
from utils import *
import csv

# Data
tf.app.flags.DEFINE_string('input',           'data/gridworld_28.mat', 'Path to data')
tf.app.flags.DEFINE_integer('imsize',         128,                      'Size of input image')
# Parameters
tf.app.flags.DEFINE_float('lr',               0.001,                  'Learning rate for RMSProp')
tf.app.flags.DEFINE_integer('epochs',         200,                     'Maximum epochs to train for')
tf.app.flags.DEFINE_integer('k',              80,                     'Number of value iterations')
tf.app.flags.DEFINE_integer('ch_i',           3,                      'Channels in input layer')
tf.app.flags.DEFINE_integer('ch_h',           150,                    'Channels in initial hidden layer')
tf.app.flags.DEFINE_integer('ch_q',           10,                     'Channels in q layer (~actions)')
tf.app.flags.DEFINE_integer('batchsize',      1,                     'Batch size')
tf.app.flags.DEFINE_integer('statebatchsize', 1,                     'Number of state inputs for each sample (real number, technically is k+1)')
tf.app.flags.DEFINE_integer('mode', 1,                  'Untie weights of VI network')
tf.app.flags.DEFINE_float('belta', 0.0002, 'l2_loss')
#tf.app.flags.DEFINE_float('keep_drop', 0.8, 'dropout parameters')
# Misc.
tf.app.flags.DEFINE_integer('seed',           0,                      'Random seed for numpy')
tf.app.flags.DEFINE_integer('display_step',   1,                      'Print summary output every n epochs')
tf.app.flags.DEFINE_boolean('log',            True,                  'Enable for tensorboard summary')
tf.app.flags.DEFINE_string('logdir',          'tmp/show/',          'Directory to store tensorboard summary')

config = tf.app.flags.FLAGS

np.random.seed(config.seed)

# symbolic input image tensor where typically first channel is image, second is the reward prior
X  = tf.placeholder(tf.float32, name="X",  shape=[1, config.imsize, config.imsize, config.ch_i])
# symbolic input batches of vertical positions
S1 = tf.placeholder(tf.int32,   name="S1", shape=[None])
# symbolic input batches of horizontal positions
S2 = tf.placeholder(tf.int32,   name="S2", shape=[None])
y  = tf.placeholder(tf.int32,   name="y",  shape=[None])
LR  = tf.placeholder(tf.float32,   name="LR",  shape=())
keep_drop  = tf.placeholder(tf.float32,   name="keep_drop",  shape=())
length = tf.placeholder(tf.int32, name="length", shape=())

# Construct model (Value Iteration Network)
if (config.mode == 0):
    logits, nn, out, value = VIN(X, S1, S2, config, length)
elif (config.mode == 1):
	logits, nn,  l2_loss, out, value = DB_Net(X, S1, S2, config, keep_drop, length)
else:
    logits, nn = DCNN(X, S1, S2, config, keep_drop,length)

m = (1-X[:,:,:,0:1])*255
mm = X[:,:,:,1:2]
n = tf.placeholder(tf.float32, name="X",  shape=[None, 32, 32, 1])
print(np.shape(out))
# Define loss and optimizer
y_ = tf.cast(y, tf.int64)
cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
    logits=logits, labels=y_, name='cross_entropy')
cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy_mean') #+ config.belta * l2_loss
tf.add_to_collection('losses', cross_entropy_mean)

cost = tf.add_n(tf.get_collection('losses'), name='total_loss')
optimizer = tf.train.RMSPropOptimizer(learning_rate=LR, epsilon=1e-6, centered=True).minimize(cost)

# Test model & calculate accuracy
cp = tf.cast(tf.argmax(nn, 1), tf.int32)
err = tf.reduce_mean(tf.cast
                     (tf.not_equal(cp, y), dtype=tf.float32))
# Initializing the variables
init = tf.global_variables_initializer()
saver = tf.train.Saver()
flag = 0
if flag:
	Xtrain, S1train, S2train, ytrain, gtrain, Gtrain, Xtest, S1test, S2test, ytest, gtest, Gtest = process_data( )
else:
	Xtest, S1test, S2test, ytest, gtest, Gtest, Xtrain, S1train, S2train, ytrain, gtrain, Gtrain = process_data( )
# environment to return rewards. penalties are proportional to distances traversed
def env_ir(x, y, goal_x, goal_y, canvas):
	if canvas[x, y] == 5:
		return 1.0, True, x, y
	elif (x <= 0 or x >=31 or y <= 0 or y >=31):
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
tf.summary.image('map', m, 100)
tf.summary.image('output', out, 100)
tf.summary.image('value', value, 100)
tf.summary.image('goal', n, 100)
merged = tf.summary.merge_all()
test_writer = tf.summary.FileWriter(config.logdir)
sess.run(init)

batch_size    = config.batchsize
print(fmt_row(10, ["Epoch", "Train Cost", "Train Err", "Epoch Time"]))
saver = tf.train.Saver()
t_variable = tf.trainable_variables()

model_file=tf.train.latest_checkpoint('ckpt/')
saver.restore(sess,model_file)

nA = 8 # number of actions


# testing

rand_idx = np.random.permutation(np.arange(len(Xtest)))
domain_index = np.random.randint(Xtest.shape[0])  # pick a domain randomly
success = 0
failure = 0
total = 0
tot_steps = 0
tot_rewards = 0
avg_cost = 0
In = []
Su = []
csvFile1 = open('xlocal.csv','w', newline='')
writer1 = csv.writer(csvFile1)
csvFile2 = open('ylocal.csv','w', newline='')
writer2 = csv.writer(csvFile2)
csvFile3 = open('In.csv','w', newline='')
writer3 = csv.writer(csvFile3)
csvFile4 = open('Su.csv','w', newline='')
writer4 = csv.writer(csvFile4)

IN = np.array([83, 751, 1115, 72, 1284, 1119, 806, 1366, 569, 922])
for i, index in enumerate(rand_idx):
	if i < 10:
		index = IN[i]
	total += 1
	flag = 0
	obstacle = True
	In.append(index)
	xlocal = np.zeros(66)
	ylocal = np.zeros(66)
	start_x = S1test[0][index][0][0]
	start_y = S2test[0][index][0][0]
	'''
	while(obstacle):
		start_x = np.random.randint(1, 32 - 1)
		start_y = np.random.randint(1, 32 - 1)

		if (gtest[index, start_x, start_y] == 0):
			obstacle = False
	# retrieve current position
	'''

	current_x = np.array(start_x).reshape(1,1)
	current_y = np.array(start_y).reshape(1,1)
	xlocal[0] = current_x[0][0]
	ylocal[0] = current_y[0][0]

	domain = Xtest[index, :, :, :].reshape(1, config.imsize, config.imsize, 3)
	goal_xx = np.where(domain[:,:,:,2]==10.)[1][0]
	goal_yy = np.where(domain[:,:,:,2]==10.)[2][0]
	goal_x = Gtest[index,0,0]
	goal_y = Gtest[index,0,1]
	#print((goal_xx-2)/4, (goal_yy-2)/4, goal_x, goal_y)

	terminate = False
	step = 0
	maximum_step = 32*2

	canvas = gtest[index]
	#print(canvas[0:20,0:20])
	# canvas_orig = domain[0,:,:,0].copy()
	#print(goal_x,goal_y)
	val = np.zeros((1,32,32,1), 'int32')
	val[0, goal_x, goal_y, 0] = 5
	canvas[goal_x, goal_y] = 5
	#print(canvas[1:31,1:31])

	iter_ = 0
	ri = []
	b_s_t = []
	domain_in = []
	m1, m2, m3, m4  = sess.run([m, n, mm, value], {X: domain, S1: current_x[0], S2: current_y[0], keep_drop : 1, length: 1, n : val})
	#for a in range(32):
		#for b in range(32):
			#if m2[0,a,b,0] > 0:
				#m1[0, 4*a:4*a+4, 4*b:4*b+4, 0] = 255
	image = np.zeros((128,128,3),'float32')
	image[:,:,0] = m1[0,:,:,0]
	image[:,:,1] = m1[0,:,:,0]
	image[:,:,2] = m1[0,:,:,0]
	image = image.astype('float32')
	image[4*current_x[0][0]-2:4*current_x[0][0]+1, 4*current_y[0][0]-2:4*current_y[0][0]+1,0] = 0
	image[4*current_x[0][0]-2:4*current_x[0][0]+1, 4*current_y[0][0]-2:4*current_y[0][0]+1,1] = 255
	image[4*current_x[0][0]-2:4*current_x[0][0]+1, 4*current_y[0][0]-2:4*current_y[0][0]+1,2] = 0



	while (terminate==False):
		#print(i,iter_)
		iter_ += 1
		step += 1
		tot_steps += 1

		A = np.zeros(nA, dtype=float)
		best_action = sess.run(cp, {X: domain, S1: current_x[0], S2: current_y[0], keep_drop : 1, length: 1})
		A[best_action] = 1.0
		action_idx = np.random.choice(np.arange(len(A)), p=A)
		current_iter = iter_ - 1

		delta_x, delta_y = action2cord(action_idx)
		next_x = current_x[0][0] + delta_x
		next_y = current_y[0][0] + delta_y
		#next_x = S1test[index][iter_]
		#next_y = S2test[index][iter_]

		#print(i, next_x,next_y, goal_x, goal_y)

		reward, terminate, next_x, next_y = env_ir(next_x, next_y, goal_x, goal_y, canvas)

		tot_rewards += reward
		ri.append(reward)
		if reward == -1.0:
			failure += 1

		b_s_t.append(np.array([current_x[0][0], current_y[0][0]]))
		domain_in.append(np.squeeze(domain))

		if(next_x == goal_x and next_y == goal_y):
			success += 1
			flag = 1
			terminate = True

		if iter_ >= maximum_step:
			terminate = True

		for a in range(4):
			image[4*current_x[0][0]+a*delta_x, 4*current_y[0][0]+a*delta_y, 0] = 255
			image[4*current_x[0][0]+a*delta_x, 4*current_y[0][0]+a*delta_y, 1] = 0
			image[4*current_x[0][0]+a*delta_x, 4*current_y[0][0]+a*delta_y, 2] = 0

		current_x[0][0] = next_x
		current_y[0][0] = next_y

		xlocal[iter_] = current_x[0][0]
		ylocal[iter_] = current_y[0][0]

	xlocal[iter_+1] = current_x[0][0]
	xlocal[iter_+1] = current_x[0][0]
	Su.append(flag)
	writer1.writerow(xlocal)
	writer2.writerow(ylocal)

	image[4*goal_x-2:4*goal_x+1, 4*goal_y-2:4*goal_y+1, 0] = 0
	image[4*goal_x-2:4*goal_x+1, 4*goal_y-2:4*goal_y+1, 1] = 0
	image[4*goal_x-2:4*goal_x+1, 4*goal_y-2:4*goal_y+1, 2] = 255

	image = image.astype('uint8')
	if i < 10:
		plt.figure()
		im = plt.imshow(m1[0,:,:,0], cmap=plt.cm.gray)
		plt.savefig("mM1"+str(i)+".png", bbox_inches = 'tight')
		#im = plt.imshow(image, cmap='jet')
		#plt.colorbar(im)
		plt.show()
		#if flag == 0:
			#plt.savefig("success"+str(i)+".eps", bbox_inches = 'tight')
		#else:
			#plt.savefig("failure"+str(i)+".eps", bbox_inches = 'tight')
		plt.figure()
		im = plt.imshow(m2[0,:,:,0]*255, cmap=plt.cm.gray)
		#plt.colorbar(im)
		plt.show()
		plt.savefig("mM2"+str(i)+".png", bbox_inches = 'tight')
		plt.figure()
		im = plt.imshow(m3[0,:,:,0]*255, cmap=plt.cm.gray)#/10000*30)
		#plt.colorbar(im)
		#print(i)
		plt.show()
		plt.savefig("mM3"+str(i)+".png", bbox_inches = 'tight')
		#plt.figure()
		#im = plt.imshow(m4[0,:,:,0]/75000)
		#plt.colorbar(im)
		#plt.show()
		#plt.savefig("m4"+str(i)+".png")

	if i % 100 == 0:
		print(success,failure)
		print("Step: " + str(i) + ", Current_Acc: " + str(success/float(total)) + ", accum rewards: " + str(tot_rewards/tot_steps) + 'total' + str(total))
writer3.writerow(In)
writer4.writerow(Su)
csvFile1.close()
csvFile2.close()
csvFile3.close()
csvFile4.close()
print("Step: " + str(i) + ", Current_Acc: " + str(success/float(total)) + ", accum rewards: " + str(tot_rewards/tot_steps) + 'total' + str(total))

