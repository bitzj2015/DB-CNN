import time
import random
import numpy as np
#import cv2
import scipy.io as sio
from gym_pathfinding.games.gridworld import generate_grid, MOUVEMENT
from gym_pathfinding.games.astar import astar


def action2cord(a):
	return {'0':[-1,0],'1':[1,0],'2':[0,1],'3':[0,-1],'4':[-1,1],'5':[-1,-1],'6':[1,1],'7':[1,-1]}.get(str(a),[0,0])
 
def cord2action(x,y):
	if x == -1 and y == 0:
		return 0
	elif x == 1 and y == 0:
		return 1
	elif x == 0 and y == 1:
		return 2
	elif x == 0 and y == -1:
		return 3
	elif x == -1 and y == 1:
		return 4
	elif x == -1 and y == -1:
		return 5
	elif x == 1 and y == 1:
		return 6
	elif x == 1 and y == -1:
		return 7
	else:
		return False

def generate_data( ):
	imsize = 128
	im_size=[imsize, imsize]
	matlab_data = sio.loadmat('data/data.mat')
	map_data = np.transpose(matlab_data["Data_mat"].astype('float32'),[2,0,1])
	im_data =( 255-np.transpose(matlab_data["Img_mat"].astype('float32'),[2,0,1]))/255
	num = len(map_data)
	rand_idx = np.random.permutation(np.arange(num))
	value_data = np.zeros((num, imsize, imsize), 'float32')
	X_data = np.zeros((num,imsize,imsize,3),'float32')
	start_data = np.zeros((num, 1, 2), 'int32')
	goal_data = np.zeros((num, 1, 2), 'int32')
	grid_data = np.zeros((num, 32, 32), 'int32')
	action_data = []
	statex_data = []
	statey_data = []
	# when tseting, num=10
	for i, index in enumerate(rand_idx):
		count = 0
		flagg=0
		End = True
		while(End):	
			count += 1
			if count > 5:
				index -= 1
				count = 0
			for m in range(32):
				for n in range(32):
					grid_data[i,m,n] = np.max(map_data[index, 4*m : 4*m+4, 4*n : 4*n+4])	
					flagg = index
			#print(i, count, index)	
			#grid_data[i,0,:] = 1
			#grid_data[i,31,:] = 1
			#grid_data[i,:,0] = 1
			#grid_data[i,:,31] = 1

				
			flag = 1
			while(flag):
				goal_data[i,0,0] = np.random.randint(30)+1
				goal_data[i,0,1] = np.random.randint(30)+1
				if grid_data[i,goal_data[i,0,0],goal_data[i,0,1]] != 0:
					flag = 1
				else:
					flag = 0
			
			flag = 1
			while(flag):
				start_data[i,0,0] = np.random.randint(30) + 1
				start_data[i,0,1] = np.random.randint(30) + 1
				if grid_data[i,start_data[i,0,0],start_data[i,0,1]] == 0:
					flag = 0
					
			X = start_data[i,0,0]
			Y = start_data[i,0,1]
			GX = goal_data[i,0,0]
			GY = goal_data[i,0,1] 
			start=list(zip(np.array([X]),np.array([Y])))
			goal=list(zip(np.array([GX]),np.array([GY])))
			S = random.Random(None).sample(start, 1)
			G = random.Random(None).sample(goal, 1)
			action = astar(grid_data[i], S[0],G[0])
			if action != False and len(action)>2:
				End = False
					
		s_len = len(action)
		statex = np.zeros((s_len-1), 'int32')
		statey = np.zeros((s_len-1), 'int32')
		ylabel = np.zeros((s_len-1), 'int32')
		for j in range(s_len-1):
			statex[j] = action[j][0]
			statey[j] = action[j][1]
			ylabel[j] = cord2action(action[j+1][0] - action[j][0], action[j+1][1] - action[j][1])
		statex_data.append(statex)
		statey_data.append(statey)
		action_data.append(ylabel)
		if flagg-index > 0:
			print("hhh")
		value_data[i,goal_data[i,0,0]*4+2,goal_data[i,0,1]*4+2] = 10
		X_data[i,:,:,0] = im_data[index,:,:]
		X_data[i,:,:,1] = map_data[index,:,:]
		X_data[i,:,:,2] = value_data[i,:,:]
		#print(np.where(X_data[i,:,:,1]==100.),goal_data[i]*4+2)
		
	all_training_samples = int(6/7*num)
	training_samples = all_training_samples
	Xtrain = X_data[0:training_samples]
	S1train = statex_data[0:training_samples]
	S2train = statey_data[0:training_samples]
	ytrain = action_data[0:training_samples]
	gtrain = grid_data[0:training_samples]
	Gtrain = goal_data[0:training_samples]

	Xtest = X_data[all_training_samples:]
	S1test = statex_data[all_training_samples:]
	S2test = statey_data[all_training_samples:]
	ytest = action_data[all_training_samples:]
	gtest = grid_data[all_training_samples:]
	Gtest = goal_data[all_training_samples:]
	
	sio.savemat('data/dataset0.mat', {'Xtrain':Xtrain, 'S1train':S1train, 'S2train':S2train, 'ytrain':ytrain, 'gtrain':gtrain, 'Gtrain':Gtrain, 'Xtest':Xtest, 'S1test':S1test, 'S2test':S2test, 'ytest':ytest, 'gtest':gtest, 'Gtest':Gtest}) 


def process_data():
	matlab_data = sio.loadmat('data/dataset0.mat')
	Xtrain = matlab_data["Xtrain"].astype('float32')
	S1train = matlab_data["S1train"]
	S2train = matlab_data["S2train"]
	ytrain = matlab_data["ytrain"]
	gtrain = matlab_data["gtrain"].astype('int32')
	Gtrain = matlab_data["Gtrain"]

	Xtest = matlab_data["Xtest"].astype('float32')
	S1test = matlab_data["S1test"]
	S2test = matlab_data["S2test"]
	ytest = matlab_data["ytest"]
	gtest = matlab_data["gtest"].astype('int32')
	Gtest = matlab_data["Gtest"]
	
	print(np.shape(Xtest), np.shape(S1train), np.shape(S2test[0][0][0]), np.shape(ytest[0][100][0]))
	print(S1test[0][100][0])
	print(S2test[0][100][0])
	print(ytest[0][100][0])
	print(gtest[0,1:31,1:31])
	print(np.max(Xtest[0,1:100,1:100,0]))
	print(np.min(Xtest[0,1:100,1:100,0]))
	
	return Xtrain, S1train, S2train, ytrain, gtrain, Gtrain, Xtest, S1test, S2test, ytest, gtest, Gtest

if __name__ == '__main__':
	generate_data( )

