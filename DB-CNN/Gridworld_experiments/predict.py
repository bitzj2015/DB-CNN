import gym
import gym_pathfinding
from time import sleep
import tensorflow as tf 
import numpy as np 
from model import VIN

def main():

    # Input tensor: Stack obstacle image and goal image, i.e. ch_i = 2
    X = tf.placeholder(tf.float32, shape=[None, 9, 9, 2], name='X')
    S1 = tf.placeholder(tf.int32, shape=[None], name='S1') # vertical positions
    S2 = tf.placeholder(tf.int32, shape=[None], name='S2') # horizontal positions

    # VIN model
    _, prob_actions = VIN(X, S1, S2, k=10, ch_i=2, ch_h=150, ch_q=4)


    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, "./model/weights.ckpt")

        env = gym.make('pathfinding-obstacle-9x9-v0')

        dones = 0
        episodes = 100
        for episode in range(episodes):
            print(episode, end="\r")
            
            state = env.reset()
            for timestep in range(15):
                # env.render()
                # sleep(0.05)

                grid, goal_grid, position = parse_state(state)

                actions_probabilities = sess.run([prob_actions], feed_dict={
                    X: [np.stack([grid, goal_grid], axis=2)],
                    S1: [position[0]], 
                    S2: [position[1]]
                })
                
                action = np.argmax(actions_probabilities)
                state, reward, done, _ = env.step(action)

                if done:
                    dones += 1
                    break
        env.close()

        print("accuracy : {}/{}".format(dones, episodes))

def parse_state(state):
    start = np.argwhere(state == 2)[0]
    state[state == 2] = 0

    goal = state == 3
    state[goal] = 0

    return state, create_goal_grid(state.shape, goal), start

def create_goal_grid(shape, goal):
    goal_grid = np.zeros(shape, dtype=np.int8)
    goal_grid[goal] = 10
    return goal_grid

if __name__ == "__main__":
    main()
