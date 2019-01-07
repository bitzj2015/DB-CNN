import gym
import gym_pathfinding

from gym_pathfinding.games.gridworld import generate_grid, MOUVEMENT
from astar import astar
from tqdm import tqdm
import numpy as np
import operator
import itertools


def generate_dataset(size, statebatchsize, shape, grid_type="free", verbose=False):

    # if verbose:
    #     progress_bar = tqdm(total=size)

    m = shape[0]
    n = shape[1]
    images = np.zeros((size, m, n, 2), 'float32')
    #goal_images = np.zeros((size, m, n), 'int32')
    S1s = np.zeros((size, statebatchsize), 'int32')
    S2s = np.zeros((size, statebatchsize), 'int32')
    labels = np.zeros((size, statebatchsize), 'int32')

    for i in range(size):
        flag = True
        grid, start, goal = generate_grid(shape, grid_type=grid_type)
        images[i, :, :, 0] = grid
        images[i, goal[0], goal[1], 1] = 100
        count = 0
        if i % 100 == 0:
            print(i)
        while flag:

            stop = 1
            while stop:
                start_x = np.random.randint(m - 2) + 1
                start_y = np.random.randint(n - 2) + 1

                if grid[start_x, start_y] == 0:
                    stop = 0
                    start = (start_x, start_y)
            path, action_planning = compute_action_planning(grid, start, goal)
            # print("hh", start, goal, path)

            if path != False:
                for action, position in zip(action_planning, path):
                    if count < statebatchsize:
                        S1s[i, count] = position[0]
                        S2s[i, count] = position[1]
                        labels[i, count] = action
                        count += 1
                    else:
                        flag = False
                # print(start, goal)
                # print(S1s[i], S2s[i], labels[i])

        # if verbose:
        #     progress_bar.update(1)

        # if i == size - 1:
        #     if verbose:
        #         progress_bar.close()
    print(np.shape(images), np.shape(S1s), np.shape(S2s), np.shape(labels))
    print(images[0, 0:24, 0:24, 0])
    print(S1s[0])
    print(S2s[0])
    print(labels[0])
    return images, S1s, S2s, labels

    # images = []
    # S1s = []
    # S2s = []
    # labels = []

    # n = 0

    # while True:

    #     grid, start, goal = generate_grid(shape, grid_type=grid_type)
    #     path, action_planning = compute_action_planning(grid, start, goal)

    #     goal_grid = create_goal_grid(grid.shape, goal)
    #     image = np.stack([grid, goal_grid], axis=2)

    #     for action, position in zip(action_planning, path):
    #         images.append(image)
    #         S1s.append(position[0])
    #         S2s.append(position[1])
    #         labels.append(action)

    #         if verbose:
    #             progress_bar.update(1)

    #         n += 1
    #         if n >= size:
    #             if verbose:
    #                 progress_bar.close()
    #             return images, S1s, S2s, labels

    # reversed MOUVEMENT dict
ACTION = {mouvement: action for action,
          mouvement in dict(enumerate(MOUVEMENT)).items()}


def compute_action_planning(grid, start, goal):
    path = astar(grid, start, goal)

    action_planning = []
    if path != False:
        for i in range(len(path) - 1):
            pos = path[i]
            next_pos = path[i + 1]

            # mouvement = (-1, 0), (1, 0), (0, -1), (0, 1)
            mouvement = tuple(map(operator.sub, next_pos, pos))

            action_planning.append(ACTION[mouvement])

    return path, action_planning


def create_goal_grid(shape, goal):
    goal_grid = np.zeros(shape, dtype=np.int8)
    goal_grid[goal] = 10
    return goal_grid


def main():
    import joblib
    import argparse

    parser = argparse.ArgumentParser(
        description='Generate data (images, S1s, S2s, labels)')
    parser.add_argument('--out', '-o', type=str,
                        default='./data/data_64.pkl', help='Path to save the dataset')
    parser.add_argument('--size', '-s', type=int,
                        default=10000, help='Number of example')
    parser.add_argument('--statebatchsize', type=int,
                        default=64, help='Number of example')
    parser.add_argument(
        '--shape', type=int, default=[64, 64], nargs=2, help='Shape of the grid (e.g. --shape 9 9)')
    parser.add_argument('--grid_type', type=str, default='obstacle',
                        help='Type of grid : "free", "obstacle" or "maze"')
    args = parser.parse_args()

    dataset = generate_dataset(args.size, args.statebatchsize, args.shape,
                               grid_type=args.grid_type, verbose=True
                               )
    # print(dataset)

    print("saving data into {}".format(args.out))

    # np.save(args.out, dataset)
    joblib.dump(dataset, args.out)

    print("done")


if __name__ == "__main__":
    main()
