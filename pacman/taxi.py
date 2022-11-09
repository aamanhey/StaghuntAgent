#!/usr/bin/env python
# coding: utf-8

import gym
from IPython.display import clear_output
from time import sleep
import numpy as np
import random
import pprint

# core gym interface
env = gym.make("Taxi-v3").env # v2 was depricated

# rendering a random state
env.reset() # sets a random initial state
print("Frame of Taxi Game")
env.render() # renders a frame of the environment

print("Action Space {}".format(env.action_space))
print("State Space {}".format(env.observation_space))

# manually encode state
state = env.encode(3, 1, 2, 0) # (taxi row, taxi column, passenger index, destination index)
pprint.pprint("State: {}".format(state))

env.s = state
env.render()

# initial reward table
pprint.pprint("Initial Reward Table: {}".format(env.P[328]))


# Brute Force Method

def brute_force():
    env.s = 328

    epochs = 0
    penalties, reward = 0, 0

    frames = [] # for animation

    done = False

    while not done:
        action = env.action_space.sample()
        state, reward, done, info = env.step(action)

        if reward == -10:
            penalties += 1

        # Put each rendered frame into dict for animation
        frames.append({
            'frame': env.render(mode='ansi'),
            'state': state,
            'action': action,
            'reward': reward
            }
        )

        epochs += 1


    print("Timesteps taken: {}".format(epochs))
    print("Penalties incurred: {}".format(penalties))
    return [frames, epochs, penalties]

result = brute_force()
frames = result[0]

# visualize an episode

def print_frames(frames):
    for i, frame in enumerate(frames):
        clear_output(wait=True)
        print(frame['frame'])
        print(f"Timestep: {i + 1}")
        print(f"State: {frame['state']}")
        print(f"Action: {frame['action']}")
        print(f"Reward: {frame['reward']}")
        sleep(.1) # 0.01

# print_frames(frames)


"""Reinforcement Learning"""
def reinforcement_learning():
    q_table = np.zeros([env.observation_space.n, env.action_space.n])

    # Hyperparameters
    alpha = 0.1 # learning rate
    gamma = 0.6 # discount factor
    epsilon = 0.1 # exploration parameter

    # For plotting metrics
    all_epochs = []
    all_penalties = []

    for i in range(1, 100001):
        state = env.reset()

        epochs, penalties, reward, = 0, 0, 0
        done = False

        while not done:
            if random.uniform(0, 1) < epsilon:
                action = env.action_space.sample() # Explore action space
            else:
                action = np.argmax(q_table[state]) # Exploit learned values

            next_state, reward, done, info = env.step(action)

            if(done):
                all_epochs.append(epochs)
                all_penalties.append(penalties)

            old_value = q_table[state, action]
            next_max = np.max(q_table[next_state])

            new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)
            q_table[state, action] = new_value

            if reward == -10:
                penalties += 1

            state = next_state
            epochs += 1

        if i % 10000 == 0:
            clear_output(wait=True)
            print(f"Episode: {i}")

    print("Training finished.\n")
    return [q_table, all_epochs, all_penalties]

result = reinforcement_learning()
q_table = result[0]

# compare our q-values with our initial table
print("Initial Reward Table: {}".format(env.P[328]))
print("Q-learning Reward Table: {}".format(q_table[328]))

"""Evaluation"""
def run_k_episodes(k, q_table):
    total_epochs, total_penalties = 0, 0
    episodes = k

    for _ in range(episodes):
        state = env.reset()
        epochs, penalties, reward = 0, 0, 0

        done = False

        while not done:
            action = np.argmax(q_table[state])
            state, reward, done, info = env.step(action)

            if reward == -10:
                penalties += 1

            epochs += 1

        total_penalties += penalties
        total_epochs += epochs

    print(f"Results after {episodes} episodes:")
    print(f"Average timesteps per episode: {total_epochs / episodes}")
    print(f"Average penalties per episode: {total_penalties / episodes}")

# run_k_episodes(100, q_table)
