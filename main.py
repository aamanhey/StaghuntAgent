import os
import numpy as np
from environment import StaghuntEnv
from agents import StaticAgent, ManualAgent, BruteForceAgent, ProximityAgent

def train_ep(env):
    # agent should already be added to the environment
    env.reset()
    while env.get_status():
        env.step()
    subject = env.get_subject()
    agent = subject["agent"]
    r = agent.reward
    agent.reset()
    return (env.current_step, r)

def train_k_eps(env, k):
    rewards = []
    steps = []
    for i in range(k):
        s, r = train_ep(env)
        steps.append(s)
        rewards.append(r)
    return (steps, rewards)

def create_training_env(map_dim, character_setup, agent):
    # Setup Staghunt Environment
    # @TODO: Need to allow maps without the border walls
    env = StaghuntEnv(map_dim, characters=character_setup)

    # Create ineraction environment for training
    env.set_subject(agent.id)
    env.add_agent(agent)

    return env

def display_training_env(map_dim, character_setup, agent):
    # Setup Staghunt Environment
    print("-----Creating Staghunt Environment-----")

    # @TODO: Need to allow maps without the border walls
    env = StaghuntEnv(map_dim, characters=character_setup)

    # Create Random Environment
    print("Game Space:")
    env.reset()
    env.render()

    print("Action Space {}".format(env.action_space))
    print("State Space {}".format(env.observation_space))

    print(env.map)

    env.set_subject(agent.id)
    print("Subject: {}".format(env.get_subject()))

    # Create ineraction environment for training
    env.add_agent(agent)

    # Let agent interact with the environment
    print("-----Training-----")

    while env.get_status():
        env.step()
        env.render()
        print("step  : {}".format(env.current_step))
        print("reward: {}\n".format(agent.reward))

    print("Training Summary:\nAgent took {} steps and earned a reward of {}.".format(env.current_step, agent.reward))

def manual():
    # Initialize agent
    manual_agent = ManualAgent("h1", "h")

    display_training_env(4, character_setup, manual_agent)

    # Create training environment with agent
    env = create_training_env(4, character_setup, manual_agent)

    while env.get_status():
        env.step()
    subject = env.get_subject()
    agent = subject["agent"]
    r = agent.reward

    print("Training Summary:\nAgent took {} steps and earned a reward of {}.".format(env.current_step, agent.reward))

def brute_force():
    os.system('clear')

    character_setup = {
        "r1": {"position": (1, 1)},
        "h1": {"position": (2, 2)}
    }

    agent = BruteForceAgent("h1")
    env = create_training_env(4, character_setup, agent)

    epochs = 0
    penalties, reward = 0, 0

    done = False

    while env.get_status():
        env.step()
        epochs += 1

    subject = env.get_subject()
    agent = subject["agent"]
    r = agent.reward
    agent.reset()

    k = 10
    t_steps, t_reward = train_k_eps(env, k)

    print("Ran {} Training Episodes".format(k))
    print("k: steps reward")
    for i in range(k):
        print("{}: {} {}".format(i, t_steps[i], t_reward[i]))

# List Results
def list_results(k, t_steps, t_reward):
    print("Ran {} Training Episodes".format(k))
    print("k: steps reward")
    for i in range(k):
        print("{}: {} {}".format(i, t_steps[i], t_reward[i]))

# Average Results
def avg_results(t_steps, t_reward):
    steps = np.average(t_steps)
    reward = np.average(t_reward)
    print("Training Summary:\nOn average, the agent took {} steps and earned a reward of {}.".format(steps, reward))

def main():
    os.system('clear')

    character_setup_m = {
        "r1": (1, 1),
        "r2": (1, 1),
        "s1": (1, 1),
        "s2": (1, 1),
        "h1": (1, 1),
        "h2": (1, 1),
        "h3": (1, 1)
    }

    character_setup3 = {
        "r1": {"agent":None, "position": (1, 1)},
        "h1": {"agent":None, "position": (4, 3)}
    }

    character_setup = {
        "r1": {"position": (1, 1)},
        "h1": {"position": (2, 2)}
    }

    # Initialize agent
    manual_agent = ManualAgent("h1", "h")

    display_training_env(4, character_setup, manual_agent)

    # Create training environment with agent
    env = create_training_env(4, character_setup, manual_agent)

    while env.get_status():
        env.step()

if __name__ == '__main__':
    result = brute_force()
    # main()
