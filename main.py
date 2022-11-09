import os
import pprint
import numpy as np
from prettytable import PrettyTable
from environment import StaghuntEnv
from agents import ManualAgent, BruteForceAgent, QLearningAgent

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

def create_env(map_dim, character_setup, agent):
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
    env = create_env(4, character_setup, manual_agent)

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
    env = create_env(4, character_setup, agent)

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

def get_prop(list, bound):
    count = 0
    for item in list:
        if item < bound:
            count += 1
    return (count/len(list))

def main():
    os.system('clear')

    character_setup_full = {
        "r1": (1, 1),
        "r2": (1, 1),
        "s1": (1, 1),
        "s2": (1, 1),
        "h1": (1, 1),
        "h2": (1, 1),
        "h3": (1, 1)
    }

    character_setup_2h1r1s = {
        "r1": {"position": (1, 1)},
        "s1": {"position": (1, 1)},
        "h1": {"position": (2, 2)},
        "h2": {"position": (2, 2)}
    }

    character_setup = {
        "r1": {"position": (1, 1)},
        "h1": {"position": (2, 2)}
    }

    # Setup Values
    map_length = 5
    alpha=0.1
    epsilon=0.5
    gamma=0.8
    delta = 0.001

    num_epochs = 10001
    percent_conv = 0.2 # the percentage of episodes with deltas below bound

    # Initialize Agent
    agent = QLearningAgent("h1", alpha, epsilon, gamma, delta)


    # Training Environment for Agent
    print("------Training Q-Learning Agent------")
    agent.toggleTraining()
    env = create_env(map_length, character_setup, agent)

    # training metrics
    all_epochs = []
    all_rewards = []
    all_deltas = []

    i = 0
    num_deltas_achieved = 0
    delta_percent = round(num_epochs * percent_conv)

    while (i < num_epochs) and (num_deltas_achieved < delta_percent):
        state = env.reset()
        while env.get_status():
            env.step()

        subject = env.get_subject()
        agent = subject["agent"]

        # update training metrics
        all_epochs.append(env.current_step)
        all_rewards.append(agent.reward)

        delta_i = agent.get_delta_avg()
        if delta_i < delta:
            num_deltas_achieved += 1
        all_deltas.append(delta_i)

        # reset agent's reward and convergence vars
        agent.reset()
        i += 1

    print("Stopped at {} epochs.".format(i))

    # Find at what epoch a percentage of delta was reached
    critical_indices = {"indices":[], "proportions": []}
    perc = 0
    for epoch in range(1, num_epochs):
        proportion = get_prop(all_deltas[:epoch], delta)
        if proportion >= perc:
            critical_indices["indices"].append(epoch)
            critical_indices["proportions"].append(proportion)
            perc = round(proportion, 1) + 0.1 if proportion > 0 else perc + 0.1

    # Print training results
    print("Trained over {} epochs.\nResults for {} epochs:".format(num_epochs, len(critical_indices["indices"])))

    metrics = {
        "episode" : critical_indices["indices"],
        "% < delta" : critical_indices["proportions"],
        "epochs" : [all_epochs[index] for index in critical_indices["indices"]],
        "rewards" : [all_rewards[index] for index in critical_indices["indices"]],
        "delta" : [all_deltas[index] for index in critical_indices["indices"]],
    }

    metrics_table = PrettyTable()
    for m_key in metrics.keys():
        metric = metrics[m_key]
        metrics_table.add_column(m_key, metric)
    print(metrics_table)

    print("\n------Testing Q-Learning Agent------")
    agent.toggleTraining()
    print("Agent Table Size: {}".format(len(agent.q_value)))

    test_env = create_env(map_length, character_setup, agent)
    state = test_env.reset()

    vals = agent.get_q_values(state)
    print("Agent's Q-Values for state {}:\n{}".format(state, vals))

    test_env.render()

    print("Testing..")
    while test_env.get_status():
       test_env.step()
       test_env.render()
       print("step  : {}".format(test_env.current_step))
       print("reward: {}\n".format(agent.reward))

    print("Testing Summary:\n  Agent took {} steps and earned a reward of {}.".format(env.current_step, agent.reward))

if __name__ == '__main__':
    main()
