import os
import sys
import time
import pickle
import pprint
import numpy as np
import beepy as beep
from setup import *
from prettytable import PrettyTable
from environment import StaghuntEnv
from multiprocessing.pool import Pool
from agents import TABLE_AGENTS, ManualAgent, BruteForceAgent, QLearningAgent, ApprxQLearningAgent

''' Setup '''

def create_env(map_dim, character_setup, agent, custom_map=None):
    # Setup Staghunt Environment
    env = StaghuntEnv(map_dim, characters=character_setup, map=custom_map)

    # Create ineraction environment for training
    env.set_subject(agent.id)
    env.add_agent(agent)

    return env

''' Non-Learning Agents '''

def manual():
    # Initialize agent
    manual_agent = ManualAgent("h1", "h")

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

''' Training '''

def train_agent(env, agent, num_epochs=10001, percent_conv=0.2, saveTable=False, tableId=None, showMap=False):
    # Training Environment for Agent
    print("------Training Q-Learning Agent------")
    print("Training on {} epochs.".format(num_epochs))
    agent.toggleTraining(True)

    # training metrics
    metrics = {
        "epochs" : [],
        "rewards" : [],
        "params": agent.get_params()
    }

    table_metrics = {
        "deltas" : [],
        "epochs ran": 0,
        "num epochs": num_epochs,
        "percent_conv": 0.2,
    }

    hasTable = (agent.model in TABLE_AGENTS) and agent.use_delta

    if hasTable:
        metrics.update(table_metrics)

    i = 0

    num_deltas_achieved = 0
    delta_percent = round(num_epochs * percent_conv) if hasTable else 1

    while (i < num_epochs) and (num_deltas_achieved < delta_percent):
        progress(i, num_epochs)
        state = env.reset()
        while env.get_status():
            # @TODO: Show current map during training
            if showMap:
                str = env.render(printToScreen=False)
            env.step()

        subject = env.get_subject()
        agent = subject["agent"]

        # update training metrics
        metrics["epochs"].append(env.current_step)
        metrics["rewards"].append(agent.reward)

        if hasTable:
            delta_i = agent.get_delta_avg()
            delta = metrics["params"]["delta"]
            if delta_i < delta:
                num_deltas_achieved += 1
            metrics["deltas"].append(delta_i)

        # reset agent's reward and convergence vars
        agent.reset()
        i += 1

    print("Finished training.")

    if hasTable:
        if saveTable:
            # Save Q-Table, if saveTable
            subject = env.get_subject()
            agent = subject["agent"]
            id = tableId
            if tableId is None:
                id = get_folder_size('./tables')
            agent.save_q_table(tableId)

        metrics["epochs ran"] = i

    return metrics

''' Testing '''

def test_agent(test_env, agent, num_epochs=10, showMap=True, saveFrames=False):
    print("\n------Testing Q-Learning Agent------")
    agent.toggleTraining(False)

    # testing metrics
    metrics = {
        "epochs" : [],
        "rewards" : [],
        "num epochs": num_epochs,
        "params": agent.get_params()
    }

    if saveFrames:
        metrics["frames"] = {}

    for i in range(1, num_epochs + 1):
        state = test_env.reset()
        print("\n\nTest {} with starting state {}:".format(i, state))
        test_env.render(showMap)
        print("--starting state--")
        while test_env.get_status():
           test_env.step()
           frame = test_env.render(showMap)
           if saveFrames:
               test_id = "test {}".format(i)
               frames = metrics["frames"]
               if test_id in frames.keys():
                   frames[test_id].append(frame)
               else:
                   frames[test_id]= [frame]
           if showMap:
               print("step  : {}".format(test_env.current_step))
               print("reward: {}\n".format(agent.reward))

        # update training metrics
        metrics["epochs"].append(test_env.current_step)
        metrics["rewards"].append(agent.reward)

        print("Testing Summary:\n  Agent took {} steps and earned a reward of {}.".format(test_env.current_step, agent.reward))

    return metrics

def test_saved_table(test_env, agent, num_epochs=10, showMap=True, saveFrames=False, tableId=None):
    status = agent.load_q_table(tableId)
    if status:
        return test_agent(test_env, agent, num_epochs, showMap, saveFrames)
    else:
        print("E: No table of id {} found.".format(tableId))

''' End-to-End '''
def train_and_test_agent(env, agent, num_train_epochs=None, num_test_epochs=10, percent_conv=0.2, showMetrics=True, saveMetrics=True, showMap=False):
    # Train Agent
    if num_train_epochs is None:
        num_train_epochs = 100 * (10**len(env.c_reg)) + 1

    hasTable = (agent.type in TABLE_AGENTS)
    training_metrics = train_agent(env, agent, num_epochs=num_train_epochs, percent_conv=percent_conv, saveTable=hasTable, showMap=showMap)

    # Test Agent
    testing_metrics = test_agent(env, agent, num_test_epochs, showMap=showMap, saveFrames=True)

    # Compute Training Metrics
    training_attrs = ["epochs", "rewards"]

    if hasTable:
        training_attrs.append("deltas")
        critical_indices = calc_delta_metrics(training_metrics)
        crit_metrics = get_crit_metrics(training_metrics, training_attrs, critical_indices["indices"])
        crit_metrics["indices"] = critical_indices["indices"]
        crit_metrics["proportions"] = critical_indices["proportions"]

    # Index Testing Metrics
    testing_metrics["test"] = range(1, len(testing_metrics["epochs"]) + 1)
    testing_attrs = ["test", "epochs", "rewards"]

    # Calculate Averages of Testing
    averages = calculate_avgs(testing_metrics, testing_attrs)

    # Display Metrics
    if showMetrics:
        print("Training Metrics:")
        if hasTable:
            display_metrics(crit_metrics, training_attrs)
        else:
            training_attrs.insert(0, "episode")
            training_metrics["episode"] = range(1, len(training_metrics["epochs"]) + 1)
            k = len(training_metrics["episode"]) // 10
            display_metrics(training_metrics, training_attrs, k)
        print("Testing Metrics:")
        display_metrics(testing_metrics, testing_attrs)
        print("Metric Averages:")
        display_avgs(averages)

    metrics = {
        "training" : training_metrics,
        "testing" : testing_metrics,
        "averages" : averages
    }

    if hasTable:
        table_metrics = {
            "critical indices" : critical_indices,
            "crit. metrics": crit_metrics
        }
        metrics.update(table_metrics)

    if saveMetrics:
        id = get_folder_size('./metrics')
        filename = 'metrics/metric-{}'.format(id)

        print("Saving metrics as '{}'".format(filename))
        with open(filename,'wb') as fp:
            pickle.dump(metrics, fp)

    return metrics

def get_test_metrics(a, e, g, d, n, p, map_length, character_setup, map):
    agent = QLearningAgent("h1", a, e, g, d)
    env = create_env(map_length, character_setup, agent, map)
    metrics = train_and_test_agent(env, agent, n, 50, p, False, False).copy()
    return metrics

def train_and_test_agent_with_params(setup_data):
    character_setup = setup_data["character_setup"]
    map_length = setup_data["map_length"]
    map = setup_data["map"]
    precision = setup_data["precision"]

    val = 0.1 * precision

    alpha_range = np.arange(val, 1, val)
    epsilon_range = np.arange(val, 1, val)
    gamma_range = np.arange(val, 1, val)
    delta_range = [0.1, 0.001, 0.0001, 0.00001]
    percent_conv_range = np.arange(0.1, 0.5, 0.1)

    num_train_epochs_range = [1000, 10000, 100000]

    min_epochs = 999999
    max_reward = -999999

    suboptimal_min_epochs = 999999
    suboptimal_max_reward = -999999

    optimal_metrics = []
    # second best metrics of each kind
    suboptimal_epoch_metrics = None
    suboptimal_reward_metrics = None

    total_options = len(alpha_range) * len(epsilon_range) * len(gamma_range) * len(delta_range) * len(percent_conv_range) * len(num_train_epochs_range)

    # Grid Search
    print("------Initiating Grid Search------")
    print("Computing Test Metrics for {} options.".format(total_options))

    count = 0
    for n in num_train_epochs_range:
        for d in delta_range:
            for p in percent_conv_range:
                for a in alpha_range:
                    for e in epsilon_range:
                        for g in gamma_range:
                            progress(count, total_options)
                            metrics = get_test_metrics(a, e, g, d, n, p, map_length, character_setup, map)

                            avgs = metrics["averages"]
                            avg_epochs = avgs["epochs"]
                            avg_rewards = avgs["rewards"]

                            # Assign to optimal if metrics are better than current
                            if avg_epochs < min_epochs and avg_rewards > max_reward:
                                optimal_metrics = [metrics]
                                min_epochs = avg_epochs
                                max_reward = avg_rewards
                            elif avg_epochs == min_epochs and avg_rewards == max_reward:
                                optimal_metrics.append(metrics)
                            elif avg_epochs < suboptimal_min_epochs and avg_rewards <= max_reward:
                                suboptimal_epoch_metrics = metrics
                                suboptimal_min_epochs = avg_epochs
                            elif avg_epochs >= min_epochs and avg_rewards > suboptimal_max_reward:
                                suboptimal_reward_metrics = metrics
                                suboptimal_max_reward = avg_rewards

                            count += 1

    final_metrics = {
        "optimal" : optimal_metrics,
        "suboptimal epoch" : suboptimal_epoch_metrics,
        "suboptimal reward" : suboptimal_reward_metrics
    }

    return final_metrics

''' Evaluation '''

def calc_delta_metrics(metrics):
     # metrics should have num epochs, deltas, and params
     # Find at what epoch a percentage of delta was reached
     critical_indices = {"indices":[], "proportions": []}
     perc = 0
     for epoch in range(1, metrics["num epochs"]):
         proportion = get_prop(metrics["deltas"][:epoch], metrics["params"]["delta"])
         if proportion >= perc:
             critical_indices["indices"].append(epoch)
             critical_indices["proportions"].append(proportion)
             perc = round(proportion, 1) + 0.1 if proportion > 0 else perc + 0.1

     return critical_indices

def get_crit_metrics(metrics, attrs, critical_indices):
    crit_metrics = {}
    for metric in attrs:
        if metric in metrics.keys():
            crit_metrics[metric] = get_sub_array(metrics[metric], critical_indices)
    return crit_metrics

def display_metrics(metrics, attrs, k=1):
    # attrs represents a list of metrics to be shown
    metrics_table = PrettyTable()
    for metric in attrs:
        if metric in metrics.keys():
            metrics_table.add_column(metric, metrics[metric][::k])
    print(metrics_table)
    return metrics_table

def display_avgs(averages):
    average_table = PrettyTable(["Metric", "Average"])
    for a_key in averages.keys():
        average_table.add_row([a_key, averages[a_key]])
    print(average_table)
    return average_table

def calculate_avgs(testing_metrics, testing_attrs):
    averages = {}
    for attr in testing_metrics.keys():
        if (attr in testing_attrs) and attr != "test":
            averages[attr] = np.average(testing_metrics[attr])
    return averages

def find_optimal_params(character_setup, map_length, map, precision):
    setup_data = {
        "character_setup" : character_setup,
        "map_length" : map_length,
        "map" : map,
        "precision" : precision
    }

    optimal_metrics = train_and_test_agent_with_params(setup_data)

    print("Optimal Metrics:\n", optimal_metrics)

    filename = 'metrics/optimal-metric-{}'.format(precision)

    print("Saving metrics as '{}'".format(filename))
    with open(filename,'wb') as fp:
        pickle.dump(optimal_metrics, fp)

''' Utils '''
def get_map_length(map):
    if len(map) > len(map[0]):
        return len(map)
    return len(map[0])

def get_folder_size(folder):
    dir_path = folder
    count = 0
    for path in os.listdir(dir_path):
        # check if current path is a file
        if os.path.isfile(os.path.join(dir_path, path)):
            count += 1
    return count

def progress(count, total, status=''):
    bar_len = 60
    filled_len = int(round(bar_len * count / float(total)))

    percents = round(100.0 * count / float(total), 1)
    bar = '=' * filled_len + '-' * (bar_len - filled_len)

    sys.stdout.write('[%s] %s%s ...%s\r' % (bar, percents, '%', status))
    sys.stdout.flush()

def get_prop(list, bound):
    count = 0
    for item in list:
        if item < bound:
            count += 1
    return (count/len(list))

def get_sub_array(arr, indices):
    return [arr[index] for index in indices]

def list_results(k, t_steps, t_reward):
    print("Ran {} Training Episodes".format(k))
    print("k: steps reward")
    for i in range(k):
        print("{}: {} {}".format(i, t_steps[i], t_reward[i]))

def avg_results(t_steps, t_reward):
    steps = np.average(t_steps)
    reward = np.average(t_reward)
    print("Training Summary:\nOn average, the agent took {} steps and earned a reward of {}.".format(steps, reward))

def print_frames(frames_dict, fps=1, clear=False):
    for f_key in frames_dict.keys():
        frames = frames_dict[f_key]
        user_input = input("Do you want to see test {}?".format(f_key))
        if user_input in ["yes", "yeah", "sure", "i guess", "1", "Y", "y"]:
            for frame in frames:
                if clear:
                    os.system('clear')
                sys.stdout.write(frame)
                sys.stdout.flush()
                time.sleep(fps)

def main():
    os.system('clear')

    character_setup = character_setup_simple
    map = custom_map

    # Setup Values
    map_length = get_map_length(map)
    alpha = 0.1
    epsilon = 0.5
    gamma = 0.8
    delta = 0.001

    num_epochs = 100 * (10**len(character_setup)) + 1

    # Initialize Agent
    agent = QLearningAgent("h1", alpha, epsilon, gamma, delta)
    env = create_env(map_length, character_setup, agent, custom_map)

    metrics = train_and_test_agent(env, agent, num_train_epochs=num_epochs, showMap=True)

if __name__ == '__main__':
    main()
