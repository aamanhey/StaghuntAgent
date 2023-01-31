import os
import sys
import time
import math
import pickle
import pprint
import matplotlib
import numpy as np
import beepy as beep

from evaluation_configs import *
from prettytable import PrettyTable
from environment import StaghuntEnv
from setup import map_init, setup_init
from feature_extractor import StaghuntExtractor
from agents import TABLE_AGENTS, ManualAgent, BruteForceAgent
from rl_agents import QLearningAgent, ApprxReinforcementAgent
from game_configs import RABBIT_VALUE, STAG_VALUE, MAX_GAME_LENGTH, maps

matplotlib.use('tkagg')
plt = matplotlib.pyplot

''' Utils '''
def create_metric_plot(num_epochs, agent_name):
    plt.ion()
    figure = plt.figure()
    ax = figure.add_subplot(111)
    ax.axis('auto')
    baseline_r = ax.plot(np.arange(num_epochs), np.full(num_epochs, RABBIT_VALUE), '--', color="lightskyblue", label="Rabbit Reward")
    baseline_s = ax.plot(np.arange(num_epochs), np.full(num_epochs, STAG_VALUE//2), ':', color="lightskyblue", label="Stag Reward")
    line1, = ax.plot([0], [0], 'b-', label="Rewards")
    line2, = ax.plot([0], [0], 'r-', label="Epochs")

    leg = plt.legend(loc='upper right')

    plt.xlim([0, num_epochs])
    plt.ylim([-30, 30])

    plt.title("Avg. Rewards and Epochs for {}".format(agent_name), fontsize=20)
    plt.xlabel("Episode #")
    plt.ylabel("Metric Avg.")
    plt.show(block=False)

    return figure, line1, line2

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

def progress(count, total, status='', count_label='Episode'):
    bar_len = 60
    filled_len = int(round(bar_len * count / float(total)))

    percents = round(100.0 * count / float(total), 1)
    bar = '=' * filled_len + '-' * (bar_len - filled_len)

    if status != '':
        status = 'Time Elapsed: ' + str(status) + ' seconds,'

    sys.stdout.write('[%s] %s%s ... %s %s: %s\r' % (bar, percents, '%', status, count_label, count))
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

    k = 10
    t_steps, t_reward = train_k_eps(env, k)

    print("Ran {} Training Episodes".format(k))
    print("k: steps reward")
    for i in range(k):
        print("{}: {} {}".format(i, t_steps[i], t_reward[i]))

''' Training '''
def train_agent(env, agent, num_epochs=10001, percent_conv=0.2, config=train_agent_config):
    config = validate_config(config, train_agent_config)

    # Training Environment for Agent
    if config["formatDisplay"]:
        print("------Training {}------".format(agent.get_name()))
        print("Training on {} epochs.".format(num_epochs))
    agent.toggleTraining(True)

    start_time = time.time()

    # training metrics
    metrics = {
        "epochs" : [],
        "rewards" : [],
        "terminal_kinds_counts" : { "TIME_STEPS" : 0, "RABBIT_CAPTURED" : 0, "STAG_CAPTURED" : 0 },
        "params": agent.get_params()
    }

    table_metrics = {
        "deltas" : [],
        "epochs ran": 0,
        "num epochs": num_epochs,
        "percent_conv": 0.2,
    }

    hasTable = (agent.get_name() in TABLE_AGENTS) and agent.use_delta

    if hasTable:
        metrics.update(table_metrics)

    episode = 0

    num_deltas_achieved = 0
    delta_percent = round(num_epochs * percent_conv) if hasTable else 1

    # Shows average rewards and epochs as a graph in realtime
    if config["showMetrics"]:
        indices = []
        avg_r = []
        avg_e = []
        figure, line1, line2 = create_metric_plot(num_epochs, agent.get_name())

    while (episode < num_epochs) and (num_deltas_achieved < delta_percent):
        if config["formatDisplay"]:
            progress(episode, num_epochs, int(time.time() - start_time))
        elif config["multipleProgressBars"]:
            progress(episode + num_epochs * config["bars"]["current"], num_epochs * config["bars"]["total"], int(time.time() - config["start_time"]))

        state = env.reset()

        while env.get_status():
            env.step()

        kind = env.get_terminal_kind()
        metrics["terminal_kinds_counts"][kind] += 1

        subject = env.get_subject()
        agent = subject["agent"]

        # update training metrics
        metrics["epochs"].append(env.current_step)
        metrics["rewards"].append(agent.reward)

        # updating plot values
        if config["showMetrics"]:
            indices.append(episode)
            avg_r.append(np.average(metrics["rewards"]))
            avg_e.append(np.average(metrics["epochs"]))

            k = num_epochs // 100 if num_epochs > 100 else 1
            if episode % k == 0:
                line1.set_xdata(indices[::k])
                line1.set_ydata(avg_r[::k])

                line2.set_xdata(indices[::k])
                line2.set_ydata(avg_e[::k])

                figure.canvas.draw()
                figure.canvas.flush_events()

        if hasTable:
            delta_i = agent.get_delta_avg()
            delta = metrics["params"]["delta"]
            if delta_i < delta:
                num_deltas_achieved += 1
            metrics["deltas"].append(delta_i)

        episode += 1

    if config["saveIntermMetrics"]:
        plt_id = get_folder_size('./results/plots')
        plt.savefig('results/plots/interm-metrics-{}.png'.format(plt_id))
        plt.ioff()

    if config["formatDisplay"]:
        print("Averages were {} epochs and {} rewards.".format(np.average(metrics["epochs"]), np.average(metrics["rewards"])))
        print("Finished training.")

    if hasTable:
        if config["saveTable"]:
            # Save Q-Table, if saveTable
            subject = env.get_subject()
            agent = subject["agent"]
            id = tableId
            if config["tableId"] is None:
                id = get_folder_size('./results/tables')
            agent.save_q_table(tableId)

        metrics["epochs ran"] = i

    metrics["training_time"] = int(time.time() - start_time)

    return metrics

''' Testing '''
def test_agent(test_env, agent, num_epochs=10, config=test_agent_config):
    config = validate_config(config, test_agent_config)

    # Testing Environment for Agent
    if config["formatDisplay"]:
        print("\n------Testing {}------".format(agent.__class__.__name__))
    agent.toggleTraining(False)

    # testing metrics
    metrics = {
        "epochs" : [],
        "rewards" : [],
        "num epochs": num_epochs,
        "params": agent.get_params()
    }

    if config["saveFrames"]:
        metrics["frames"] = {}

    for i in range(1, num_epochs + 1):
        state = test_env.reset()
        if config["formatDisplay"]:
            print("\n\nTest {} with starting state {}:".format(i, state))
            test_env.render(config["showMap"])
            print("--starting state--")
        while test_env.get_status():
           test_env.step()
           frame = test_env.render(config["showMap"])
           if config["saveFrames"]:
               test_id = "test {}".format(i)
               frames = metrics["frames"]
               if test_id in frames.keys():
                   frames[test_id].append(frame)
               else:
                   frames[test_id]= [frame]
           if config["showMap"]:
               print("step  : {}".format(test_env.current_step))
               print("reward: {}\n".format(agent.reward))

        # update training metrics
        metrics["epochs"].append(test_env.current_step)
        metrics["rewards"].append(agent.reward)

        if config["formatDisplay"]:
            print("Testing Summary:\n  Agent took {} steps and earned a reward of {}.".format(test_env.current_step, agent.reward))

    return metrics

def visualize_test_agent(test_env, agent):
    test_agent = True
    while test_agent:
        state = test_env.reset()
        time.sleep(3)
        print("New Test")
        while test_env.get_status():
            test_env.step()
            test_env.full_render()
            input("continue?")
        user_input = input("Test again?(Y/N): ")
        test_agent = (user_input in ["Y","y", "Yes", "yes"])
    test_env.stop_pygame()



def test_saved_table(test_env, agent, num_epochs=10, config=test_saved_table):
    config = validate_config(config, test_saved_table)

    status = agent.load_q_table(tableId)
    if status:
        return test_agent(test_env, agent, num_epochs, config=config)
    else:
        print("E: No table of id {} found.".format(tableId))

''' End-to-End '''
def train_and_test_agent(env, agent, num_train_epochs=None, num_test_epochs=10, percent_conv=0.2, config=train_and_test_agent_config):
    config = validate_config(config, train_and_test_agent_config)

    # Train Agent
    if num_train_epochs is None:
        num_train_epochs = 100 * (10**len(env.c_reg)) + 1

    hasTable = (agent.type in TABLE_AGENTS)
    config["saveTable"] = hasTable
    training_metrics = train_agent(env, agent, num_epochs=num_train_epochs, percent_conv=percent_conv, config=config)

    # Test Agent
    config["showMap"] = config["showTestMap"]
    testing_metrics = test_agent(env, agent, num_test_epochs, config=config)

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
    if config["showMetrics"]:
        print("Training Metrics:")
        if hasTable:
            display_metrics(crit_metrics, training_attrs)
        else:
            training_attrs.insert(0, "episode")
            training_metrics["episode"] = range(1, len(training_metrics["epochs"]) + 1)
            k = len(training_metrics["episode"]) // 10 if len(training_metrics["episode"]) >= 10 else 1
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

    if config["saveMetrics"]:
        id = get_folder_size('./results/metrics')
        filename = 'results/metrics/metric-{}'.format(id)

        print("Saving metrics as '{}'".format(filename))
        with open(filename,'wb') as fp:
            pickle.dump(metrics, fp)

    return metrics

def get_test_metrics(agent_id, setup, map, num_epochs, alpha, epsilon, gamma, config=grid_search_metrics_config):
    setup, map = setup_init("character_setup_2h1r1s", BruteForceAgent("h2"), "shum_map_A")

    # Setup Values
    map_length = get_map_length(map)

    # Initialize Agent
    agent = ApprxReinforcementAgent(agent_id, alpha, epsilon, gamma)
    env = create_env(map_length, setup, agent, map)

    # Agent Training & Metrics
    metrics = train_and_test_agent(env, agent, num_train_epochs=num_epochs, config=config)

    return metrics

def train_and_test_agent_with_params(agent_id, setup, map, num_epochs, precision):
    # Grid-Search
    M = 0.1 * precision

    alpha_range = np.arange(M, 1, M)
    epsilon_range = np.arange(M, 1, M)
    gamma_range = np.arange(M, 1, M)

    # Q-Learning Params
    delta_range = [0.1, 0.001, 0.0001, 0.00001]
    percent_conv_range = np.arange(0.1, 0.5, 0.1)

    min_epochs = 999999
    max_reward = -999999

    suboptimal_min_epochs = 999999
    suboptimal_max_reward = -999999

    optimal_metrics = []
    # second best metrics of each kind
    suboptimal_epoch_metrics = None
    suboptimal_reward_metrics = None

    final_params = { "optimal" : None, "suboptimal_epochs" : None, "suboptimal_rewards" : None}

    # Grid Search
    print("------Initiating Grid Search------")
    config = grid_search_metrics_config
    start_time = time.time()
    config["start_time"] = start_time
    total_options = len(alpha_range) * len(epsilon_range) * len(gamma_range)
    config["bars"] = { "current" : 0, "total" : total_options }
    print("Computing Test Metrics for {} options.".format(total_options))

    count = 0
    for a in alpha_range:
        for e in epsilon_range:
            for g in gamma_range:
                config["bars"]["current"] = count
                progress(count, total_options, int(time.time() - start_time))
                metrics = get_test_metrics(agent_id, setup, map, num_epochs, a, e, g)

                avgs = metrics["averages"]
                avg_epochs = avgs["epochs"]
                avg_rewards = avgs["rewards"]

                hyper_params = { "alpha" : a, "epsilon" : e, "gamma" : g }

                # Assign to optimal if metrics are better than current
                if avg_epochs < min_epochs and avg_rewards > max_reward:
                    optimal_metrics = [avgs]
                    final_params["optimal"] = [hyper_params]
                    min_epochs = avg_epochs
                    max_reward = avg_rewards
                elif avg_epochs == min_epochs and avg_rewards == max_reward:
                    optimal_metrics.append(avgs)
                    final_params["optimal"].append(hyper_params)
                elif avg_epochs < suboptimal_min_epochs and avg_rewards <= max_reward:
                    suboptimal_epoch_metrics = avgs
                    suboptimal_min_epochs = avg_epochs
                    final_params["suboptimal_epochs"] = hyper_params
                elif avg_epochs >= min_epochs and avg_rewards > suboptimal_max_reward:
                    suboptimal_reward_metrics = avgs
                    suboptimal_max_reward = avg_rewards
                    final_params["suboptimal_rewards"] = hyper_params

                count += 1

    final_metrics = {
        "optimal" : optimal_metrics,
        "suboptimal epoch" : suboptimal_epoch_metrics,
        "suboptimal reward" : suboptimal_reward_metrics,
        "hyper_params" : final_params
    }

    return final_metrics

''' Evaluation '''
def calc_delta_metrics(metrics):
     '''
     Metrics should have num epochs, deltas, and params.
     Find at what epoch a percentage of delta was reached.
     '''
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

def find_optimal_params(agent_id, precision, num_epochs, character_setup_name="character_setup_2h1r1s", map_name="shum_map_A", companion_agent=BruteForceAgent("h2")):
    setup, map = setup_init(character_setup_name, companion_agent, map_name)
    optimal_metrics = train_and_test_agent_with_params(agent_id, setup, map, num_epochs, precision)

    print("Optimal Metrics:\n", optimal_metrics)

    filename = 'results/metrics/optimal-metric-{}'.format(precision)

    print("Saving metrics as '{}'".format(filename))
    with open(filename,'wb') as fp:
        pickle.dump(optimal_metrics, fp)

    return optimal_metrics

'''
@TODO:
- Get metrics for optimization
    - What was the improvement when the hyper parameters were configured?
    - What was the improvement after the cache was emptied?
- What are the limits of the Q-Learning method?
    - At what point does it become unreasonable?
'''

''' Main '''
def main():
    os.system('clear')

    print("Approximate Agent")

    setup, map = setup_init(setup_name="character_setup_2h1r1s", map_name="shum_map_A")

    # Setup Values
    map_length = get_map_length(map)
    alpha = 0.2
    epsilon = 0.6
    gamma = 0.4
    delta = 0.001

    # Increased epochs from 10000 to 100000 and STAG_VALUE from 20 to 50

    default_epochs = {'xs' : 11, 's' : 100, 'm' : 10000, 'l' : 100000, 'xl' : 10000000, 'dynamic' : math.ceil(1000 * (10**len(setup)) + 1) }
    num_epochs = default_epochs['xs']

    # Initialize Agent
    agent_id = "h1"
    agent = ApprxReinforcementAgent(agent_id, alpha, epsilon, gamma) #, extractor=StaghuntExtractor(agent_id)

    env = create_env(map_length, setup, agent, map)

    # Agent Training & Metrics
    config = default_metrics_config
    config["saveIntermMetrics"] = True
    metrics = train_and_test_agent(env, agent, num_train_epochs=num_epochs, config=config)
    # agent.print_weights()
    # agent.print_features_info()

    visualize_test_agent(env, agent)

    # optimal_metrics = find_optimal_params("h1", 2, 10000)

# Test Updated Rendering
# @Todo: Fix rendering
def test_render():
    os.system('clear')
    print('Testing rendering')
    setup, map = setup_init(setup_name="character_setup_simple", map_name="shum_map_A")
    map_length = get_map_length(map)

    alpha = 0.2
    epsilon = 0.6
    gamma = 0.4

    agent_id = "h1"
    agent = ApprxReinforcementAgent(agent_id, alpha, epsilon, gamma)

    env = create_env(map_length, setup, agent, map)
    env.reset()
    env.full_render()
    print('done')

if __name__ == '__main__':
    main()
