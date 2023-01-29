import pickle
import random
import collections
import numpy as np
import matplotlib

from os.path import exists
from game_configs import STEP_COST
from agents import BasicHunterAgent
from prettytable import PrettyTable
from feature_extractor import StaghuntExtractor
from interaction_manager import InteractionManager

matplotlib.use('tkagg')
plt = matplotlib.pyplot

class ReinforcementAgent(BasicHunterAgent):
    def __init__(self, id, alpha=0.1, epsilon=0.5, gamma=0.8, delta=0.001):
        BasicHunterAgent.__init__(self, id)
        '''
        alpha    - learning rate
        epsilon  - exploration rate
        gamma    - discount factor
        delta    - convergence factor
        '''
        self.alpha = float(alpha)
        self.epsilon = float(epsilon)
        self.gamma = float(gamma)

        self.inTraining = False

    def get_params(self):
        params = {
            "aplha"   : self.alpha,
            "epsilon" : self.epsilon,
            "gamma"   : self.gamma
        }
        return params

    # Training Methods
    def reset(self):
        self.reward = 0

    def training_complete(self):
        return (not self.inTraining)

    def toggleTraining(self, val=None):
        if val is not None:
            self.inTraining = val
        else:
            self.inTraining = not self.inTraining

    # Policy Methods
    def get_utility(self, state, action):
        raise NotImplementedError("E: You have not defined a utility function for the {} agent.".format(self.id))

    def calc_optimal_moves(self, state):
        # Gives optimal move at a state
        max_val = None
        moves = self.generate_valid_moves(state.positions[self.id])
        optimal_moves = []
        move_data = {}
        for move in moves:
            q_val = self.get_utility(state, move)
            if max_val is None or q_val > max_val:
                max_val = q_val
                optimal_moves = [move]
                move_data = {q_val : [move]}
            elif q_val == max_val:
                optimal_moves.append(move)
                move_data[q_val].append(move)
        return optimal_moves, move_data

    def use_policy(self, state):
        optimal_moves, move_data = self.calc_optimal_moves(state)
        return random.choice(optimal_moves)

    def calc_max_utility(self, state):
        # Gives max utility of possible moves at a state, equating the value of that state
        max_utility = None
        moves = self.generate_valid_moves(state.positions[self.id])
        for move in moves:
            utility = self.get_utility(state, move)
            if max_utility is None or utility > max_utility:
                max_utility = utility
        return max_utility

    def greedy_sample(self, state):
        # Epsilon Greedy
        if random.uniform(0, 1) < self.epsilon:
            moves = self.generate_valid_moves(state.positions[self.id])
            move = random.choice(moves)
        else:
            move = self.use_policy(state)
        return move

    def get_move(self, state):
        # Gives move based on Policy
        self.map = state.map.copy()

        # Sample actions
        if self.inTraining:
            move = self.greedy_sample(state)
        else:
            move = self.use_policy(state)

        return move

    def step(self, state, action, next_state, reward):
        final_reward = reward
        if reward == 0 and self.type == "h":
            final_reward = STEP_COST

        if self.inTraining:
            # Update policy
            pass

        self.reward += final_reward

class QLearningAgent(ReinforcementAgent):
    def __init__(self, id, alpha, epsilon, gamma, delta=0.001):
        ReinforcementAgent.__init__(self, id, alpha, epsilon, gamma)
        self.inTraining = False
        self.q_value = {}

        # Convergence Methods
        self.use_delta = False
        self.delta = float(delta)
        self.hasConverged = False
        self.deltas = []

    def get_params(self):
        params = {
            "aplha"   : self.alpha,
            "epsilon" : self.epsilon,
            "gamma"   : self.gamma,
            "delta"   : self.delta
        }
        return params

    # Convergence Methods
    def set_use_delta(self, val):
        self.use_delta = val

    def get_delta_avg(self):
        return np.average(self.deltas)

    def calc_delta(self, q_i, q_j):
        delta = (q_i - q_j)
        divisor = q_j
        if q_i == 0 and q_j == 0:
            divisor = 1
        elif q_i != 0 and q_j == 0:
            divisor = q_i
        return abs(delta/divisor)

    def save_q_table(self, id=None):
        filename = 'results/tables/q-table-{}'.format(id)

        print("Saving table as '{}'".format(filename))
        with open(filename,'wb') as fp:
            pickle.dump(self.q_value, fp)

    def load_q_table(self, id=None):
        filename = 'results/tables/q-table-{}'.format(id)
        file_exists = exists(filename)
        if file_exists:
            with open(filename,'rb') as fp:
                self.q_value = pickle.load(fp)
            return True
        return False

    # Training Methods

    def reset(self):
        self.reward = 0
        self.hasConverged = False
        self.deltas = []

    def training_complete(self):
        if self.use_delta:
            return self.hasConverged
        return (not self.inTraining)

    # Q-Value Methods
    def init_q(self, map_id, action):
        c = False
        if map_id not in self.q_value.keys():
            self.q_value[map_id] = {}
            c = True
        if action not in self.q_value[map_id].keys():
            self.q_value[map_id][action] = 0
            c = True
        return c

    def get_q_values(self, state_id):
        return self.q_value[state_id]

    def get_utility(self, state, action):
        map_id = self.encoder.encode(state.map)
        self.init_q(map_id, action)
        return self.q_value[map_id][action]

    def print_q_table(self):
        print("Q-Table for {}:".format(self.id))
        for state_id in self.q_value.keys():
            values = self.q_value[state_id]
            print("{}:".format(state_id))
            for val in values:
                print("- {}: {}".format(val, round(self.q_value[state_id][val], 4)))

    def step(self, state, action, next_state, reward):
        # Update agent logic
        final_reward = reward
        if reward == 0 and self.type == "h":
            final_reward = STEP_COST

        if self.inTraining:
            feedback = final_reward + self.gamma * self.calc_max_utility(next_state)
            map_id = self.encoder.encode(state.map)
            self.init_q(map_id, action)

            # Update Q-Values
            old_q_val = self.q_value[map_id][action]
            new_q_val = (1.0 - self.alpha) * self.get_utility(state, action) + self.alpha * feedback
            self.q_value[map_id][action] = new_q_val

            # Update convergence metrics
            delta = self.calc_delta(new_q_val, old_q_val)
            self.deltas.append(round(delta, len(str(delta)) - 1))
            if self.get_delta_avg() < self.delta:
                self.hasConverged = True

        self.reward += final_reward

class ApprxReinforcementAgent(ReinforcementAgent):
    def __init__(self, id, alpha, epsilon, gamma, extractor=None):
        ReinforcementAgent.__init__(self, id, alpha=alpha, epsilon=epsilon, gamma=gamma)
        self.feature_extractor = extractor if extractor is not None else StaghuntExtractor(id)
        self.weights = collections.Counter()
        self.action_history = []
        self.showMetrics = False

    def reset(self):
        self.reward = 0
        self.action_history = []

    # Metric Methods
    def initMetrics(self, show=False):
        self.showMetrics = show
        if show:
            self.indices = []
            self.q_metric = []
            num_epochs = 1000001
            self.max_i = 10
            self.q_min = -0.5
            self.q_max = 0
            figure, line1 = self.create_metric_plot(num_epochs, -10)
            self.figure = figure
            self.line1 = line1

    def create_metric_plot(self, num_epochs, baseline_val):
        plt.ion()
        figure = plt.figure()
        ax = figure.add_subplot(111)
        ax.axis('auto')
        baseline_q = ax.plot(np.arange(num_epochs), np.full(num_epochs, baseline_val), '--', color="lightskyblue", label="Q-Value Lower Bound")
        line1, = ax.plot([0], [0], 'b-', label="Q-Value")

        leg = plt.legend(loc='upper right')

        plt.xlim([0, self.max_i])
        plt.ylim([self.q_min, self.q_max])

        plt.title("Q-Values", fontsize=20)
        plt.xlabel("Episode #")
        plt.ylabel("Value")
        plt.show(block=False)

        return figure, line1

    def redraw_metrics(self, max_next_val, k):
        self.indices.append(len(self.indices))
        self.q_metric.append(max_next_val)
        self.line1.set_xdata(self.indices[::k])
        self.line1.set_ydata(self.q_metric[::k])

        if max_next_val < self.q_min:
            self.q_min = max_next_val
        if max_next_val > self.q_max:
            self.q_max = max_next_val

        if len(self.indices) >= self.max_i - 1:
            self.max_i += k

        plt.xlim([0, self.max_i])
        plt.ylim([self.q_min, self.q_max])

        self.figure.canvas.draw()
        self.figure.canvas.flush_events()

    # Approximation Display Data Methods
    def get_weights(self):
        return self.weights

    def print_weights(self):
        weight_table = PrettyTable()
        weight_table.field_names = ["Feature", "Weight"]
        for w_key in self.weights.keys():
                weight_table.add_row([w_key, self.weights[w_key]])
        print("Weights Table")
        print(weight_table)

    def print_features_info(self):
        self.feature_extractor.print_cache()

    # Approximation Methods

    def init_extractor_map(self, map):
        # @TODO: Check to see if map layout is the same as preprocessed one
        if not self.feature_extractor.pre_processed:
            self.feature_extractor.pre_process_map(map)

    def get_utility(self, state, action):
        new_state = state.move_character(self.id, action, self.encoder)
        features = self.feature_extractor.get_features(new_state)
        utility = 0
        for feature in features.keys():
            utility += self.weights[feature] * features[feature]
        return utility

    def update_weights(self, state, action, reward, max_next_val):
        new_state = state.move_character(self.id, action, self.encoder, set_as_state=False)
        features = self.feature_extractor.get_features(new_state)
        diff = (reward + (self.gamma * max_next_val) - self.get_utility(state, action))

        # update weights using delta-rule
        for feature in features.keys():
            self.weights[feature] += self.alpha * diff * features[feature]

    def get_move(self, state):
        # Gives move based on Apprx. Q-Value
        self.map = state.map.copy()
        self.init_extractor_map(state.map)

        # Sample actions
        if self.inTraining:
            move = self.greedy_sample(state)
        else:
            move = self.use_policy(state)

        self.action_history.append(move)
        return move

    def step(self, state, action, next_state, reward):
        reward -= 1 if (reward == 0 and self.type == "h") else 0

        # find the max Q-value for the next optimal action
        maxqnext = self.calc_max_utility(next_state)

        if self.inTraining:
            k = 100
            if self.showMetrics and len(self.indices) % k == 0:
                self.redraw_metrics(maxqnext, k)

            self.update_weights(state, action, reward, maxqnext)

        self.reward += reward
