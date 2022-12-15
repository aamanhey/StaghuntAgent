# Code Architecture

The code of the project is split by functionality. The main entry point is 'main.py', where the agent can be trained and tested. Each component is overviewed below:

## Components

### The Environment
The environment and related classes handle the creating and modification of the environment and enforcing the rules of the game. The 'environement.py' file contains the *StaghuntEnv* class, which represents the environment. This class holds data about the environment such as the characters present, the current timestep, the map, etc. This class has 4 main methods:
1. **init()**: The initialization method sets up the environment. If any components have not been defined, such as a character does not have an agent to represent it, the environment will assign it one based on its type. Similarly it will create a map if not provided.
2. **reset()**: The reset method resets the environment with a new, randomly set, state. The map will not change but the character positions will and each of the agents will be reset (e.g. reward set to 0).
3. **step()**: The step method transitions the environment between states. It queries a move from the agents of each of the characters, calculates the reward for each character, and then updates each agent with the reward they recieved.
4. **render()**: The render method outputs the current state of the environment to the terminal as a highlighted string.

Additionally, the environment has many utility functions that allow the environment to be queried, such as the 'get_subject' method which returns the agent that is currently being trained.

Some of the related classes include the *State* class which represents each of the states in the game. The class holds data for the string representation of a map (state id), the current map, the character positions, the current number of steps, and the position of the subject. This class also allows for the transitioning between states by moving a single character, if desired.

The *Encoder* class, which handles the conversion between the map's matrix and string representation. It also encodes character ids into numerical ids so that they may be represented in the map. This class has a test class to test the conversion of maps and character ids.

The *InteractionManager* class calculates the reward given for a particular set of characters. This means that if a rabbit and a hunter are at the same position, it returns the resulting point values for the hunter. Similar to the Encoder class, this class also has a test class for the calculation of rewards and its helper functions.

The *Registry* class holds all of the characters. Each character has an associated position, which is a tuple, and an agent. This class handles CRUD operations on the character registry.

Lastly the 'setup.py' and 'config.py' files hold data for initializing training or setting default values. The setups available have a variety of maps to choose from, including those used in [Shum et. al](https://ojs.aaai.org/index.php/AAAI/article/view/4574). The config file specifies different parameters that can be used to configure the outputs of the metrics, such as when to save them.

### Agents
The agents in this project were developed iteravely to test out different functionality of the environment. The agents defined in 'agents.py' are defined below,
- StaticAgent: A non-mobile agent
- RandomAgent: A mobile agent which moves at random
- StaghuntAgent: An agent with stag hunt attributes (e.g. type)
- ManualAgent: An agent that takes input from the terminal to move
- PreyAgent: An agent representing the stag which moves away from the hunter.
- BasicHunterAgent: The companion agent to the subject, who uses BFS to target the sag 75% of the time and the rabbit 25%.

The agents defined in 'rl_agent.py' use reinforcement learning techniques to operate. There is base *ReinforcementAgent* class which defines necessary attributes, while the *QLearningAgent* and *ApprxReinforcementAgent* use their respective techniques to train.

The ApprxReinforcementAgent class used *StaghuntExtractor*, defined in 'feature_extractor.py' in order to extract features. The cache used for the features is defined there.
