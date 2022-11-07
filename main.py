import os
from environment import StaghuntEnv
from agents import StaticAgent, ManualAgent

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

    print("-----Creating Staghunt Environment-----")

    character_setup = {
        "r1": {"position": (1, 1)},
        "h1": {"position": (2, 2)}
    }

    # Setup Staghunt Environment
    # @TODO: Need to allow maps without the border walls
    env = StaghuntEnv(map_dim=4, characters=character_setup)

    # Create Random Environment
    env.reset()
    env.render()

    print("Action Space {}".format(env.action_space))
    print("State Space {}".format(env.observation_space))

    print(env.map)

    # Initialize agent
    agent = ManualAgent("h1", "h")

    #Create ineraction environment for training
    env.add_agent(agent)

    while env.get_status():
        env.step()
        env.render()

if __name__ == '__main__':
    main()
