# Replication Instructions

These are the instructions needed to run the project. To begin, go to [github](https://github.com/aamanhey/StaghuntAgent) and download a version of the repository.
If you are unfamiliar to this process you may refer to the [documentation on cloning repos](https://docs.github.com/en/repositories/creating-and-managing-repositories/cloning-a-repository).

## Requirements
A 'requirements.txt' file is provided with the project in order to help with the instillation of dependencies.
In order to download, use the command `pip install -r /path/to/requirements.txt` in the directory of the project.

## Running the Code
The entry point of the code is the 'main.py' file.
This file should be run, using `python3 main.py` in order to train and test the agent.
The parameters of the environment can be changed in the 'main()' method of the file, which defines parameters such as the number of epochs, learning rate, etc.
Different configurations can be made in either the 'config.py' file or by defining a dictionary using the 'default' template provided.
This controls operations such as the live updates to the training plot.
*Note:* When the training plot is exited, Python will through a quit/exit error, but this does not break or change anything.
This error is currently in the process of being remedied.

By default, the 'train_and_test_agent' method is used, which will go through the process of training an agent, from start to finish.
However, if certain functionality would like to be tested, individual training and testing methods can be called.
Additionally, if the user wants to swtich the agent, they are able to do so by instantiating an agent from the 'agent.py' or 'rl_agent.py' files.
