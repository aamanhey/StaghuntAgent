import matplotlib as plt

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
