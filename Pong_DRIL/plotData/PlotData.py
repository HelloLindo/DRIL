from decimal import Decimal
from matplotlib import pyplot as plt


def read_file(file_name):
    with open(file_name, "r") as file:
        recorder = []
        for lines in file:
            recorder.append(float(Decimal(lines).quantize(Decimal('0.00'))))
    return recorder


def smooth(target, wight):
    smoothed = []
    last = target[0]
    for value in target:
        smoothed_val = last * wight + (1 - wight) * value
        smoothed.append(smoothed_val)
        last = smoothed_val
    return smoothed

PLOT_REWARDS = True
PLOT_Q_VALUES = True
SAVE_PLOT = True

if PLOT_REWARDS:
    DQN_time_line = read_file("./pong_rules_episodes.txt")
    DQN_rate = read_file("./pong_origin_avg_rewards.txt")

    DQN_rate = smooth(DQN_rate, 0.8)

    rule_time_line = read_file("./pong_rules_episodes.txt")
    rule_rate = read_file("./pong_rules_avg_rewards.txt")

    rule_rate = smooth(rule_rate, 0.8)

    plt.rcParams['savefig.dpi'] = 300
    plt.rcParams['figure.dpi'] = 300
    fig_r, ax_r = plt.subplots()
    plt.xlabel("Episode", fontsize=20)
    plt.ylabel("Average Reward", fontsize=20)
    plt.title("Average Reward on Pong", fontsize=20)
    dqn, = plt.plot(DQN_time_line, DQN_rate, color="#cc3311")
    dqn_rule, = plt.plot(rule_time_line, rule_rate, color="#0077bb")
    plt.legend(handles=[dqn, dqn_rule], labels=['DQN', 'DRIL'], loc='lower right')
    if SAVE_PLOT:
        plt.savefig("./Pong_Avg_Rewards_transBG.png", transparent=True)
    plt.show()

if PLOT_Q_VALUES:
    DQN_time_line = read_file("./pong_rules_episodes.txt")
    DQN_rate = read_file("./pong_origin_avg_q_values.txt")

    rule_time_line = read_file("./pong_rules_episodes.txt")
    rule_rate = read_file("./pong_rules_avg_q_values.txt")

    plt.rcParams['savefig.dpi'] = 300
    plt.rcParams['figure.dpi'] = 300
    fig_r, ax_r = plt.subplots()
    plt.xlabel("Training Epochs", fontsize=20)
    plt.ylabel("Average Q Value", fontsize=20)
    plt.title("Average Q on Pong", fontsize=20)
    dqn, = plt.plot(DQN_time_line, DQN_rate, color="#cc3311")
    dqn_rule, = plt.plot(rule_time_line, rule_rate, color="#0077bb")
    plt.legend(handles=[dqn, dqn_rule], labels=['DQN', 'DRIL'], loc='lower right')
    if SAVE_PLOT:
        plt.savefig("./Pong_Avg_Q_Values_transBG.png", transparent=True)
    plt.show()