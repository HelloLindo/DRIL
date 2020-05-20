from matplotlib import pyplot as plt
import numpy as np


def load_data(path, is_smooth=False):
    if is_smooth:
        return smooth(np.nan_to_num(np.load(path)), 0.90)
    else:
        return np.load(path)


def smooth(target, wight):
    smoothed = []
    last = target[0]
    for value in target:
        smoothed_val = last * wight + (1 - wight) * value
        smoothed.append(smoothed_val)
        last = smoothed_val
    return smoothed


if __name__ == '__main__':
    PLOT_REWARDS = True
    PLOT_Q_VALUES = True
    SAVE_PLOT = True
    DQN_DATA_DIR = "../data/20200515/DQN"
    DRIL_DATA_DIR = "../data/20200519/DRIL"

    plt.rcParams['savefig.dpi'] = 300
    plt.rcParams['figure.dpi'] = 300

    if PLOT_REWARDS:
        # DQN
        dqn_episode_data = load_data(DQN_DATA_DIR + "/episode_pong_dqn.npy")
        dqn_reward_data = load_data(DQN_DATA_DIR + "/reward_pong_dqn.npy", True)

        # DRIL
        dril_episode_data = load_data(DRIL_DATA_DIR + "/episode_pong_dril.npy")
        dril_reward_data = load_data(DRIL_DATA_DIR + "/reward_pong_dril.npy", True)

        fig_r, ax_r = plt.subplots()
        plt.xlabel("Episode", fontsize=18)
        plt.ylabel("Average Reward", fontsize=18)
        plt.title("Average Reward on Pong", fontsize=20)
        dqn, = plt.plot(dqn_episode_data, dqn_reward_data, color="#cc3311")
        dqn_rule, = plt.plot(dril_episode_data, dril_reward_data, color="#0077bb")
        plt.legend(handles=[dqn, dqn_rule], labels=['DQN', 'RIL'], loc='lower right')
        if SAVE_PLOT:
            # plt.savefig("./Pong_Avg_Rewards.png", transparent=True)
            plt.savefig("./Pong_Avg_Rewards.png")
        plt.show()

    if PLOT_Q_VALUES:
        # DQN
        dqn_episode_data = load_data(DQN_DATA_DIR + "/episode_pong_dqn.npy")
        dqn_q_value_data = load_data(DQN_DATA_DIR + "/q_value_pong_dqn.npy", True)

        # DRIL
        dril_episode_data = load_data(DRIL_DATA_DIR + "/episode_pong_dril.npy")
        dril_q_value_data = load_data(DRIL_DATA_DIR + "/q_value_pong_dril.npy", True)

        fig_r, ax_r = plt.subplots()
        plt.xlabel("Training Epochs", fontsize=18)
        plt.ylabel("Average Q Value", fontsize=18)
        plt.title("Average Q on Pong", fontsize=20)
        dqn, = plt.plot(dqn_episode_data, dqn_q_value_data, color="#cc3311")
        dqn_rule, = plt.plot(dril_episode_data, dril_q_value_data, color="#0077bb")
        plt.legend(handles=[dqn, dqn_rule], labels=['DQN', 'RIL'], loc='lower right')
        if SAVE_PLOT:
            # plt.savefig("./Pong_Avg_Q_Values.png", transparent=True)
            plt.savefig("./Pong_Avg_Q_Values.png")
        plt.show()
