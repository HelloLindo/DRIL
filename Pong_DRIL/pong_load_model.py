'''
    Yuling Wu's Undergraduate Thesis Experiment
    Deep Q-Networks Playing Pong Game - load trained model
    Game's Action Set: 6 action space, 0/1:freeze, 2/4:up, 3/5:down, e.g: tensor([[2]])
'''
import torch
import time
import os
import argparse

from collections import namedtuple
from itertools import count
from env.wrappers import *
from env.models import *

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

def get_args():
    parser = argparse.ArgumentParser()
    # Record Test Playing Process
    parser.add_argument('--record', default=False, action="store_true", help="Record Test Playing Process(ffmpeg required)")
    # Data Saving Directory
    parser.add_argument('--load', type=str, default="./data/20200519/DRIL/pong_model_dril.pt", help="the Path of model.pt File.")
    # Render
    parser.add_argument('--no-render', default=False, action="store_true", help="Render the Playing Process")
    # Test Episodes
    parser.add_argument('--test-episodes', type=int, default=5, help="The times of testing trained model.")
    # Device
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_known_args()[0]
    return args

def get_state(obs):
    state = np.array(obs)
    state = state.transpose((2, 0, 1))
    state = torch.from_numpy(state)
    return state.unsqueeze(0)

def test_model(args, env, policy):
    if args.record:
        env = gym.wrappers.Monitor(env, os.path.dirname(args.load) + '/videos')
    for episode in range(args.test_episodes):
        obs = env.reset()
        state = get_state(obs)
        total_reward = 0.0
        for t in count():
            action = policy(state.to(args.device)).max(1)[1].view(1, 1)

            if not args.no_render:
                env.render()
                time.sleep(0.02)

            obs, reward, done, info = env.step(action)

            total_reward += reward

            if not done:
                next_state = get_state(obs)
            else:
                next_state = None

            state = next_state

            if done:
                print("Finished Episode {} with reward {}".format(episode, total_reward))
                break

    env.close()
    return

def pong_load_model(args=None):
    if args is None:
        args = get_args()

    if not os.path.exists(args.load):
        assert False, "No Such File. Please check."

    # print loaded parameters
    directory = os.path.dirname(args.load)
    if os.path.exists(directory + "/pong_params.npy"):
        params = np.load((directory + "/pong_params.npy"), allow_pickle=True).item().__dict__
        print("======\nThe Parameters of the model is:")
        for item in params:
            print(item + ":   " + str(params[item]))
        print("======\n")

    # create environment
    env = gym.make("PongNoFrameskip-v4")
    env = make_env(env)

    load_model = DQN(n_actions=4).to(args.device)
    load_model.load_state_dict(torch.load(args.load, map_location=torch.device(args.device)))
    print("======\nBegin to Test model.\n======\n")
    test_model(args, env, load_model)

if __name__ == '__main__':
    pong_load_model(get_args())