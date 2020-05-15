'''
    Yuling Wu's Undergraduate Thesis Experiment
    Dynamically Rule-Interposing Learning Playing Pong Game
    Game's Action Set: 6 action space, 0/1:freeze, 2/4:up, 3/5:down, e.g: tensor([[2]])
'''
import torch
import torch.optim as optim
import torch.nn.functional as F
import math
import random
import time
import os
import argparse

from collections import namedtuple
from itertools import count
from env.wrappers import *
from env.memory import ReplayMemory
from env.models import *
from rules.rules_detection import *
from env.decay_models import *

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

def get_args():
    parser = argparse.ArgumentParser()
    # Human Rules Intervene
    parser.add_argument('--no-dril', default=False, action="store_true", help="No DRIL")
    # Decay Model
    parser.add_argument('--decay', type=str, default="quad", help="rules-interposing decay model, [exp, quad, poly]")
    # Record Test Playing Process
    parser.add_argument('--no-record', default=False, action="store_true", help="Do Not Record Test Playing Process")
    # Data Saving Directory
    parser.add_argument('--save-dir', type=str, default="./data/" + time.strftime("%Y%m%d", time.localtime()) + "/DRIL")
    # Device
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    # Hyperparameters
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--eps-start', type=float, default=1.00)
    parser.add_argument('--eps-end', type=float, default=0.02)
    parser.add_argument('--eps-decay', type=int, default=1000000)
    parser.add_argument('--target-update', type=int, default=1000)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--initial-memory', type=int, default=50000)
    parser.add_argument('--train-episodes', type=int, default=1000)
    parser.add_argument('--render-train', default=False, action="store_true", help="Render the Training Process")
    parser.add_argument('--render-test', default=False, action="store_true", help="Render the Testing Process")
    args = parser.parse_known_args()[0]

    if args.decay == "quad":
        params = {"QUAD_DECAY_XX": -1 * (6.25 * 1e-06),
                  "QUAD_DECAY_X": (3.75 * 1e-04),
                  "QUAD_DECAY_CONST": 0.95}
        print(f"=====\nThe Rules-Interposing Decay Model is Quadratic\nParams: {params}\n=====")
    elif args.decay == "exp":
        params = {"EXP_INITIAL_OMEGA": 0.95,
                  "EXP_DECAY_RATE": 0.50,
                  "EXP_DECAY_STEPS": 400}
        print(f"=====\nThe Rules-Interposing Decay Model is Exponential\nParams: {params}\n=====")
    elif args.decay == "poly":
        params = {"POLY_DECAY_XXX": -2.708e-08,
                  "POLY_DECAY_XX": 9.375e-06,
                  "POLY_DECAY_X": -0.001542,
                  "POLY_DECAY_CONST": 0.95}
        print(f"=====\nThe Rules-Interposing Decay Model is Polynomial\nParams: {params}\n=====")
    else:
        assert False, "Please choose a supported decay model from [quad, exp, poly]"
    args.params = params
    args.memory_size = 10 * args.initial_memory
    return args

def select_action(args, state, episode):
    sample = random.random()
    eps_threshold = args.eps_end + (args.eps_start - args.eps_end) * \
                    math.exp(-1. * args.steps_done / args.eps_decay)
    args.steps_done += 1

    # Choose human rules to make action
    if not args.no_dril and args.decay == "quad" and sample < get_quad_decay(args.params, episode):
        return choose_action_by_rules(state)
    elif not args.no_dril and args.decay == "exp" and sample < get_exponential_decay(args.params, episode):
        return choose_action_by_rules(state)
    elif not args.no_dril and args.decay == "poly" and sample < get_polynomial_decay(args.params, episode):
        return choose_action_by_rules(state)
    else:
        # Choose DQN or random
        if sample > eps_threshold:
            with torch.no_grad():
                return args.policy_net(state.to(args.device)).max(1)[1].view(1, 1)
        else:
            return torch.tensor([[random.randrange(4)]], device=args.device, dtype=torch.long)

def optimize_model(args):
    if len(args.memory) < args.batch_size:
        return
    transitions = args.memory.sample(args.batch_size)
    """
    zip(*transitions) unzips the transitions into
    Transition(*) creates new named tuple
    batch.state - tuple of all the states (each state is a tensor)
    batch.next_state - tuple of all the next states (each state is a tensor)
    batch.reward - tuple of all the rewards (each reward is a float)
    batch.action - tuple of all the actions (each action is an int)    
    """
    batch = Transition(*zip(*transitions))

    actions = tuple((map(lambda a: torch.tensor([[a]], device=args.device), batch.action)))
    rewards = tuple((map(lambda r: torch.tensor([r], device=args.device), batch.reward)))

    non_final_mask = torch.tensor(
        tuple(map(lambda s: s is not None, batch.next_state)),
        device=args.device, dtype=torch.bool)

    non_final_next_states = torch.cat([s for s in batch.next_state
                                       if s is not None]).to(args.device)

    state_batch = torch.cat(batch.state).to(args.device)
    action_batch = torch.cat(actions)
    reward_batch = torch.cat(rewards)

    state_action_values = args.policy_net(state_batch).gather(1, action_batch)
    with torch.no_grad():
        mean_action_values = torch.mean(state_action_values).cpu().detach().numpy()

    next_state_values = torch.zeros(args.batch_size, device=args.device)
    next_state_values[non_final_mask] = args.target_net(non_final_next_states).max(1)[0].detach()
    expected_state_action_values = (next_state_values * args.gamma) + reward_batch

    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

    args.optimizer.zero_grad()
    loss.backward()
    for param in args.policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    args.optimizer.step()
    return mean_action_values

def get_state(obs):
    state = np.array(obs)
    state = state.transpose((2, 0, 1))
    state = torch.from_numpy(state)
    return state.unsqueeze(0)

def train(args, env, n_episodes, render=False):
    # Save records
    episode_list = []
    reward_list = []
    q_value_list = []

    q_values = []
    for episode in range(n_episodes):
        obs = env.reset()
        state = get_state(obs)
        total_reward = 0.0
        for t in count():

            action = select_action(args, state, episode)
            obs, reward, done, info = env.step(action)

            if render:
                env.render()

            total_reward += reward

            if not done:
                next_state = get_state(obs)
            else:
                next_state = None

            reward = torch.tensor([reward], device=args.device)

            args.memory.push(state, action.to('cpu'), next_state, reward.to('cpu'))
            state = next_state

            if args.steps_done > args.initial_memory:
                q_value = optimize_model(args)
                q_values.append(q_value)

                if args.steps_done % args.target_update == 0:
                    args.target_net.load_state_dict(args.policy_net.state_dict())

            if done:
                episode_list.append(episode)
                reward_list.append(total_reward)
                q_value_list.append(np.mean(q_values))
                q_values = []
                break
        if episode % 10 == 0 or episode == n_episodes-1:
            print('Total steps: {} \t Episode: {}/{} \t Total reward: {}'.format(args.steps_done, episode, t,
                                                                                 total_reward))
            np.save(args.save_dir + "/episode_pong_dril", episode_list)
            np.save(args.save_dir + "/reward_pong_dril", reward_list)
            np.save(args.save_dir + "/q_value_pong_dril", q_value_list)

    env.close()
    return

def test(args, env, n_episodes, policy, render=True):
    if not args.no_record:
        env = gym.wrappers.Monitor(env, args.save_dir + '/videos')
    for episode in range(n_episodes):
        obs = env.reset()
        state = get_state(obs)
        total_reward = 0.0
        for t in count():
            action = policy(state.to(args.device)).max(1)[1].view(1, 1)

            if render:
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

def pong_DRIL(args=None):
    if args is None:
        args = get_args()

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    # Save parameters
    np.save(args.save_dir + "/pong_params", args)
    print("=====\nParameters saved.\n=====\n")

    # create networks
    policy_net = DQN(n_actions=4).to(args.device)
    target_net = DQN(n_actions=4).to(args.device)
    target_net.load_state_dict(policy_net.state_dict())
    # setup optimizer
    optimizer = optim.Adam(policy_net.parameters(), lr=args.lr)
    # setup steps
    steps_done = 0
    # initialize replay memory
    memory = ReplayMemory(args.memory_size)

    # add into args
    args.policy_net = policy_net
    args.target_net = target_net
    args.optimizer = optimizer
    args.steps_done = steps_done
    args.memory = memory

    # create environment
    env = gym.make("PongNoFrameskip-v4")
    env = make_env(env)

    # train model
    train(args, env, args.train_episodes, render=args.render_train)
    print("=====\nTrain model finished.\n=====\n")
    # save model
    SAVE_PATH = args.save_dir + "/pong_model_dril.pt"
    torch.save(policy_net.state_dict(), SAVE_PATH)
    print("=====\nModel saved.\n=====\n")

    load_model = DQN(n_actions=4).to(args.device)
    load_model.load_state_dict(torch.load(SAVE_PATH))
    print("======\nBegin to Test model for 5 times.\n======\n")
    test(args, env, 5, load_model, render=args.render_test)

if __name__ == '__main__':
    pong_DRIL(get_args())

