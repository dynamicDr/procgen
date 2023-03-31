import argparse
import pickle
import queue
import threading
import time

import tensorboard
import torch
import gym
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import os

from CNN_DQN import DQNAgent
from DNN_DQN import DQN

from replays.default_replay import DefaultReplay
from replays.proportional_PER.proportional import ProportionalPER
from replays.rank_PER.rank_based import RankPER

from distutils.util import strtobool

def train(args):
    restore_dir = f"./models/{args.restore_env_name}/{args.restore_num}/{args.restore_step_k}k"
    args.restore_dir = restore_dir
    if not torch.cuda.is_available():
        args.device = "cpu"

    env = gym.make(args.env_name)
    # print(env.action_space)
    # print(env.observation_space)
    args.state_space = env.observation_space
    args.action_space = env.action_space
    args.state_dim = np.prod(env.observation_space.shape)
    args.action_dim = env.action_space.n
    # save setting
    directory = f"./models/{args.env_name}/{args.replay}/{args.number}"
    os.makedirs(directory, exist_ok=True)
    save_args(args,directory+"args.txt")

    if args.alg_name == "DNN_DQN":
        ALG = DQN
    elif args.alg_name == "CNN_DQN":
        ALG = DQNAgent
    else:
        raise Exception(f"No alg type found: {args.alg_name}")

    if args.replay == "default":
        replay_buffer = DefaultReplay(args.replay_max_size,args.batch_size)
    elif args.replay == "rank_PER":
        replay_buffer = RankPER(args.replay_max_size,args.batch_size)
    elif args.replay == "proportional_PER":
        replay_buffer = ProportionalPER(args.replay_max_size,args.batch_size)
    else:
        raise Exception(f"No replay type found: {args.replay}")

    policy = ALG(args,replay_buffer)
    if args.restore:
        policy.load(restore_dir)


    # logging variables:
    ep_reward = 0
    ep_step = 0
    total_step = 0

    writer = SummaryWriter(log_dir=f'./runs/{args.env_name}/{args.number}')

    time_queue = queue.Queue()
    for i in range(100):
        time_queue.put(0)
    # goal_queue = queue.Queue()
    # for i in range(100):
    #     goal_queue.put(0)

    # Training procedure:
    for episode in range(1, args.max_episodes + 1):
        start_time = time.time()
        state = env.reset()
        while True:
            total_step += 1
            ep_step += 1
            # select action and add exploration noise:
            action = policy.select_action(state)
            # take action in env:
            next_state, reward, done, info = env.step(action)
            if args.render:
                env.render()

            replay_buffer.add((state, action, reward, next_state, float(done)),priority=replay_buffer.max_priority())
            state = next_state
            ep_reward += reward

            # if episode is done then update policy:
            if done:
                break

        policy.update()

        # logging updates:
        writer.add_scalar("reward", ep_reward, global_step=total_step)

        # save checkpoint:
        if episode % args.save_rate == 0:
            policy.save(directory, int(total_step / 1000))

        episode_time = time.time() - start_time
        time_queue.put(episode_time)
        time_queue.get()
        # goal_queue.put(info["goal"])
        # goal_queue.get()
        if episode < time_queue.qsize():
            avg_epi_time = sum(list(time_queue.queue)) / episode
        else:
            avg_epi_time = sum(list(time_queue.queue)) / time_queue.qsize()

        print("Episode: {}\t"
              "Step: {}k\t"
              "Reward: {}\t"
              "Goal: {} \t"
              "Epi_step: {} \t"
              "Goal_in_100_Epi: {} \t"
              "Avg_Epi_Time: {} ".format(episode, int(total_step / 1000),
                                     round(ep_reward, 2),
                                     # info["goal"],
                                         None,
                                     ep_step,
                                         0,
                                     # sum(list(goal_queue.queue)),
                                     avg_epi_time))
        ep_reward = 0
        ep_step = 0

def save_args(args, file_path):
    with open(file_path, 'w') as f:
        for arg in vars(args):
            value = getattr(args, arg)
            f.write('{}={}\n'.format(arg, value))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training arguments')
    parser.register('type', 'boolean', strtobool)
    parser.add_argument('--env_name', type=str, default='procgen:procgen-coinrun-v0', help='environment name')
    parser.add_argument('--alg_name', type=str, default='DNN_DQN', help='alg name')
    parser.add_argument('--number', type=int, default=0, help='number')
    parser.add_argument('--gamma', type=float, default=0.99, help='discount for future rewards')
    parser.add_argument('--batch_size', type=int, default=128, help='num of transitions sampled from replay buffer')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--epsilon_init', type=float, default=1.0,help='initial value of epsilon for epsilon-greedy exploration')
    parser.add_argument('--epsilon_min', type=float, default=0.01, help='minimum value of epsilon')
    parser.add_argument('--epsilon_decay', type=float, default=0.995, help='decay rate of epsilon')
    parser.add_argument('--max_episodes', type=int, default=10000000000000, help='max num of episodes')
    # parser.add_argument('--max_timesteps', type=int, default=200, help='max timesteps in one episode')
    parser.add_argument('--save_rate', type=int, default=5000, help='save the check point per ? step')
    parser.add_argument('--restore', type='boolean', default=False, help='restore from checkpoint or not')
    parser.add_argument('--restore_env_name', type=str, default="", help='')
    parser.add_argument('--restore_num', type=int, default=1, help='restore number')
    parser.add_argument('--restore_step_k', type=int, default=4731, help='restore step k')
    parser.add_argument('--device', type=str, default="cuda", help='')
    parser.add_argument('--render', type='boolean', default=False, help='')
    parser.add_argument('--replay', type=str, default="default", help='')
    parser.add_argument('--replay_max_size', type=int, default=5e5, help='')
    args = parser.parse_args()
    print(args)
    train(args)
