import torch
import torch.nn as nn
import torch.optim as optim
import random
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from gym.vector.utils import spaces


class DQN(nn.Module):
    def __init__(self,
                 observation_space: spaces.Box,
                 action_space: spaces.Discrete):
        super().__init__()
        print(observation_space)
        print(action_space)
        assert type(
            observation_space) == spaces.Box, 'observation_space must be of type Box'
        assert len(
            observation_space.shape) == 3, 'observation space must have the form channels x width x height'
        assert type(
            action_space) == spaces.Discrete, 'action_space must be of type Discrete'

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=4, stride=2),
            nn.ReLU()
        )

        self.fc = nn.Sequential(
            nn.Linear(in_features=32*9*9 , out_features=256),
            nn.ReLU(),
            nn.Linear(in_features=256, out_features=action_space.n)
        )

    def forward(self, x):
        conv_out = self.conv(x).reshape(x.size()[0],-1)
        return self.fc(conv_out)

# class DQNAgent:
#     def __init__(self, args, replay_buffer):
#         self.args = args
#         self.epsilon = self.args.epsilon_init
#         self.q_network = self._build_model().to(self.args.device)
#         self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.args.lr)
#         self.loss_fn = nn.MSELoss()
#         self.replay_buffer = replay_buffer
#
#     def _build_model(self):
#         # CNN模型定义
#         model = DQN(self.args.state_space,self.args.action_space)
#         return model
#
#     def select_action(self, state):
#         if random.random() < self.epsilon:
#             action = random.randint(0, self.args.action_dim - 1)
#         else:
#             state = torch.FloatTensor(state).unsqueeze(0).to(self.args.device)
#             q_values = self.q_network(state)
#             _, action = torch.max(q_values, 1)
#             action = action.item()
#         return action
#
#     def update(self):
#         # Sample a batch of transitions from replay buffer:
#         if self.replay_buffer.size < self.args.batch_size:
#             return
#         sample = self.replay_buffer.sample()
#
#         if sample is None:
#             return
#         state, action, reward, next_state, done, weights, indices = sample
#         if state is None:
#             return
#
#         state = torch.FloatTensor(state).to(self.args.device)
#         action = torch.LongTensor(action).to(self.args.device)
#         reward = torch.FloatTensor(reward).reshape((self.args.batch_size, 1)).to(self.args.device)
#         next_state = torch.FloatTensor(next_state).to(self.args.device)
#         done = torch.FloatTensor(done).reshape((self.args.batch_size, 1)).to(self.args.device)
#
#         state = state.transpose(3, 2).transpose(2, 1)
#         next_state = next_state.transpose(3, 2).transpose(2, 1)
#         print(state.shape)
#         q_values = self.q_network(state)
#         next_q_values = self.q_network(next_state)
#         next_q_state_value, _ = torch.max(next_q_values, 1)
#         target_q_values = reward + (1 - done) * self.args.gamma * next_q_state_value.detach()
#         q_value = q_values.gather(1, action.unsqueeze(1)).squeeze(1)
#
#         loss = self.loss_fn(q_value, target_q_values)
#
#         self.optimizer.zero_grad()
#         loss.backward()
#         self.optimizer.step()
#
#         return loss.item()
#
#     def update_epsilon(self):
#         self.epsilon = max(self.epsilon * self.args.epsilon_decay, self.args.epsilon_min)
#
#     def save(self, dir, step):
#         step = str(step)
#         torch.save(self.q_network.state_dict(), dir)
#
#
#     def load(self, dir):
#         self.q_network.load_state_dict(torch.load(dir, map_location=lambda storage, loc: storage))
from gym import spaces
import numpy as np

import torch
import torch.nn.functional as F


class DQNAgent:
    def __init__(self, args, replay_buffer):
        self.memory = replay_buffer
        self.batch_size = args.batch_size
        self.use_double_dqn = False
        self.gamma = args.gamma
        self.args = args
        self.policy_network = DQN(args.state_space, args.action_space).to(self.args.device)
        self.target_network = DQN(args.state_space, args.action_space).to(self.args.device)
        self.update_target_network()
        self.target_network.eval()

        self.optimiser = torch.optim.RMSprop(self.policy_network.parameters()
            , lr=args.lr)
        ## self.optimiser = torch.optim.Adam(self.policy_network.parameters(), lr=lr)

        self.device = args.device

    def update(self):
        """
        Optimise the TD-error over a single minibatch of transitions
        :return: the loss
        """
        device = self.device

        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        states = np.array(states) / 255.0
        next_states = np.array(next_states) / 255.0
        states = torch.from_numpy(states).float().to(device)
        actions = torch.from_numpy(actions).long().to(device)
        rewards = torch.from_numpy(rewards).float().to(device)
        next_states = torch.from_numpy(next_states).float().to(device)
        dones = torch.from_numpy(dones).float().to(device)
        states = states.transpose(3, 2).transpose(2, 1)
        next_states = next_states.transpose(3, 2).transpose(2, 1)

        with torch.no_grad():
            if self.use_double_dqn:
                _, max_next_action = self.policy_network(next_states).max(1)
                max_next_q_values = self.target_network(next_states).gather(1, max_next_action.unsqueeze(1)).squeeze()
            else:
                next_q_values = self.target_network(next_states)
                max_next_q_values, _ = next_q_values.max(1)
            target_q_values = rewards + (1 - dones) * self.gamma * max_next_q_values

        input_q_values = self.policy_network(states)
        input_q_values = input_q_values.gather(1, actions.unsqueeze(1)).squeeze()

        loss = F.smooth_l1_loss(input_q_values, target_q_values)

        self.optimiser.zero_grad()
        loss.backward()
        self.optimiser.step()
        del states
        del next_states
        return loss.item()

    def update_target_network(self):
        """
        Update the target Q-network by copying the weights from the current Q-network
        """
        self.target_network.load_state_dict(self.policy_network.state_dict())

    def select_action(self, state: np.ndarray):
        """
        Select an action greedily from the Q-network given the state
        :param state: the current state
        :return: the action to take
        """

        device = self.device
        state = np.array(state) / 255.0
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        with torch.no_grad():
            state = state.transpose(3, 2).transpose(2, 1)
            q_values = self.policy_network(state)
            _, action = q_values.max(1)
            return action.item()

# if __name__ == '__main__':
#     model = DQN(a)
#     input_data = torch.randn(1024, 3, 64,64)
#     output = model.conv(input_data)
#     print(output.shape)