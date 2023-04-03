import copy

import torch
import torch.nn as nn
import torch.optim as optim
import random
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from gym.vector.utils import spaces

class Conv2d_tf(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super(Conv2d_tf, self).__init__(*args, **kwargs)
        self.padding = kwargs.get("padding", "same")

    def _compute_padding(self, input, dim):
        input_size = input.size(dim + 2)
        filter_size = self.weight.size(dim + 2)
        effective_filter_size = (filter_size - 1) * self.dilation[dim] + 1
        out_size = (input_size + self.stride[dim] - 1) // self.stride[dim]
        total_padding = max(0, (out_size - 1) * self.stride[dim] + effective_filter_size - input_size)
        additional_padding = int(total_padding % 2 != 0)
        return additional_padding, total_padding

    def forward(self, input):
        if self.padding == "VALID":
            return F.conv2d(
                input,
                self.weight,
                self.bias,
                self.stride,
                padding=0,
                dilation=self.dilation,
                groups=self.groups,
            )
        rows_odd, padding_rows = self._compute_padding(input, dim=0)
        cols_odd, padding_cols = self._compute_padding(input, dim=1)
        if rows_odd or cols_odd:
            input = F.pad(input, [0, cols_odd, 0, rows_odd])

        return F.conv2d(
            input,
            self.weight,
            self.bias,
            self.stride,
            padding=(padding_rows // 2, padding_cols // 2),
            dilation=self.dilation,
            groups=self.groups,
        )


class L2Pool(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.pool = nn.AvgPool2d(*args, **kwargs)
        self.n = self.pool.kernel_size ** 2

    def forward(self, x):
        return torch.sqrt(self.pool(x ** 2) * self.n)

class SimpleDQN(nn.Module):
    def __init__(self, args):
        super(SimpleDQN, self).__init__()
        self.action_dim = args.action_dim
        self.conv1 = Conv2d_tf(3, 16, kernel_size=7, stride=1, padding="same")
        self.pool1 = L2Pool(kernel_size=2, stride=2)
        self.conv2a = Conv2d_tf(16, 32, kernel_size=5, stride=1, padding="same")
        self.conv2b = Conv2d_tf(32, 32, kernel_size=5, stride=1, padding="same")
        self.pool2 = L2Pool(kernel_size=2, stride=2)
        self.conv3 = Conv2d_tf(32, 32, kernel_size=7, stride=1, padding="same")
        self.pool3 = L2Pool(kernel_size=2, stride=2)
        self.conv4 = Conv2d_tf(32, 32, kernel_size=7, stride=1, padding="same")
        self.pool4 = L2Pool(kernel_size=2, stride=2)
        self.flat = nn.Flatten()
        self.fc_h1_v = nn.Linear(512, 256)
        self.fc_h1_a = nn.Linear(512, 256)
        self.fc_h2_v = nn.Linear(256, 512)
        self.fc_h2_a = nn.Linear(256, 512)
        self.fc_z_v = nn.Linear(512, 1)
        self.fc_z_a = nn.Linear(512, self.action_dim)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2b(F.relu(self.conv2a(x)))))
        x = self.pool3(F.relu(self.conv3(x)))
        x = self.pool4(F.relu(self.conv4(x)))
        x = self.flat(x)
        value = F.relu(self.fc_h1_v(x))
        value = F.relu(self.fc_h2_v(value))
        value = self.fc_z_v(value)
        advantage = F.relu(self.fc_h1_a(x))
        advantage = F.relu(self.fc_h2_a(advantage))
        advantage = self.fc_z_a(advantage)
        value, advantage = (
            value.view(
                -1,
                1,
            ),
            advantage.view(-1, self.action_space),
        )
        q = value + advantage - advantage.mean(1, keepdim=True)
        return q

    def effective_rank(self, delta=0.01):
        _, s, _ = torch.svd(self.fc_h_v.weight)
        diag_sum = torch.sum(s)
        partial_sum = s[0]
        k = 0
        while (partial_sum / diag_sum) < (1 - delta):
            k += 1
            partial_sum += s[k]
        return

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
class CNN_DQNAgent(object):
    def __init__(self, args, replay_buffer):
        self.args = args
        self.epsilon = self.args.epsilon_init

        self.loss_fn = nn.MSELoss()
        self.replay_buffer = replay_buffer
        self.device = args.device
        self.action_space = args.action_space
        self.batch_size = args.batch_size
        self.gamma = args.gamma
        self.q_network = SimpleDQN(args).to(self.device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.args.lr)

    def select_action(self, state):
        with torch.no_grad():
            q = self.q_network(state)
            action = q.argmax(1).reshape(-1, 1)
            max_q = q.max(1)[0]
            mean_q = q.mean(1)
            return action

    #     def select_action(self, state):
    #         if random.random() < self.epsilon:
    #             action = random.randint(0, self.args.action_dim - 1)
    #         else:
    #             state = torch.FloatTensor(state).unsqueeze(0).to(self.args.device)
    #             q_values = self.q_network(state)
    #             _, action = torch.max(q_values, 1)
    #             action = action.item()
    #         return action

    def update(self):
        # Sample a batch of transitions from replay buffer:
        if self.replay_buffer.size < self.args.batch_size:
            return
        sample = self.replay_buffer.sample()

        if sample is None:
            return
        state, action, reward, next_state, done, weights, indices = sample
        if state is None:
            return

        state = torch.FloatTensor(state).to(self.args.device)
        action = torch.LongTensor(action).to(self.args.device)
        reward = torch.FloatTensor(reward).reshape((self.args.batch_size, 1)).to(self.args.device)
        next_state = torch.FloatTensor(next_state).to(self.args.device)
        done = torch.FloatTensor(done).reshape((self.args.batch_size, 1)).to(self.args.device)

        state = state.transpose(3, 2).transpose(2, 1)
        next_state = next_state.transpose(3, 2).transpose(2, 1)
        print(state.shape)
        q_values = self.q_network(state)
        next_q_values = self.q_network(next_state)
        next_q_state_value, _ = torch.max(next_q_values, 1)
        target_q_values = reward + (1 - done) * self.args.gamma * next_q_state_value.detach()
        q_value = q_values.gather(1, action.unsqueeze(1)).squeeze(1)

        loss = self.loss_fn(q_value, target_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()


    def save(self, dir, step):
        step = str(step)
        torch.save(self.q_network.state_dict(), f"{dir}/step_{step}k.pth")


    def load(self, dir):
        self.q_network.load_state_dict(torch.load(dir, map_location=lambda storage, loc: storage))


# if __name__ == '__main__':
#     model = DQN(a)
#     input_data = torch.randn(1024, 3, 64,64)
#     output = model.conv(input_data)
#     print(output.shape)