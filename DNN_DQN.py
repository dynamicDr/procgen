import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np


class DQN:
    def __init__(self, args, replay_buffer):
        self.args = args
        self.epsilon = self.args.epsilon_init
        self.q_network = self._build_model().to(self.args.device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.args.lr)
        self.loss_fn = nn.MSELoss()
        self.replay_buffer = replay_buffer

    def _build_model(self):
        # DNN模型定义
        model = nn.Sequential(
            nn.Linear(self.args.state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, self.args.action_dim)
        )
        return model


    def forward(self, x):
        x = self.pool1(nn.functional.relu(self.conv1(x)))
        x = self.pool2(nn.functional.relu(self.conv2(x)))
        x = x.view(-1, 32 * 16 * 16)
        x = nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    def select_action(self, state):
        if random.random() < self.epsilon:
            action = random.randint(0, self.args.action_dim - 1)
        else:
            state = torch.FloatTensor(state).unsqueeze(0).to(self.args.device)
            q_values = self.q_network(state)
            _, action = torch.max(q_values, 1)
            action = action.item()
        return action

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

        state = torch.FloatTensor(state).to(self.args.device).view(self.args.batch_size, -1)
        action = torch.LongTensor(action).to(self.args.device)
        reward = torch.FloatTensor(reward).reshape((self.args.batch_size, 1)).to(self.args.device)
        next_state = torch.FloatTensor(next_state).to(self.args.device).view(self.args.batch_size, -1)
        done = torch.FloatTensor(done).reshape((self.args.batch_size, 1)).to(self.args.device)

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

    def update_epsilon(self):
        self.epsilon = max(self.epsilon * self.args.epsilon_decay, self.args.epsilon_min)

    def save(self, dir, step):
        step = str(step)
        torch.save(self.q_network.state_dict(), dir)


    def load(self, dir):
        self.q_network.load_state_dict(torch.load(dir, map_location=lambda storage, loc: storage))
