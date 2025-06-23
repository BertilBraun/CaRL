import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
import os
from collections import deque
from typing import List, Tuple, Deque

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class DQN(nn.Module):
    def __init__(self, input_dim: int, output_dim: int) -> None:
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class DQNAgent:
    def __init__(self, state_dim: int, action_dim: int) -> None:
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.policy_net = DQN(state_dim, action_dim).to(device)
        self.target_net = DQN(state_dim, action_dim).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=0.001)
        self.memory: Deque[Tuple[List[float], int, float, List[float], bool]] = deque(maxlen=10000)

        self.batch_size = 64
        self.gamma = 0.99
        self.epsilon_start = 0.2
        self.epsilon_end = 0.01
        self.epsilon_decay = 0.995
        self.epsilon = self.epsilon_start

        self.target_update = 1

    def select_actions(self, states: List[List[float]]) -> List[int]:
        if random.random() < self.epsilon:
            return [random.randrange(self.action_dim) for _ in range(len(states))]
        else:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(np.array(states)).to(device)
                q_values = self.policy_net(state_tensor)
                return q_values.argmax(dim=1).tolist()

    def store_transition(
        self, state: List[float], action: int, reward: float, next_state: List[float], done: bool
    ) -> None:
        self.memory.append((state, action, reward, next_state, done))

    def experience_replay(self) -> None:
        if len(self.memory) < self.batch_size:
            return

        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states_tensor = torch.FloatTensor(np.array(states)).to(device)
        actions_tensor = torch.LongTensor(actions).unsqueeze(1).to(device)
        rewards_tensor = torch.FloatTensor(rewards).unsqueeze(1).to(device)
        next_states_tensor = torch.FloatTensor(np.array(next_states)).to(device)
        dones_tensor = torch.BoolTensor(dones).unsqueeze(1).to(device)

        current_q_values = self.policy_net(states_tensor).gather(1, actions_tensor)

        next_q_values = self.target_net(next_states_tensor).max(1)[0].unsqueeze(1)
        next_q_values[dones_tensor] = 0.0

        expected_q_values = rewards_tensor + (self.gamma * next_q_values)

        loss = nn.MSELoss()(current_q_values, expected_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target_net(self) -> None:
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def decay_epsilon(self) -> None:
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    def save(self, directory: str, filename: str) -> None:
        if not os.path.exists(directory):
            os.makedirs(directory)

        checkpoint = {
            'policy_net_state_dict': self.policy_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
        }
        torch.save(checkpoint, os.path.join(directory, filename))

    def load(self, directory: str, filename: str) -> None:
        checkpoint_path = os.path.join(directory, filename)
        if not os.path.exists(checkpoint_path):
            print(f'Checkpoint not found at {checkpoint_path}. Starting new training.')
            return

        checkpoint = torch.load(checkpoint_path)
        self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']

        print(f'Loaded model and optimizer state from {checkpoint_path}')
