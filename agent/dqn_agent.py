import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque
from typing import List, Deque, Tuple, Union


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

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.policy_net = DQN(state_dim, action_dim).to(self.device)
        self.target_net = DQN(state_dim, action_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=0.001)
        self.memory: Deque[Tuple[List[float], int, float, List[float], bool]] = deque(maxlen=10000)

        self.batch_size = 64
        self.gamma = 0.99
        self.epsilon_start = 1.0
        self.epsilon_end = 0.01
        self.epsilon_decay = 0.995
        self.epsilon = self.epsilon_start

        self.target_update = 10

    def select_action(self, states: List[List[float]], explore: bool = True) -> List[int]:
        if explore and random.random() < self.epsilon:
            return [random.randrange(self.action_dim) for _ in range(len(states))]
        else:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(np.array(states)).to(self.device)
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

        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.BoolTensor(dones).unsqueeze(1).to(self.device)

        current_q_values = self.policy_net(states).gather(1, actions)

        next_q_values = self.target_net(next_states).max(1)[0].unsqueeze(1)
        next_q_values[dones] = 0.0

        expected_q_values = rewards + (self.gamma * next_q_values)

        loss = nn.MSELoss()(current_q_values, expected_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target_net(self) -> None:
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def decay_epsilon(self) -> None:
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
