import os
import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque
from typing import List, Tuple, Deque
import torch.nn.functional as F

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
        self.memory: Deque[Tuple[List[float], int, float, List[float], bool]] = deque(maxlen=50_000)

        self.batch_size = 256

        self.gamma = 0.999
        # | γ value         | Horizon you're telling the agent to look at                                     |
        # | --------------- | ------------------------------------------------------------------------------- |
        # | **0.90**        | ≈ 10 steps (reward 10 steps out is worth (0.9)¹⁰ ≈ 0.35 of an immediate reward) |
        # | **0.99**        | ≈ 100 steps horizon                                                             |
        # | **0.995–0.999** | Several hundred to a few thousand steps                                         |

        self.epsilon_start = 1.0
        self.epsilon_end = 0.01
        self.epsilon_decay = 0.995
        self.epsilon = self.epsilon_start
        # Epsilon is the probability of taking a random action which is used to explore the environment
        # The epsilon is decayed over time to epsilon_end to gradually reduce the exploration

        self.target_update = 10
        # The target network is updated every target_update steps, to make the policy network more stable

    def select_actions(self, states: List[List[float]]) -> List[int]:
        with torch.no_grad():
            state_tensor = torch.FloatTensor(np.array(states)).to(device)
            q_values = self.policy_net(state_tensor)

        # Using epsilon as a temperature for softmax distribution for exploration.
        # A higher epsilon means a higher temperature, which leads to a more uniform probability distribution over actions, encouraging exploration.
        # A lower epsilon means a lower temperature, resulting in a sharper distribution, favoring exploitation of known good actions.
        temperature = self.epsilon

        # A small threshold to switch to greedy action selection for numerical stability and to ensure pure exploitation when epsilon is very low.
        if temperature < 1e-3:
            return q_values.argmax(dim=1).tolist()

        scaled_q_values = q_values / temperature
        probs = F.softmax(scaled_q_values, dim=1)
        actions = torch.multinomial(probs, num_samples=1)
        return actions.squeeze(1).tolist()

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

    def load(self, directory: str, filename: str) -> bool:
        checkpoint_path = os.path.join(directory, filename)
        if not os.path.exists(checkpoint_path):
            print(f'Checkpoint not found at {checkpoint_path}. Starting new training.')
            return False

        checkpoint = torch.load(checkpoint_path)
        self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']

        print(f'Loaded model and optimizer state from {checkpoint_path}')
        return True
