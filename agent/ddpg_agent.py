import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from collections import deque
import random
import os
from typing import List, Deque, Tuple

# --- Hyperparameters ---
BUFFER_SIZE = 100000
BATCH_SIZE = 64
GAMMA = 0.99
TAU = 0.001  # For soft target updates
LR_ACTOR = 0.0001
LR_CRITIC = 0.001
WEIGHT_DECAY = 0  # L2 weight decay
NOISE_STDDEV = 0.2
NOISE_CLIP = 0.5

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# --- Actor Network ---
class Actor(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, max_action: float):
        super(Actor, self).__init__()
        self.layer_1 = nn.Linear(state_dim, 400)
        self.layer_2 = nn.Linear(400, 300)
        self.layer_3 = nn.Linear(300, action_dim)
        self.max_action = max_action

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.layer_1(x))
        x = F.relu(self.layer_2(x))
        x = torch.tanh(self.layer_3(x)) * self.max_action
        return x


# --- Critic Network ---
class Critic(nn.Module):
    def __init__(self, state_dim: int, action_dim: int):
        super(Critic, self).__init__()
        self.layer_1 = nn.Linear(state_dim + action_dim, 400)
        self.layer_2 = nn.Linear(400, 300)
        self.layer_3 = nn.Linear(300, 1)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        x = torch.cat([state, action], 1)
        x = F.relu(self.layer_1(x))
        x = F.relu(self.layer_2(x))
        x = self.layer_3(x)
        return x


# --- Replay Buffer ---
class ReplayBuffer:
    def __init__(self, buffer_size: int):
        self.buffer: Deque[Tuple[np.ndarray, np.ndarray, float, np.ndarray, bool]] = deque(maxlen=buffer_size)

    def add(self, state: np.ndarray, action: np.ndarray, reward: float, next_state: np.ndarray, done: bool) -> None:
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.array, zip(*batch))
        return state, action, reward, next_state, done

    def __len__(self) -> int:
        return len(self.buffer)


# --- DDPG Agent ---
class DDPGAgent:
    def __init__(self, state_dim: int, action_dim: int, max_action: float):
        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=LR_ACTOR)

        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = Critic(state_dim, action_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=LR_CRITIC, weight_decay=WEIGHT_DECAY)

        self.replay_buffer = ReplayBuffer(BUFFER_SIZE)
        self.max_action = max_action
        self.action_dim = action_dim

        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01

    def select_actions(self, states: List[List[float]]) -> List[np.ndarray]:
        states_tensor = torch.FloatTensor(np.array(states)).to(device)
        actions = self.actor(states_tensor).cpu().data.numpy()
        return [
            np.random.uniform(-self.max_action, self.max_action, size=self.action_dim)
            if random.random() < self.epsilon
            else action.flatten()
            for action in actions
        ]

    def store_transition(
        self, state: List[float], action: np.ndarray, reward: float, next_state: List[float], done: bool
    ) -> None:
        self.replay_buffer.add(np.array(state), action, reward, np.array(next_state), done)

    def learn(self) -> None:
        if len(self.replay_buffer) < BATCH_SIZE:
            return

        state, action, reward, next_state, done = self.replay_buffer.sample(BATCH_SIZE)

        state = torch.FloatTensor(state).to(device)
        action = torch.FloatTensor(action).to(device)
        reward = torch.FloatTensor(reward.reshape(-1, 1)).to(device)
        next_state = torch.FloatTensor(next_state).to(device)
        done = torch.FloatTensor(done.reshape(-1, 1)).to(device)

        # --- Update Critic ---
        with torch.no_grad():
            target_actions = self.actor_target(next_state)
            target_q = self.critic_target(next_state, target_actions)
            target_q = reward + (1 - done) * GAMMA * target_q
        current_q = self.critic(state, action)
        critic_loss = F.mse_loss(current_q, target_q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # --- Update Actor ---
        actor_loss = -self.critic(state, self.actor(state)).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

    def decay_epsilon(self) -> None:
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def update_target_net(self) -> None:
        # --- Soft update target networks ---
        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(TAU * param.data + (1.0 - TAU) * target_param.data)

        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(TAU * param.data + (1.0 - TAU) * target_param.data)

    def save(self, directory: str, filename: str) -> None:
        if not os.path.exists(directory):
            os.makedirs(directory)
        torch.save(self.critic.state_dict(), os.path.join(directory, filename + '_critic.pth'))
        torch.save(self.actor.state_dict(), os.path.join(directory, filename + '_actor.pth'))

    def load(self, directory: str, filename: str) -> None:
        self.critic.load_state_dict(torch.load(os.path.join(directory, filename + '_critic.pth')))
        self.critic_target = self.critic
        self.actor.load_state_dict(torch.load(os.path.join(directory, filename + '_actor.pth')))
        self.actor_target = self.actor
