# CaRL: A Deep Reinforcement Learning Driving Agent

This project showcases a 2D reinforcement learning simulation where a car, powered by a Deep Q-Network (DQN) agent, learns to navigate complex racetracks. The agent is trained using a parallelized simulation approach, where hundreds of racers learn simultaneously, sharing their experiences to accelerate learning.

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10-blue.svg">
  <img src="https://img.shields.io/badge/Pygame-2.1-orange.svg">
  <img src="https://img.shields.io/badge/PyTorch-1.13-red.svg">
</p>

## Showcase

The videos below demonstrate the agent's performance. The training video shows multiple agents learning simultaneously, starting from random points on the track to improve generalization. The evaluation video shows a single, trained agent driving a full lap.

### Training

![Training](documentation/train_100_200_500.gif)

The training video shows 100 parallel racers learning simultaneously, starting from random points on the track and added noise to the actions to improve generalization. The training episodes are 100, 200, and 500.

### Evaluation

![Evaluation](documentation/evaluation_10_50_200_500.gif)

The evaluation video shows 100 racers driving with slight offsets in the initial angle to show variation in the agent's performance. The evaluation episodes are 10, 50, 200, and 500.

## Core Features

- **Modular Track System**: Easily define complex racetracks by specifying a series of centerline nodes. A visual `track_creator.py` script is included to facilitate the creation of custom tracks.
- **Deep Q-Network (DQN) Agent**: A DQN agent learns to navigate the track from scratch. The agent uses an experience replay buffer and a target network for stable learning.
- **Parallelized Simulation**: The training architecture supports running hundreds of simulations in parallel. All racers share a single "brain," contributing their experiences to a shared replay buffer, which significantly speeds up learning.
- **Headless & Visual Modes**: Train the agent at maximum speed in headless mode or enable rendering to visually inspect the training process. The evaluation script can also export the agent's performance as a GIF.
- **Typed Codebase**: The project is fully type-hinted for improved clarity, robustness, and maintainability.

## The Environment

Documentation can be found here: [![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/BertilBraun/CaRL)

### State Representation
The agent perceives its environment through a state vector containing:
1.  **LIDAR Readings**: A set of distance measurements to the track boundaries, simulating LIDAR sensors.
2.  **Normalized Forward Speed**: The car's velocity component in the direction of the next checkpoint.
3.  **Normalized Distance to Centerline**: The car's perpendicular distance from the track's centerline.
4.  **Normalized Angle to Centerline**: The car's heading relative to the direction of the centerline.

### Reward Function
The reward function is designed to encourage fast and efficient driving along the track's centerline.
- **Positive Rewards**:
    - **Progress**: The primary reward is proportional to the distance covered along the centerline towards the next checkpoint.
    - **Checkpoint Crossing**: A fixed reward of `+10` is given for crossing a checkpoint.
    - **Finishing the Lap**: A large reward of `+1000` is given for successfully completing the track.
- **Penalties (Negative Rewards)**:
    - **Time Penalty**: A small penalty of `-0.2` is applied at every step to encourage speed.
    - **Crashing**: A large penalty of `-500` is given if the car collides with the track boundaries.
    - **Getting Stuck**: A penalty of `-500` is applied if the car stops moving for an extended period.
    - **Stagnation**: A penalty of `-100` is given if the car takes too long to reach the next checkpoint.

To prevent overfitting and improve generalization, each racer in a training episode starts at a random position along the track's centerline.

## Key Scripts

- **`main.py`**: The main script for training the agent. It sets up the environment, agent, and runs the training loop. It handles the parallel simulation of multiple racers and periodically saves the model.
- **`eval.py`**: Used to evaluate a trained agent. It loads a saved model, runs it in the environment with near-deterministic actions, and reports performance metrics. It can also generate a GIF of the evaluation run.
- **`track_creator.py`**: A utility script that provides a simple graphical interface for creating new tracks. Click to place nodes that define the track's centerline, and press 'S' to print the node coordinates to the console.

---

### Open Todos & Next Steps
- [ ] **Reward Shaping**: Experiment with more advanced reward functions to encourage more complex behaviors (e.g., rewarding speed, penalizing jerky movements, following a racing line).
- [ ] **Advanced Agent Architectures**: Explore more advanced RL algorithms beyond DQN, such as Dueling DQN or PPO.
- [ ] **UI/Dashboard**: Create a simple dashboard to display training metrics like average reward, episode length, and epsilon value in real-time.
- [ ] **Varied Car Configurations**: Define different car types with unique physics (e.g., higher top speed, better handling) for racers to use.
