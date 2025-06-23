# Car-RL Project

A 2D reinforcement learning simulation where a car learns to drive around a track using a Deep Q-Network (DQN) agent.

## Core Features
- **Modular Track System**: Define complex racetracks using a series of centerline nodes.
- **DQN Agent**: A Deep Q-Network agent learns to navigate the track.
- **Parallelized Simulation**: The architecture supports running multiple simulations in parallel for efficient training, using a single shared "brain" to learn from all racers' experiences.
- **Headless & Visual Modes**: Train the agent at maximum speed without rendering, or visualize the process periodically to inspect performance.
- **Typed Codebase**: The project is fully type-hinted for improved clarity, robustness, and maintainability.

---

### Open Todos & Next Steps
- [ ] **Save/Load Models**: Implement functionality to save the trained `DQNAgent`'s state and load it for continued training or inference.
- [ ] **Reward Shaping**: Experiment with more advanced reward functions to encourage more complex behaviors (e.g., rewarding speed, penalizing jerky movements, following a racing line).
- [ ] **Advanced Agent Architectures**: Explore more advanced RL algorithms beyond DQN, such as Dueling DQN or PPO.
- [ ] **UI/Dashboard**: Create a simple dashboard to display training metrics like average reward, episode length, and epsilon value in real-time.
- [ ] **Varied Car Configurations**: Define different car types with unique physics (e.g., higher top speed, better handling) for racers to use.

---

### **NOTE FOR THE AI ASSISTANT**
1.  Do not run `git commit` unless explicitly instructed to do so.
2.  Do not run the program (`python main.py` or any other script) unless explicitly instructed to do so. 