from game.car import Car
from .dqn_agent import DQNAgent


class Racer:
    def __init__(self, agent_config, car_config, track):
        # Create the underlying DQN agent
        self.dqn_agent = DQNAgent(state_dim=agent_config['state_dim'], action_dim=agent_config['action_dim'])
        self.track = track
        self.car_config = car_config
        self.reset()

    def reset(self):
        # Create a new car for the racer
        start_pos = self.track.get_start_position()
        start_angle = self.track.get_start_angle()
        self.car = Car(start_pos[0], start_pos[1], angle=start_angle, **self.car_config)

        # Reset racer's progress state
        self.last_pos = self.car.position.copy()
        self.next_checkpoint = 0
        self.lap_count = 0
        self.time_since_checkpoint = 0
        self.total_reward = 0
        self.done = False

    def select_action(self, state):
        return self.dqn_agent.select_action(state)

    def store_transition(self, state, action, reward, next_state, done):
        self.dqn_agent.store_transition(state, action, reward, next_state, done)

    def experience_replay(self):
        self.dqn_agent.experience_replay()

    def update_target_net(self):
        self.dqn_agent.update_target_net()

    def decay_epsilon(self):
        self.dqn_agent.decay_epsilon()
