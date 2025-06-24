import random
import pygame
import os
import numpy as np
from typing import Dict

from game.environment import GameEnvironment
from agent.racer import Racer
from agent.dqn_agent import DQNAgent
from game.track import Track

# --- Config ---
RACERS = 100
MAX_ITERATIONS = 1000  # Increased to allow more time for completion
INITIAL_ANGLE_VARIANCE = 3
EPSILON_FOR_EVALUATION = 0.01


def evaluate_model(agent: DQNAgent, env: GameEnvironment, screen: pygame.Surface) -> None:
    original_epsilon = agent.epsilon
    agent.epsilon = EPSILON_FOR_EVALUATION  # Disable randomness almost completely for evaluation

    # Start all racers at the beginning of the track
    racers = active_racers = [Racer(env.track, progress_on_track=0.05) for _ in range(RACERS)]

    # randomly adjust the initial angle by a tiny amount
    for r in racers:
        r.car.angle += random.uniform(-INITIAL_ANGLE_VARIANCE, INITIAL_ANGLE_VARIANCE)

    racer_iterations: Dict[Racer, int] = {r: 0 for r in racers}
    for r in racers:
        r.current_state = env._get_state(r)

    while active_racers:
        # --- Agent-Environment Interaction ---
        # 1. Get states from all active racers
        states = [r.current_state for r in active_racers]

        # 2. Get actions from the agent (deterministic)
        actions = agent.select_actions(states)

        # 3. Apply actions and get results
        for racer, action in zip(active_racers, actions):
            next_state, reward, done = env.step(racer, action)
            racer.done = racer.done or done
            racer.total_reward += reward
            racer.current_state = next_state
            racer_iterations[racer] += 1

        # Prepare active racers for next iteration
        active_racers = [r for r in active_racers if not r.done]

        # --- Rendering ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return

        env.draw(screen, racers[::10])  # Render a subset of racers
        pygame.display.flip()

    # --- End of Episode ---
    finished_racers = [r for r in racers if r.next_checkpoint >= len(env.track.checkpoints) - 1]
    completion_rate = len(finished_racers) / RACERS if RACERS > 0 else 0
    avg_reward = sum(r.total_reward for r in racers) / RACERS if RACERS > 0 else 0

    if finished_racers:
        completion_times = [racer_iterations[r] for r in finished_racers]
        best_time = min(completion_times)
        avg_time = np.mean(completion_times)
    else:
        best_time = -1
        avg_time = -1

    print(
        f'Evaluation Results | '
        f'Finished: {len(finished_racers)}/{RACERS} ({completion_rate:.2%}) | '
        f'Avg Reward: {avg_reward:.2f} | '
        f'Best Time: {best_time} steps | '
        f'Avg Time: {avg_time:.2f} steps'
    )

    # Reset epsilon to original value
    agent.epsilon = original_epsilon


if __name__ == '__main__':

    def main() -> None:
        CHECKPOINT_DIR = 'checkpoints'
        CHECKPOINT_FILE = 'dqn_model.pth'
        # --- Setup ---
        pygame.init()
        screen_width = 1280
        screen_height = 720
        screen = pygame.display.set_mode((screen_width, screen_height))
        pygame.display.set_caption('Car RL - Evaluation')

        from track import track_nodes

        track = Track(track_nodes)
        env = GameEnvironment(track, MAX_ITERATIONS)

        # --- Agent Setup ---
        state_dim = len(env._get_state(Racer(track, 0.0)))
        action_dim = len(env.action_map)
        agent = DQNAgent(state_dim=state_dim, action_dim=action_dim)

        # Load existing model
        model_path = os.path.join(CHECKPOINT_DIR, CHECKPOINT_FILE)
        if not os.path.exists(model_path):
            print(f'Model not found at {model_path}. Exiting.')
            return

        agent.load(CHECKPOINT_DIR, CHECKPOINT_FILE)

        print('\n--- Starting Evaluation ---')
        evaluate_model(agent, env, screen)

        pygame.quit()

    main()
