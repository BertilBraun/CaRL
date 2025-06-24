import os
import random
import pygame
import imageio
import numpy as np
from typing import Dict

from game.environment import GameEnvironment
from agent.racer import Racer
from agent.dqn_agent import DQNAgent

# --- Config ---
RACERS = 100
MAX_ITERATIONS = 1000  # Increased to allow more time for completion
INITIAL_ANGLE_VARIANCE = 5
EPSILON_FOR_EVALUATION = 0.001

GENERATE_GIFS = True


def evaluate_model(agent: DQNAgent, env: GameEnvironment, screen: pygame.Surface, output_gif_file: str) -> None:
    original_epsilon = agent.epsilon
    agent.epsilon = EPSILON_FOR_EVALUATION  # Disable randomness almost completely for evaluation

    # Start all racers at the beginning of the track
    racers = active_racers = [Racer(env.track, progress_on_track=random.random() * 0.95 + 0.02) for _ in range(RACERS)]

    # randomly adjust the initial angle by a tiny amount
    for r in racers:
        r.car.angle += random.uniform(-INITIAL_ANGLE_VARIANCE, INITIAL_ANGLE_VARIANCE)

    racer_iterations: Dict[Racer, int] = {r: 0 for r in racers}
    for r in racers:
        r.current_state = env.get_state(r)

    frames = []
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

        env.draw(screen, racers)
        pygame.display.flip()
        if GENERATE_GIFS:
            frame = pygame.surfarray.array3d(screen).transpose((1, 0, 2))
            frames.append(frame)

    # --- End of Episode ---
    if GENERATE_GIFS:
        print('Saving evaluation GIF...')
        imageio.mimsave(output_gif_file, frames, fps=300)
        print(f'GIF saved as {output_gif_file}')

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
    CHECKPOINT_DIR = 'checkpoints'
    CHECKPOINT_FILE = 'dqn_model.pth'

    def main(checkpoint_file: str) -> None:
        from track import track_nodes
        from main import setup_screen, setup_environment, setup_agent

        # --- Setup ---
        screen = setup_screen()

        env = setup_environment(track_nodes)

        # --- Agent Setup ---
        agent = setup_agent(env)

        # Load existing model
        if not agent.load(CHECKPOINT_DIR, checkpoint_file):
            print(f'Model not found {checkpoint_file}. Exiting.')
            return

        print('\n--- Starting Evaluation ---')
        output_gif_file = f'evaluation_{checkpoint_file}.gif'
        evaluate_model(agent, env, screen, output_gif_file)

        pygame.quit()

    main('dqn_model_200.pth')
