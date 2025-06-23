import pygame
import os
from typing import List, Tuple

from game.environment import GameEnvironment
from agent.racer import Racer
from agent.ddpg_agent import DDPGAgent
from game.track import Track


def main() -> None:
    # --- Config ---
    EPISODES = 10
    RACERS = 10  # TODO: DDPG is off-policy, better to train with one racer at a time??
    MAX_ITERATIONS = 2000
    RENDER = True
    CHECKPOINT_DIR = 'checkpoints'
    CHECKPOINT_FILE = 'ddpg_model'

    # --- Setup ---
    pygame.init()
    screen_width = 1280
    screen_height = 720
    screen = pygame.display.set_mode((screen_width, screen_height))
    pygame.display.set_caption('Car RL')
    clock = pygame.time.Clock()

    track_nodes: List[Tuple[float, float]] = [
        (300.0, 570.0),
        (200.0, 360.0),
        (300.0, 150.0),
        (980.0, 150.0),
        (1080.0, 360.0),
        (980.0, 570.0),
    ]
    track = Track(track_nodes)
    env = GameEnvironment(track)

    # --- Agent Setup ---
    state_dim = len(env._get_state(Racer(track)))
    action_dim = 2  # [acceleration, steering]
    max_action = 1.0  # Corresponds to car's max inputs
    agent = DDPGAgent(state_dim, action_dim, max_action)

    # Load existing model if found
    try:
        agent.load(CHECKPOINT_DIR, CHECKPOINT_FILE)
        print('Loaded model checkpoint.')
    except FileNotFoundError:
        print('No checkpoint found, starting new training.')

    # --- Main Loop ---
    for episode in range(EPISODES):
        racers = [Racer(track) for _ in range(RACERS)]

        for _ in range(MAX_ITERATIONS):
            if RENDER:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        exit()

            # --- Agent-Environment Interaction ---
            # 1. Get states from all active racers
            states = [env._get_state(r) for r in racers]

            # 2. Get actions from the agent (batched)
            actions = agent.select_actions(states)

            # 3. Apply actions and get results
            for i, racer in enumerate(racers):
                # The previous state is what we stored earlier
                prev_state = states[i]
                action = actions[i]

                # The environment steps forward
                next_state, reward, done = env.step(racer, action)

                # Store the transition in the agent's memory
                agent.store_transition(prev_state, action, reward, next_state, done)
                racer.done = done

            racers = [r for r in racers if not r.done]

            # 4. Perform one step of learning
            agent.learn()

            # --- Rendering ---
            if RENDER:
                env.draw(screen, racers)
                pygame.display.flip()
                clock.tick(60)

            if not racers:
                break

        print(f'Episode {episode} finished. All racers crashed or finished.')
        agent.decay_epsilon()  # Decay epsilon at the end of an episode
        agent.update_target_net()

        agent.save(CHECKPOINT_DIR, CHECKPOINT_FILE)

        # Log results for the episode
        avg_reward = sum([r.total_reward for r in racers]) / RACERS
        avg_laps = sum([r.lap_count for r in racers]) / RACERS
        print(
            f'Episode {episode + 1}/{EPISODES} | Avg Reward: {avg_reward:.2f} | Avg Laps: {avg_laps:.2f} | Epsilon: {agent.epsilon:.2f}'
        )

    pygame.quit()


if __name__ == '__main__':
    main()
