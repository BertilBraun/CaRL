import random
import pygame
import os
from typing import List, Tuple

from game.environment import GameEnvironment
from agent.racer import Racer
from agent.dqn_agent import DQNAgent
from game.track import Track


def main() -> None:
    # --- Config ---
    EPISODES = 100
    RACERS = 20
    MAX_ITERATIONS = 500
    EPISODES_TO_RENDER = 2
    CHECKPOINT_DIR = 'checkpoints'
    CHECKPOINT_FILE = 'dqn_model.pth'

    # --- Setup ---
    pygame.init()
    screen_width = 1280
    screen_height = 720
    screen = pygame.display.set_mode((screen_width, screen_height))
    pygame.display.set_caption('Car RL')
    clock = pygame.time.Clock()

    from track import track_nodes

    track = Track(track_nodes)
    env = GameEnvironment(track, MAX_ITERATIONS)

    # --- Agent Setup ---
    state_dim = len(env._get_state(Racer(track, 0.0)))
    action_dim = len(env.action_map)
    agent = DQNAgent(state_dim=state_dim, action_dim=action_dim)

    # Load existing model if found
    if os.path.exists(os.path.join(CHECKPOINT_DIR, CHECKPOINT_FILE)):
        agent.load(CHECKPOINT_DIR, CHECKPOINT_FILE)

    # --- Main Loop ---
    for episode in range(EPISODES):
        racers = active_racers = [Racer(track, random.random() * 0.9) for _ in range(RACERS)]

        while active_racers:
            # --- Agent-Environment Interaction ---
            # 1. Get states from all active racers
            states = [env._get_state(r) for r in active_racers]

            # 2. Get actions from the agent (batched)
            actions = agent.select_actions(states)

            # 3. Apply actions and get results
            for i, racer in enumerate(active_racers):
                # The previous state is what we stored earlier
                prev_state = states[i]
                action = actions[i]

                next_state, reward, done = env.step(racer, action)
                agent.store_transition(prev_state, action, reward, next_state, done)

                racer.done = racer.done or done
                racer.total_reward += reward

            # prepare active racers for next iteration
            active_racers = [r for r in active_racers if not r.done]

            # 4. Perform one step of learning
            agent.experience_replay()

            # --- Rendering ---
            if EPISODES_TO_RENDER > 0 and episode % EPISODES_TO_RENDER == 0:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        exit()

                env.draw(screen, active_racers)
                pygame.display.flip()
                clock.tick(60)

        # --- End of Episode ---
        agent.decay_epsilon()
        if episode > 0 and episode % agent.target_update == 0:
            agent.update_target_net()

        agent.save(CHECKPOINT_DIR, CHECKPOINT_FILE)

        # Log results for the episode
        finished_racers = [r for r in racers if r.next_checkpoint >= len(track.checkpoints) - 1]
        avg_reward = sum([r.total_reward for r in racers]) / RACERS if RACERS > 0 else 0

        print(
            f'Episode {episode + 1}/{EPISODES} | '
            f'Finished: {len(finished_racers)}/{RACERS} | '
            f'Avg Reward: {avg_reward:.2f} | '
            f'Epsilon: {agent.epsilon:.2f}'
        )

    pygame.quit()


if __name__ == '__main__':
    main()
