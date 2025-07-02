import random
from typing import List, Tuple
import pygame
import os

from game.car import Car
from track import track_nodes

from game.environment import GameEnvironment
from agent.racer import Racer
from agent.dqn_agent import DQNAgent
from game.track import Track
from eval import evaluate_model


def setup_screen() -> pygame.Surface:
    pygame.init()
    screen_width = 1280
    screen_height = 720
    screen = pygame.display.set_mode((screen_width, screen_height))
    pygame.display.set_caption('Car RL')
    return screen


def setup_environment(track_nodes: List[Tuple[float, float]]) -> GameEnvironment:
    track = Track(track_nodes)
    return GameEnvironment(track, timeout_to_do_something=100)


def setup_agent(env: GameEnvironment) -> DQNAgent:
    state_dim = len(env.get_state(Racer(env.track, 0.0, 0.0, 0.0)))
    action_dim = len(env.ACTION_MAP)
    agent = DQNAgent(state_dim=state_dim, action_dim=action_dim)
    return agent


def main() -> None:
    # --- Config ---
    EPISODES = 1000
    RACERS = 500
    MAX_ITERATIONS = 200
    EPISODES_TO_RENDER = 2
    EPISODES_TO_EVALUATE = [5, 10, 20, 50, 100, 200, 350, 500, 750, 1000]
    INITIAL_ANGLE_VARIANCE = 3
    CHECKPOINT_DIR = 'checkpoints'
    CHECKPOINT_FILE = 'dqn_model.pth'
    EVAL_CHECKPOINT_FILE_FORMAT = 'dqn_model_{}.pth'

    # --- Setup ---
    screen = setup_screen()

    env = setup_environment(track_nodes)

    # --- Agent Setup ---
    agent = setup_agent(env)

    # Load existing model if found
    if os.path.exists(os.path.join(CHECKPOINT_DIR, CHECKPOINT_FILE)):
        agent.load(CHECKPOINT_DIR, CHECKPOINT_FILE)

    # --- Main Loop ---
    for episode in range(EPISODES):
        racers = active_racers = [
            Racer(
                env.track,
                progress_on_track=random.random() * 0.95 + 0.01,
                initial_velocity=random.random() * Car.max_velocity,
                initial_angle_variance=INITIAL_ANGLE_VARIANCE,
            )
            for _ in range(RACERS)
        ]

        for r in racers:
            r.current_state = env.get_state(r)

        for _ in range(MAX_ITERATIONS):
            # --- Agent-Environment Interaction ---
            # 1. Get states from all active racers
            states = [r.current_state for r in active_racers]

            # 2. Get actions from the agent (batched)
            actions = agent.select_actions(states)

            # 3. Apply actions and get results
            for racer, action in zip(active_racers, actions):
                # The previous state is what we stored earlier
                prev_state = racer.current_state

                next_state, reward, done = env.step(racer, action)
                racer.done = racer.done or done
                racer.total_reward += reward
                racer.current_state = next_state

                if racer.car.velocity > 1.0 or racer.done:
                    agent.store_transition(prev_state, action, reward, next_state, racer.done)

            # 4. Perform one step of learning
            agent.experience_replay()

            # 5. prepare active racers for next iteration
            active_racers = [r for r in active_racers if not r.done]

            # 6. check if all racers are done
            if not active_racers:
                break

            # --- Rendering ---
            if EPISODES_TO_RENDER > 0 and episode % EPISODES_TO_RENDER == 0:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        exit()

                env.draw(screen, racers[::10])
                pygame.display.flip()

        # --- End of Episode ---
        agent.decay_epsilon()
        agent.save(CHECKPOINT_DIR, CHECKPOINT_FILE)

        if episode in EPISODES_TO_EVALUATE:
            eval_checkpoint_file = EVAL_CHECKPOINT_FILE_FORMAT.format(episode)
            agent.save(CHECKPOINT_DIR, eval_checkpoint_file)
            evaluate_model(agent, env, screen, f'evaluation_{episode}.gif')

        # Log results for the episode
        finished_racers = [r for r in racers if r.next_checkpoint >= len(env.track.checkpoints) - 1]
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
