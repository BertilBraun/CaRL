import pygame
import imageio
from typing import List

from game.environment import GameEnvironment
from agent.racer import Racer
from agent.dqn_agent import DQNAgent

# --- Config ---
RACERS = 100
MAX_ITERATIONS = 1000  # Increased to allow more time for completion
INITIAL_ANGLE_VARIANCE = 5
EPSILON_FOR_EVALUATION = 0.001


def evaluate_model(agents: List[DQNAgent], env: GameEnvironment, screen: pygame.Surface, output_gif_file: str) -> None:
    for agent in agents:
        agent.epsilon = EPSILON_FOR_EVALUATION

    # Start all racers at the beginning of the track
    racers = active_racers = [
        [
            Racer(
                env.track, progress_on_track=0.01, initial_velocity=0.0, initial_angle_variance=INITIAL_ANGLE_VARIANCE
            )
            for _ in range(RACERS)
        ]
        for agent in agents
    ]

    for r in racers:
        for racer in r:
            racer.current_state = env.get_state(racer)

    frames = []
    while any(active_racers):
        # --- Agent-Environment Interaction ---
        actions = [
            agent.select_actions([r.current_state for r in racers]) if racers else []
            for agent, racers in zip(agents, active_racers)
        ]

        # 3. Apply actions and get results
        for racer, action in zip(active_racers, actions):
            for racer, action in zip(racer, action):
                next_state, reward, done = env.step(racer, action)
                racer.done = racer.done or done
                racer.total_reward += reward
                racer.current_state = next_state

        # Prepare active racers for next iteration
        active_racers = [[r for r in r if not r.done] for r in active_racers]

        # --- Rendering ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return

        env.track.draw(screen)

        for i, racer in enumerate(racers):
            # Lerp between red and green based on i / len(racers)
            v = max(0.0, min(1.0, i / len(racers)))
            color = (
                int(255 * (1 - v)),  # R: 255->0
                int(255 * v),  # G: 0->255
                0,  # B: always 0
            )
            for racer in racer:
                racer.car.draw(screen, color)  # if not racer.done else (255, 0, 0))
        pygame.display.flip()

        frame = pygame.surfarray.array3d(screen).transpose((1, 0, 2))
        frames.append(frame)

    # --- End of Episode ---
    print('Saving evaluation GIF...')
    imageio.mimsave(output_gif_file, frames[::3], fps=300)
    print(f'GIF saved as {output_gif_file}')


if __name__ == '__main__':
    CHECKPOINT_DIR = 'checkpoints'

    def main(checkpoint_files: List[str]) -> None:
        from track import track_nodes
        from main import setup_screen, setup_environment, setup_agent

        # --- Setup ---
        screen = setup_screen()

        env = setup_environment(track_nodes)

        # --- Agent Setup ---
        agents: List[DQNAgent] = []
        for checkpoint_file in checkpoint_files:
            agent = setup_agent(env)
            agent.load(CHECKPOINT_DIR, checkpoint_file)
            agents.append(agent)

        print('\n--- Starting Evaluation ---')
        output_gif_file = 'evaluation.gif'
        evaluate_model(agents, env, screen, output_gif_file)

        pygame.quit()

    checkpoint_files = [f'dqn_model_{i}.pth' for i in (1, 5, 10, 20, 50, 100, 200, 350, 500)]
    main(checkpoint_files)
