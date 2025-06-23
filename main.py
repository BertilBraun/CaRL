import pygame
from game.environment import GameEnvironment
from agent.racer import Racer
from agent.dqn_agent import DQNAgent


def main():
    # --- Config ---
    NUM_WORKERS = 8  # Number of racers to simulate in parallel
    width, height = 1280, 720
    num_episodes = 1000
    render_every_n_episodes = 10  # Set to 0 to disable rendering

    # --- Setup Environment ---
    track_nodes = [(300, 570), (200, 360), (300, 150), (980, 150), (1080, 360), (980, 570)]
    env = GameEnvironment(track_nodes=track_nodes)

    # --- Setup Agent (The Brain) ---
    state_dim = 6  # 1 for velocity, 5 for lidar
    agent = DQNAgent(state_dim=state_dim, action_dim=len(env.action_map))

    # --- Setup Racers (The Actors) ---
    car_config = {}  # Use default car properties for now
    racers = [Racer(car_config, env.track) for _ in range(NUM_WORKERS)]

    # --- Pygame setup (optional) ---
    screen = None
    clock = None
    if render_every_n_episodes > 0:
        pygame.init()
        screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption('Car RL')
        clock = pygame.time.Clock()

    # --- Training Loop ---
    running = True
    for episode in range(num_episodes):
        if not running:
            break

        for racer in racers:
            racer.reset()

        render_this_episode = render_every_n_episodes > 0 and episode % render_every_n_episodes == 0

        while True:  # Loop for simulation steps
            if render_this_episode:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
                if not running:
                    break

            # --- Batch process all active racers ---
            active_racers = [r for r in racers if not r.done]
            if not active_racers:
                break  # End episode if all racers are done

            # 1. Collect states from active racers
            states = [env._get_state(r.car) for r in active_racers]

            # 2. Agent selects actions for the batch
            actions = agent.select_action(states)

            # 3. Step environment and store transitions for each racer
            for i, racer in enumerate(active_racers):
                action = actions[i]
                state = states[i]

                next_state, reward, done = env.step(racer, action)
                racer.done = done
                racer.total_reward += reward

                agent.store_transition(state, action, reward, next_state, done)

            # 4. Perform one learning step
            agent.experience_replay()

            if render_this_episode:
                env.draw(screen, racers)
                pygame.display.flip()
                if clock:
                    clock.tick(60)

        # After an episode, update the agent
        agent.decay_epsilon()
        if episode > 0 and episode % agent.target_update == 0:
            agent.update_target_net()

        # Log results for the episode
        avg_reward = sum([r.total_reward for r in racers]) / len(racers)
        avg_laps = sum([r.lap_count for r in racers]) / len(racers)
        print(
            f'Episode {episode + 1}/{num_episodes} | Avg Reward: {avg_reward:.2f} | Avg Laps: {avg_laps:.2f} | Epsilon: {agent.epsilon:.2f}'
        )

    if screen:
        pygame.quit()


if __name__ == '__main__':
    main()
