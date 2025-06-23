import pygame
from game.environment import GameEnvironment
from agent.racer import Racer


def main():
    # --- Config ---
    width, height = 1280, 720
    num_episodes = 1000
    render_every_n_episodes = 10  # Set to 0 to disable rendering

    # --- Setup Environment ---
    track_nodes = [(300, 570), (200, 360), (300, 150), (980, 150), (1080, 360), (980, 570)]
    env = GameEnvironment(track_nodes=track_nodes)

    # --- Setup Racer ---
    state_dim = 6  # 1 for velocity, 5 for lidar
    agent_config = {'state_dim': state_dim, 'action_dim': len(env.action_map)}
    car_config = {}  # Use default car properties for now
    racer = Racer(agent_config, car_config, env.track)

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

        racer.reset()
        state = env._get_state(racer.car)

        render_this_episode = render_every_n_episodes > 0 and episode % render_every_n_episodes == 0

        while not racer.done:
            # Handle Pygame events only when rendering
            if render_this_episode:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False  # This will break the outer loop
                        racer.done = True  # This will break the inner loop
            if not running:
                break

            action = racer.select_action(state)
            next_state, reward, done = env.step(racer, action)
            racer.done = done
            racer.total_reward += reward

            racer.store_transition(state, action, reward, next_state, done)
            racer.experience_replay()

            state = next_state

            if render_this_episode:
                env.draw(screen, [racer])
                pygame.display.flip()
                if clock:
                    clock.tick(60)

        # After an episode
        racer.decay_epsilon()
        if episode > 0 and episode % racer.dqn_agent.target_update == 0:
            racer.update_target_net()

        print(
            f'Episode {episode + 1}/{num_episodes} finished. Reward: {racer.total_reward:.2f}. Laps: {racer.lap_count}. Epsilon: {racer.dqn_agent.epsilon:.2f}'
        )

    if screen:
        pygame.quit()


if __name__ == '__main__':
    main()
