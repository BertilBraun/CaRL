import pygame
from game.environment import GameEnvironment
from agent.racer import Racer


def test_car_movement():
    pygame.init()

    # --- Config ---
    width, height = 1280, 720
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption('Car Movement Test')
    clock = pygame.time.Clock()

    # --- Setup ---
    rect_track_nodes = [(1088, 576), (192, 576), (192, 144), (1088, 144)]
    env = GameEnvironment(track_nodes=rect_track_nodes)

    state_dim = 6  # 1 for velocity, 5 for lidar
    agent_config = {'state_dim': state_dim, 'action_dim': len(env.action_map)}
    car_config = {}
    racer = Racer(agent_config, car_config, env.track)

    # --- Test Loop ---
    running = True
    action_accelerate = 1  # The action for straight acceleration

    while running and not racer.done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # --- Perform Action ---
        _, _, done = env.step(racer, action_accelerate)
        racer.done = done

        # --- Print state for debugging ---
        if not racer.done:
            print(
                f'Position: {racer.car.position}, Velocity: {racer.car.velocity:.2f}, Angle: {racer.car.angle:.2f}, Crashed: {racer.done}'
            )

        # --- Drawing ---
        screen.fill((0, 0, 0))  # Black background
        env.draw(screen, [racer])
        pygame.display.flip()

        clock.tick(30)

    print('\n--- CRASH DETECTED ---')
    pygame.quit()


if __name__ == '__main__':
    test_car_movement()
