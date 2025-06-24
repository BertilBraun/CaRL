import pygame
import sys
from game.track import Track
from typing import List, Tuple

# --- Configuration ---
SCREEN_WIDTH = 1280
SCREEN_HEIGHT = 720
BG_COLOR = (50, 50, 50)
NODE_COLOR = (255, 255, 0)
NODE_RADIUS = 5
TEXT_COLOR = (255, 255, 255)


def main() -> None:
    """Main function to run the track creator."""
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption('Track Creator')
    font = pygame.font.SysFont('monospace', 20)

    nodes: List[Tuple[float, float]] = []
    running = True

    while running:
        # --- Event Handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            if event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:  # Left click
                    nodes.append(event.pos)

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False

                if event.key == pygame.K_c:  # Clear nodes
                    nodes = []
                    print('Track cleared.')

                if event.key == pygame.K_r:  # Remove last node
                    nodes.pop()
                    print('Last node removed.')

                if event.key == pygame.K_s and len(nodes) > 1:  # Save nodes
                    # Print in a copy-paste friendly format
                    print('\n--- Track Nodes ---')
                    formatted_nodes = '[\n'
                    for node in nodes:
                        formatted_nodes += f'    ({node[0]:.1f}, {node[1]:.1f}),\n'
                    formatted_nodes += ']'
                    print(formatted_nodes)
                    print('-------------------')
                    print('Track data printed to console.')

        # --- Drawing ---
        screen.fill(BG_COLOR)

        # Draw the track if we have enough nodes
        if len(nodes) >= 2:
            try:
                track = Track(nodes)
                track.draw(screen)
            except Exception as e:
                # Catch potential errors during track generation (e.g., normalization of zero vector)
                print(f'Error generating track preview: {e}')

        # Draw the nodes on top
        for node in nodes:
            pygame.draw.circle(screen, NODE_COLOR, node, NODE_RADIUS)

        # Draw instructions
        instructions = [
            'Click to add nodes.',
            "Press 'S' to print nodes to console.",
            "Press 'C' to clear.",
            "Press 'ESC' to quit.",
        ]
        for i, text in enumerate(instructions):
            text_surface = font.render(text, True, TEXT_COLOR)
            screen.blit(text_surface, (10, 10 + i * 25))

        pygame.display.flip()

    pygame.quit()
    sys.exit()


if __name__ == '__main__':
    main()
