import pygame
import math
from utils.geometry import get_infinite_line_intersection


class Track:
    def __init__(self, nodes=None, width=150):
        self.track_width = width

        if nodes is None:
            # Default rectangular track centerline, defined in absolute coordinates
            nodes = [(1088, 576), (192, 576), (192, 144), (1088, 144)]

        self.center_nodes = [pygame.math.Vector2(p) for p in nodes]
        self._calculate_boundaries()

        # Create a line for the start/finish line
        start_pos_vec = pygame.math.Vector2(self.get_start_position())
        direction = (self.center_nodes[0] - self.center_nodes[-1]).normalize()
        perp_vec = pygame.math.Vector2(direction.y, -direction.x)  # Use the correct normal for CW
        self.start_finish_line = (
            start_pos_vec + perp_vec * (self.track_width / 2),
            start_pos_vec - perp_vec * (self.track_width / 2),
        )

        self._create_checkpoints()

    def _calculate_boundaries(self):
        nodes = self.center_nodes
        self.outer_points = []
        self.inner_points = []

        for i in range(len(nodes)):
            p_curr = nodes[i]
            p_prev = nodes[i - 1]  # Wraps around due to negative index
            p_next = nodes[(i + 1) % len(nodes)]

            v_in = (p_curr - p_prev).normalize()
            v_out = (p_next - p_curr).normalize()

            # Correctly calculate outward-pointing normal for a CW track
            n_in = pygame.math.Vector2(v_in.y, -v_in.x)
            n_out = pygame.math.Vector2(v_out.y, -v_out.x)

            outer1_p1 = p_prev + n_in * (self.track_width / 2)
            outer1_p2 = p_curr + n_in * (self.track_width / 2)

            outer2_p1 = p_curr + n_out * (self.track_width / 2)
            outer2_p2 = p_next + n_out * (self.track_width / 2)

            inner1_p1 = p_prev - n_in * (self.track_width / 2)
            inner1_p2 = p_curr - n_in * (self.track_width / 2)

            inner2_p1 = p_curr - n_out * (self.track_width / 2)
            inner2_p2 = p_next - n_out * (self.track_width / 2)

            outer_corner = get_infinite_line_intersection(outer1_p1, outer1_p2, outer2_p1, outer2_p2)
            inner_corner = get_infinite_line_intersection(inner1_p1, inner1_p2, inner2_p1, inner2_p2)

            if outer_corner is None:
                outer_corner = outer1_p2
            if inner_corner is None:
                inner_corner = inner1_p2

            self.outer_points.append(outer_corner)
            self.inner_points.append(inner_corner)

        self._create_checkpoints()

    def _create_checkpoints(self):
        self.checkpoints = []
        for i in range(len(self.center_nodes)):
            p1 = self.inner_points[i]
            p2 = self.outer_points[i]
            self.checkpoints.append((p1, p2))

    def draw(self, screen):
        track_color = (100, 100, 100)  # Gray
        grass_color = (34, 139, 34)  # Forest green
        line_color = (255, 255, 255)  # White

        # Fill background with grass color
        screen.fill(grass_color)

        # Draw the track
        pygame.draw.polygon(screen, track_color, self.outer_points)
        pygame.draw.polygon(screen, grass_color, self.inner_points)

        # Draw the start/finish line
        pygame.draw.line(screen, line_color, self.start_finish_line[0], self.start_finish_line[1], 5)

        # for each of the checkpoints, draw a point
        self.draw_checkpoints(screen)

    def draw_checkpoints(self, screen):
        for i, (p1, p2) in enumerate(self.checkpoints):
            # Draw checkpoints in a different color for debugging
            color = (0, 255, 255) if i % 2 == 0 else (255, 0, 255)
            pygame.draw.line(screen, color, p1, p2, 2)

    def get_start_position(self):
        # Start in the middle of the last centerline segment
        p1 = self.center_nodes[-1]
        p2 = self.center_nodes[0]
        return ((p1.x + p2.x) / 2, (p1.y + p2.y) / 2)

    def get_start_angle(self):
        p1 = self.center_nodes[-1]
        p2 = self.center_nodes[0]
        dx = p2.x - p1.x
        dy = p2.y - p1.y
        return -math.degrees(math.atan2(dy, dx))
