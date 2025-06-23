import pygame
import math
from utils.geometry import get_infinite_line_intersection
from typing import List, Tuple


class Track:
    def __init__(self, nodes: List[Tuple[float, float]]) -> None:
        self.width = 60
        self.nodes = [pygame.math.Vector2(p) for p in nodes]

        self.outer_points: List[pygame.math.Vector2] = []
        self.inner_points: List[pygame.math.Vector2] = []
        self.checkpoints: List[Tuple[pygame.math.Vector2, pygame.math.Vector2]] = []
        self._generate_track_boundaries()

    def _generate_track_boundaries(self) -> None:
        # Convert node tuples to Vector2 for easier math
        self.outer_points = []
        self.inner_points = []

        for i in range(len(self.nodes)):
            p_curr = self.nodes[i]
            p_prev = self.nodes[i - 1]  # Wraps around for the last point
            p_next = self.nodes[(i + 1) % len(self.nodes)]

            # Vector from previous to current, and from current to next
            v_in = (p_curr - p_prev).normalize()
            v_out = (p_next - p_curr).normalize()

            # Correctly calculate outward-pointing normal for a CW track
            n_in = pygame.math.Vector2(v_in.y, -v_in.x)
            n_out = pygame.math.Vector2(v_out.y, -v_out.x)

            # Mitre joint calculation
            # For each corner, we define two infinite lines for the outer boundary
            # and two for the inner boundary, then find their intersection.
            outer1_p1 = p_prev + n_in * (self.width / 2)
            outer1_p2 = p_curr + n_in * (self.width / 2)

            outer2_p1 = p_curr + n_out * (self.width / 2)
            outer2_p2 = p_next + n_out * (self.width / 2)

            inner1_p1 = p_prev - n_in * (self.width / 2)
            inner1_p2 = p_curr - n_in * (self.width / 2)

            inner2_p1 = p_curr - n_out * (self.width / 2)
            inner2_p2 = p_next - n_out * (self.width / 2)

            outer_corner = get_infinite_line_intersection(outer1_p1, outer1_p2, outer2_p1, outer2_p2)
            if outer_corner:
                self.outer_points.append(outer_corner)
            else:
                # Fallback for parallel lines (though unlikely with this logic)
                self.outer_points.append(p_curr + n_in * self.width / 2)

            inner_corner = get_infinite_line_intersection(inner1_p1, inner1_p2, inner2_p1, inner2_p2)
            if inner_corner:
                self.inner_points.append(inner_corner)
            else:
                # Fallback
                self.inner_points.append(p_curr - n_in * self.width / 2)

        # Create checkpoints from the boundary points
        self.checkpoints = []
        for i in range(len(self.inner_points)):
            p1 = self.inner_points[i]
            p2 = self.outer_points[i]
            self.checkpoints.append((p1, p2))

    def draw(self, screen: pygame.Surface) -> None:
        track_color = (100, 100, 100)  # Gray
        grass_color = (34, 139, 34)  # Forest green
        line_color = (255, 255, 255)  # White

        # Fill background with grass color
        screen.fill(grass_color)

        # Draw the track
        pygame.draw.polygon(screen, track_color, self.outer_points)
        pygame.draw.polygon(screen, grass_color, self.inner_points)

        # Draw the start/finish line
        start_pos_vec = pygame.math.Vector2(self.get_start_position())
        direction = (self.nodes[0] - self.nodes[-1]).normalize()
        perp_vec = pygame.math.Vector2(direction.y, -direction.x)  # Use the correct normal for CW
        self.start_finish_line = (
            start_pos_vec + perp_vec * (self.width / 2),
            start_pos_vec - perp_vec * (self.width / 2),
        )

        pygame.draw.line(screen, line_color, self.start_finish_line[0], self.start_finish_line[1], 5)

        # for each of the checkpoints, draw a point
        for i, (p1, p2) in enumerate(self.checkpoints):
            # Draw checkpoints in a different color for debugging
            color = (0, 255, 255) if i % 2 == 0 else (255, 0, 255)
            pygame.draw.line(screen, color, p1, p2, 2)

    def get_start_position(self) -> pygame.math.Vector2:
        # Start in the middle of the last centerline segment
        return (self.nodes[0] + self.nodes[-1]) / 2

    def get_start_angle(self) -> float:
        d = self.nodes[0] - self.nodes[-1]
        return -math.degrees(math.atan2(d.y, d.x))
