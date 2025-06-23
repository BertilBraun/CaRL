import pygame
import math
import numpy as np
from utils.geometry import get_line_segment_intersection_fast
from typing import List, Tuple


class Track:
    def __init__(self, nodes: List[Tuple[float, float]]) -> None:
        self.width = 60
        self.nodes = [pygame.math.Vector2(p) for p in nodes]

        # Continuously add nodes at the halfway point of long segments
        # until all segments are shorter than 50 units.
        while True:
            nodes_added_in_pass = False
            index = 0
            while index < len(self.nodes) - 1:
                p1 = self.nodes[index]
                p2 = self.nodes[index + 1]
                if p1.distance_to(p2) > 50:
                    # Insert a new node at the midpoint of the segment
                    mid_point = p1.lerp(p2, 0.5)
                    self.nodes.insert(index + 1, mid_point)
                    nodes_added_in_pass = True
                index += 1

            if not nodes_added_in_pass:
                break

        self.outer_points: List[pygame.math.Vector2] = []
        self.inner_points: List[pygame.math.Vector2] = []
        self.checkpoints: List[Tuple[pygame.math.Vector2, pygame.math.Vector2]] = []

        self.segment_lengths: List[float] = []
        self.cumulative_lengths: List[float] = [0.0]
        self.total_length: float = 0.0

        self._calculate_path_lengths()
        self._generate_track_boundaries()

    def _calculate_path_lengths(self) -> None:
        """Calculates the length of each segment and the total path length."""
        for i in range(len(self.nodes) - 1):
            p1 = self.nodes[i]
            p2 = self.nodes[i + 1]
            length = p1.distance_to(p2)
            self.segment_lengths.append(length)

        self.total_length = sum(self.segment_lengths)

        # Calculate cumulative lengths for easy lookup
        cumulative = 0.0
        for length in self.segment_lengths:
            cumulative += length
            self.cumulative_lengths.append(cumulative)

    def _generate_track_boundaries(self) -> None:
        # Start cap
        v_start = (self.nodes[1] - self.nodes[0]).normalize()
        n_start = pygame.math.Vector2(-v_start.y, v_start.x)
        self.outer_points.append(self.nodes[0] - n_start * self.width / 2)
        self.inner_points.append(self.nodes[0] + n_start * self.width / 2)
        self.checkpoints.append((self.inner_points[0], self.outer_points[0]))

        # Intermediate segments
        for i in range(1, len(self.nodes) - 1):
            p_curr = self.nodes[i]
            p_prev = self.nodes[i - 1]
            p_next = self.nodes[i + 1]

            v_in = (p_curr - p_prev).normalize()
            v_out = (p_next - p_curr).normalize()
            n_in = pygame.math.Vector2(v_in.y, -v_in.x)
            n_out = pygame.math.Vector2(v_out.y, -v_out.x)

            outer1_p1 = p_prev + n_in * self.width / 2
            outer1_p2 = p_curr + n_in * self.width / 2
            outer2_p1 = p_curr + n_out * self.width / 2
            outer2_p2 = p_next + n_out * self.width / 2
            inner1_p1 = p_prev - n_in * self.width / 2
            inner1_p2 = p_curr - n_in * self.width / 2
            inner2_p1 = p_curr - n_out * self.width / 2
            inner2_p2 = p_next - n_out * self.width / 2

            outer1_p1_np = np.array([outer1_p1.x, outer1_p1.y], dtype=np.float32)
            outer1_p2_np = np.array([outer1_p2.x, outer1_p2.y], dtype=np.float32)
            outer2_p1_np = np.array([outer2_p1.x, outer2_p1.y], dtype=np.float32)
            outer2_p2_np = np.array([outer2_p2.x, outer2_p2.y], dtype=np.float32)
            inner1_p1_np = np.array([inner1_p1.x, inner1_p1.y], dtype=np.float32)
            inner1_p2_np = np.array([inner1_p2.x, inner1_p2.y], dtype=np.float32)
            inner2_p1_np = np.array([inner2_p1.x, inner2_p1.y], dtype=np.float32)
            inner2_p2_np = np.array([inner2_p2.x, inner2_p2.y], dtype=np.float32)

            outer_corner_np = get_line_segment_intersection_fast(outer1_p1_np, outer1_p2_np, outer2_p1_np, outer2_p2_np)
            if outer_corner_np is not None:
                outer_corner = pygame.math.Vector2(outer_corner_np[0], outer_corner_np[1])
            else:
                outer_corner = p_curr + n_in * self.width / 2

            inner_corner_np = get_line_segment_intersection_fast(inner1_p1_np, inner1_p2_np, inner2_p1_np, inner2_p2_np)
            if inner_corner_np is not None:
                inner_corner = pygame.math.Vector2(inner_corner_np[0], inner_corner_np[1])
            else:
                inner_corner = p_curr - n_in * self.width / 2

            self.outer_points.append(outer_corner)
            self.inner_points.append(inner_corner)
            self.checkpoints.append((inner_corner, outer_corner))

        # End cap
        v_end = (self.nodes[-1] - self.nodes[-2]).normalize()
        n_end = pygame.math.Vector2(-v_end.y, v_end.x)
        self.outer_points.append(self.nodes[-1] - n_end * self.width / 2)
        self.inner_points.append(self.nodes[-1] + n_end * self.width / 2)
        self.checkpoints.append((self.inner_points[-1], self.outer_points[-1]))

    def get_point_at_fraction(self, fraction: float) -> Tuple[pygame.math.Vector2, float, int]:
        """
        Gets a point on the track's centerline at a given fraction of its total length.
        Also returns the index of the next checkpoint from that point.
        """
        if not (0.0 <= fraction <= 1.0):
            raise ValueError('Fraction must be between 0.0 and 1.0')

        target_dist = self.total_length * fraction

        # Find which segment the target distance falls into
        segment_index = -1
        for i, cum_len in enumerate(self.cumulative_lengths):
            if target_dist <= cum_len:
                segment_index = i - 1
                break

        # This handles the case where fraction is 1.0
        if segment_index == -1:
            segment_index = len(self.nodes) - 2

        start_node = self.nodes[segment_index]
        end_node = self.nodes[segment_index + 1]

        dist_into_segment = target_dist - self.cumulative_lengths[segment_index]
        segment_vector = end_node - start_node

        # Interpolate the position
        point = start_node + segment_vector.normalize() * dist_into_segment

        # The next checkpoint is the end of the current segment
        next_checkpoint_index = segment_index + 1

        angle = -math.degrees(math.atan2(segment_vector.y, segment_vector.x))

        return point, angle, next_checkpoint_index

    def draw(self, screen: pygame.Surface) -> None:
        track_color = (100, 100, 100)
        grass_color = (34, 139, 34)

        screen.fill(grass_color)

        # Create a single polygon for the whole track for drawing
        track_poly = self.outer_points + self.inner_points[::-1]
        pygame.draw.polygon(screen, track_color, track_poly)

        # Draw start and finish lines
        pygame.draw.line(screen, (0, 255, 0), self.checkpoints[0][0], self.checkpoints[0][1], 5)  # Green for start
        pygame.draw.line(screen, (255, 0, 0), self.checkpoints[-1][0], self.checkpoints[-1][1], 5)  # Red for finish

        # Draw the checkpoints
        for checkpoint in self.checkpoints:
            pygame.draw.line(screen, (255, 255, 255), checkpoint[0], checkpoint[1], 5)
