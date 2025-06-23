import pygame
import math
from utils.geometry import (
    get_line_segment_intersection,
    numpy_to_vector,
    vector_to_numpy,
    get_lidar_readings_fast,
    point_in_polygon_fast,
    find_closest_point_on_segment,
)
from typing import List, Tuple, TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from game.car import Car


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
                if p1.distance_to(p2) > 100:
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

        # Cache numpy versions of track boundaries for Numba
        self.outer_points_np = np.array([(p.x, p.y) for p in self.outer_points], dtype=np.float32)
        self.inner_points_np = np.array([(p.x, p.y) for p in self.inner_points], dtype=np.float32)
        self.track_lines_np = self._get_track_lines()

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

    def _get_track_lines(self) -> np.ndarray:
        lines = []
        # Outer boundary lines
        for i in range(len(self.outer_points_np)):
            p1 = self.outer_points_np[i - 1]
            p2 = self.outer_points_np[i]
            lines.append([p1, p2])
        # Inner boundary lines
        for i in range(len(self.inner_points_np)):
            p1 = self.inner_points_np[i - 1]
            p2 = self.inner_points_np[i]
            lines.append([p1, p2])

        return np.array(lines, dtype=np.float32)

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

            outer_corner = get_line_segment_intersection(outer1_p1, outer1_p2, outer2_p1, outer2_p2)
            if outer_corner is None:
                outer_corner = p_curr + n_in * self.width / 2

            inner_corner = get_line_segment_intersection(inner1_p1, inner1_p2, inner2_p1, inner2_p2)
            if inner_corner is None:
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

    def get_lidar_readings(self, car: 'Car') -> Tuple[List[float], List[pygame.math.Vector2]]:
        car_pos_np = vector_to_numpy(car.position)
        readings, end_points = get_lidar_readings_fast(car.angle, car_pos_np, self.track_lines_np)
        return readings.tolist(), [numpy_to_vector(p) for p in end_points]

    def check_collision(self, car: 'Car', next_checkpoint: int) -> bool:
        car_points = car.get_corners_np()

        num_checkpoints = len(self.checkpoints)
        next_cp = next_checkpoint

        # Define indices for the 3-segment window, clamping at the track ends
        prev_prev_cp = max(0, next_cp - 2)
        prev_cp = max(0, next_cp - 1)
        next_next_cp = min(num_checkpoints - 1, next_cp + 1)

        # Define the three local track segments using numpy
        p_prev_prev_inner = np.array(self.checkpoints[prev_prev_cp][0], dtype=np.float32)
        p_prev_prev_outer = np.array(self.checkpoints[prev_prev_cp][1], dtype=np.float32)
        p_prev_inner = np.array(self.checkpoints[prev_cp][0], dtype=np.float32)
        p_prev_outer = np.array(self.checkpoints[prev_cp][1], dtype=np.float32)
        p_next_inner = np.array(self.checkpoints[next_cp][0], dtype=np.float32)
        p_next_outer = np.array(self.checkpoints[next_cp][1], dtype=np.float32)
        p_next_next_inner = np.array(self.checkpoints[next_next_cp][0], dtype=np.float32)
        p_next_next_outer = np.array(self.checkpoints[next_next_cp][1], dtype=np.float32)

        # Polygons are defined in a consistent order (e.g., CCW)
        local_poly1 = np.array([p_prev_outer, p_prev_prev_outer, p_prev_prev_inner, p_prev_inner])
        local_poly2 = np.array([p_next_outer, p_prev_outer, p_prev_inner, p_next_inner])
        local_poly3 = np.array([p_next_next_outer, p_next_outer, p_next_inner, p_next_next_inner])

        for point in car_points:
            in_poly1 = point_in_polygon_fast(point, local_poly1)
            in_poly2 = point_in_polygon_fast(point, local_poly2)
            in_poly3 = point_in_polygon_fast(point, local_poly3)

            if not in_poly1 and not in_poly2 and not in_poly3:
                return True

        return False

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

    def get_progress_on_track(self, point: pygame.math.Vector2) -> float:
        """
        Calculates the progress of a point along the track's centerline, returned as a fraction of total length.
        """
        min_dist_sq = float('inf')
        closest_proj_point = None
        closest_segment_index = -1

        # Find the closest segment on the track's centerline
        for i in range(len(self.nodes) - 1):
            p1 = self.nodes[i]
            p2 = self.nodes[i + 1]
            proj_point = find_closest_point_on_segment(point, p1, p2)
            dist_sq = point.distance_squared_to(proj_point)

            if dist_sq < min_dist_sq:
                min_dist_sq = dist_sq
                closest_proj_point = proj_point
                closest_segment_index = i

        if closest_segment_index == -1 or closest_proj_point is None:
            return 0.0

        # Calculate the distance along the track to this point
        dist_along_track = self.cumulative_lengths[closest_segment_index]
        dist_along_track += self.nodes[closest_segment_index].distance_to(closest_proj_point)

        return dist_along_track / self.total_length

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
