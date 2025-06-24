import numba
import pygame
import numpy as np
from typing import Callable, List, Tuple


def vector_to_numpy(v: pygame.math.Vector2) -> np.ndarray:
    """Converts a pygame Vector2 to a numpy array."""
    return np.array([v.x, v.y], dtype=np.float32)


def numpy_to_vector(a: np.ndarray) -> pygame.math.Vector2:
    """Converts a 2-element numpy array to a pygame Vector2."""
    return pygame.math.Vector2(a[0], a[1])


def get_line_segment_intersection(
    p1: pygame.math.Vector2, p2: pygame.math.Vector2, p3: pygame.math.Vector2, p4: pygame.math.Vector2
) -> pygame.math.Vector2 | None:
    """
    Calculates the intersection point of two finite line segments using Numba.
    :return: A Vector2 of the intersection point, or None if they do not intersect.
    """
    den = (p1.x - p2.x) * (p3.y - p4.y) - (p1.y - p2.y) * (p3.x - p4.x)
    if den == 0:
        return None

    t_num = (p1.x - p3.x) * (p3.y - p4.y) - (p1.y - p3.y) * (p3.x - p4.x)
    t = t_num / den

    u_num = -((p1.x - p2.x) * (p1.y - p3.y) - (p1.y - p2.y) * (p1.x - p3.x))
    u = u_num / den

    if 0 < t < 1 and 0 < u < 1:
        return p1 + t * (p2 - p1)

    return None


@numba.njit(fastmath=True, cache=True)
def get_corners_numba(x: float, y: float, angle: float, length: float, width: float) -> np.ndarray:
    center = np.array([x, y], dtype=np.float32)
    angle_rad = np.radians(-angle)

    half_len = length / 2
    half_wid = width / 2

    corners = np.array(
        [
            [-half_len, -half_wid],
            [half_len, -half_wid],
            [half_len, half_wid],
            [-half_len, half_wid],
        ],
        dtype=np.float32,
    )

    rotation_matrix = np.array(
        [[np.cos(angle_rad), -np.sin(angle_rad)], [np.sin(angle_rad), np.cos(angle_rad)]], dtype=np.float32
    )

    rotated_corners = corners @ rotation_matrix.T
    return rotated_corners + center


@numba.njit(fastmath=True, cache=True)
def get_line_segment_intersection_fast(p1: np.ndarray, p2: np.ndarray, p3s: np.ndarray, p4s: np.ndarray):
    """
    Calculates the intersection of a line segment with a batch of other line segments.
    """
    den = (p1[0] - p2[0]) * (p3s[:, 1] - p4s[:, 1]) - (p1[1] - p2[1]) * (p3s[:, 0] - p4s[:, 0])

    t_num = (p1[0] - p3s[:, 0]) * (p3s[:, 1] - p4s[:, 1]) - (p1[1] - p3s[:, 1]) * (p3s[:, 0] - p4s[:, 0])
    u_num = -((p1[0] - p2[0]) * (p1[1] - p3s[:, 1]) - (p1[1] - p2[1]) * (p1[0] - p3s[:, 0]))

    # Using a small epsilon to avoid division by zero
    den[np.abs(den) < 1e-6] = 1e-6

    t = t_num / den
    u = u_num / den

    mask = (t > 0) & (t < 1) & (u > 0) & (u < 1)

    if not np.any(mask):
        return None, None

    valid_t = t[mask]
    min_t_idx = np.argmin(valid_t)

    closest_intersection = p1 + valid_t[min_t_idx] * (p2 - p1)
    distance = np.linalg.norm(closest_intersection - p1)

    return closest_intersection, distance


def get_fast_collision_checker(
    checkpoints: List[Tuple[pygame.math.Vector2, pygame.math.Vector2]],
) -> Callable[[np.ndarray, int], bool]:
    """Creates a fast collision checker function that can be used to check if a car is colliding with the track boundaries.
    The checker takes the car's points and the current checkpoint, and returns a boolean indicating if the car is colliding.
    """

    checkpoints_np = np.array([[(cp[0].x, cp[0].y), (cp[1].x, cp[1].y)] for cp in checkpoints], dtype=np.float32)
    num_checkpoints = len(checkpoints)

    @numba.njit(fastmath=True, cache=True)
    def check_collision_fast(car_points: np.ndarray, next_checkpoint: int) -> bool:
        window_size = 5
        start_idx = max(0, next_checkpoint - window_size)
        end_idx = min(num_checkpoints - 1, next_checkpoint + window_size)

        for i in range(len(car_points)):
            point = car_points[i]
            is_in_any_poly = False
            for j in range(start_idx, end_idx):
                p_curr_inner = checkpoints_np[j, 0]
                p_curr_outer = checkpoints_np[j, 1]
                p_next_inner = checkpoints_np[j + 1, 0]
                p_next_outer = checkpoints_np[j + 1, 1]

                local_poly = np.empty((4, 2), dtype=np.float32)
                local_poly[0, :] = p_next_outer
                local_poly[1, :] = p_curr_outer
                local_poly[2, :] = p_curr_inner
                local_poly[3, :] = p_next_inner

                if point_in_polygon_fast(point, local_poly):
                    is_in_any_poly = True
                    break

            if not is_in_any_poly:
                return True

        return False

    return check_collision_fast


def get_fast_lidar_reader(
    outer_points: List[pygame.math.Vector2],
    inner_points: List[pygame.math.Vector2],
    num_rays: int = 5,
    ray_length: float = 300.0,
    vicinity: int = 10,
) -> Callable[[float, pygame.math.Vector2, int], Tuple[List[float], List[pygame.math.Vector2]]]:
    """Creates a fast lidar reader function that can be used to get lidar readings for a car against track boundaries.
    The reader takes the car's angle, position, and current checkpoint, and returns a list of lidar distances and end points.
    """

    def _get_track_lines() -> np.ndarray:
        outer_points_np = np.array([(p.x, p.y) for p in outer_points], dtype=np.float32)
        inner_points_np = np.array([(p.x, p.y) for p in inner_points], dtype=np.float32)

        lines = []
        # Outer boundary lines
        for i in range(len(outer_points_np) - 1):
            p1 = outer_points_np[i]
            p2 = outer_points_np[i + 1]
            lines.append([p1, p2])
        # Inner boundary lines
        for i in range(len(inner_points_np) - 1):
            p1 = inner_points_np[i]
            p2 = inner_points_np[i + 1]
            lines.append([p1, p2])

        return np.array(lines, dtype=np.float32)

    # Cache numpy versions of track boundaries for Numba
    track_lines = _get_track_lines()

    @numba.njit(fastmath=True, cache=True)
    def get_lidar_readings_fast(
        car_angle: float,
        car_position: np.ndarray,
        current_checkpoint: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculates lidar readings for a car against track boundaries using Numba.
        This version is vectorized to check all track lines at once for each ray.
        It only considers track lines in the vicinity of the car's current checkpoint.
        """
        angles = np.linspace(-90, 90, num_rays)
        readings = np.full(num_rays, ray_length, dtype=np.float32)
        lidar_end_points = np.zeros((num_rays, 2), dtype=np.float32)

        start_pos = car_position

        num_track_lines_total = track_lines.shape[0]
        num_segments_per_side = num_track_lines_total // 2

        start_idx = max(0, current_checkpoint - vicinity)
        end_idx = min(num_segments_per_side, current_checkpoint + vicinity)

        outer_indices = np.arange(start_idx, end_idx)
        inner_indices = np.arange(num_segments_per_side + start_idx, num_segments_per_side + end_idx)

        indices = np.concatenate((outer_indices, inner_indices))

        relevant_track_lines = track_lines[indices]
        p3s = relevant_track_lines[:, 0, :]
        p4s = relevant_track_lines[:, 1, :]

        ray_angles_rad = np.radians(car_angle + angles)
        cos_rays = np.cos(ray_angles_rad)
        sin_rays = np.sin(ray_angles_rad)

        for i in range(num_rays):
            end_pos_long = start_pos + np.array([cos_rays[i] * ray_length, -sin_rays[i] * ray_length])

            intersection, distance = get_line_segment_intersection_fast(start_pos, end_pos_long, p3s, p4s)

            if intersection is not None:
                readings[i] = distance
                lidar_end_points[i, :] = intersection
            else:
                readings[i] = ray_length
                lidar_end_points[i, :] = end_pos_long

        return readings / ray_length, lidar_end_points

    def lidar_reader(
        car_angle: float, car_position: pygame.math.Vector2, current_checkpoint: int
    ) -> Tuple[List[float], List[pygame.math.Vector2]]:
        car_pos_np = vector_to_numpy(car_position)
        readings, end_points = get_lidar_readings_fast(car_angle, car_pos_np, current_checkpoint)
        return readings.tolist(), [numpy_to_vector(p) for p in end_points]

    return lidar_reader


@numba.njit(fastmath=True, cache=True)
def point_in_polygon_fast(point: np.ndarray, polygon: np.ndarray) -> bool:
    """
    Checks if a point is inside a polygon using the ray-casting algorithm with Numba.
    """
    x, y = point[0], point[1]
    n = len(polygon)
    inside = False
    p1 = polygon[0]
    for i in range(n + 1):
        p2 = polygon[i % n]
        if y > min(p1[1], p2[1]):
            if y <= max(p1[1], p2[1]):
                if x <= max(p1[0], p2[0]):
                    xinters = 0.0
                    if p1[1] != p2[1]:
                        xinters = (y - p1[1]) * (p2[0] - p1[0]) / (p2[1] - p1[1]) + p1[0]

                    if p1[0] == p2[0] or x <= xinters:
                        inside = not inside
        p1 = p2
    return inside


def find_closest_point_on_segment(
    p: pygame.math.Vector2, a: pygame.math.Vector2, b: pygame.math.Vector2
) -> pygame.math.Vector2:
    """Finds the closest point on a line segment to a given point."""
    line_vec = b - a
    p_vec = p - a
    line_len_sq = line_vec.length_squared()

    if line_len_sq == 0:
        return a

    t = p_vec.dot(line_vec) / line_len_sq
    t = max(0, min(1, t))  # Clamp to segment

    return a + t * line_vec
