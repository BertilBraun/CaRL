import numba
import numpy as np
from typing import Tuple, Optional
import pygame


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


@numba.njit
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


@numba.njit
def get_lidar_readings_fast(
    car_angle: float,
    car_position: np.ndarray,
    track_lines: np.ndarray,
    num_rays: int = 5,
    ray_length: float = 300.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculates lidar readings for a car against track boundaries using Numba.
    This version is vectorized to check all track lines at once for each ray.
    """
    angles = np.linspace(-90, 90, num_rays)
    readings = np.full(num_rays, ray_length, dtype=np.float32)
    lidar_end_points = np.zeros((num_rays, 2), dtype=np.float32)

    start_pos = car_position
    p3s = track_lines[:, 0, :]
    p4s = track_lines[:, 1, :]

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


@numba.njit
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
