import pygame
import math
import numpy as np
from .track import Track
from utils.geometry import get_line_segment_intersection
from typing import List, Tuple, Dict

# Guard imports for type-hinting to prevent circular dependencies
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agent.racer import Racer
    from game.car import Car


class GameEnvironment:
    def __init__(self, track: Track, timeout_to_reach_next_checkpoint: int) -> None:
        self.track = track
        self.timeout_to_reach_next_checkpoint = timeout_to_reach_next_checkpoint
        self.action_map: Dict[int, Tuple[float, float]] = {
            0: (0.0, 0.0),  # coast
            1: (1.0, 0.0),  # accelerate
            2: (-1.0, 0.0),  # brake
            3: (0.0, 1.0),  # steer left
            4: (0.0, -1.0),  # steer right
            5: (1.0, 1.0),  # accelerate and steer left
            6: (1.0, -1.0),  # accelerate and steer right
        }

        # Cache numpy versions of track boundaries for Numba
        self.outer_points_np = np.array([(p.x, p.y) for p in self.track.outer_points], dtype=np.float32)
        self.inner_points_np = np.array([(p.x, p.y) for p in self.track.inner_points], dtype=np.float32)

    def step(self, racer: 'Racer', action: int) -> Tuple[List[float], float, bool]:
        # Update car physics
        acceleration_input, steering_input = self.action_map[action]
        racer.car.update(acceleration_input, steering_input)

        # --- Reward Calculation ---
        # 1. Calculate reward for progressing towards the current next checkpoint
        next_checkpoint_line = self.track.checkpoints[racer.next_checkpoint]
        checkpoint_center = (next_checkpoint_line[0] + next_checkpoint_line[1]) / 2
        dist_before = racer.last_pos.distance_to(checkpoint_center)
        dist_now = racer.car.position.distance_to(checkpoint_center)
        progress_reward = dist_before - dist_now

        # 2. Check for events and sparse rewards
        done = self._check_collision(racer)
        checkpoint_reward = self._check_checkpoint_crossing(racer)

        # 3. Combine rewards into a final reward value
        reward = self._calculate_reward(done, checkpoint_reward, progress_reward)

        # --- State and Updates ---
        state = self._get_state(racer)

        # Update racer state for the next step
        racer.last_pos = racer.car.position.copy()
        racer.time_since_checkpoint += 1

        if racer.time_since_checkpoint > self.timeout_to_reach_next_checkpoint:
            done = True

        return state, reward, done

    def _check_checkpoint_crossing(self, racer: 'Racer') -> float:
        reward = 0
        p1 = racer.last_pos
        p2 = racer.car.position

        next_checkpoint_line = self.track.checkpoints[racer.next_checkpoint]
        p3 = next_checkpoint_line[0]
        p4 = next_checkpoint_line[1]

        if get_line_segment_intersection(p1, p2, p3, p4):
            racer.next_checkpoint += 1
            racer.time_since_checkpoint = 0
            reward = 50

            if racer.next_checkpoint >= len(self.track.checkpoints):
                racer.next_checkpoint = 0
                racer.lap_count += 1
                reward = 1000

        return reward

    def _check_collision(self, racer: 'Racer') -> bool:
        """
        Checks if the car is within the local track area, defined by two polygons
        spanning from the previous-previous to the next checkpoint.
        A point is valid if it's in *either* of these two polygons.
        """
        car_points = self._get_car_corners(racer.car)

        num_checkpoints = len(self.track.checkpoints)
        next_cp = racer.next_checkpoint
        prev_cp = (next_cp - 1 + num_checkpoints) % num_checkpoints
        next_next_cp = (next_cp + 1 + num_checkpoints) % num_checkpoints
        prev_prev_cp = (next_cp - 2 + num_checkpoints) % num_checkpoints

        # Define the three local track segments
        p_prev_prev_inner, p_prev_prev_outer = self.track.checkpoints[prev_prev_cp]
        p_prev_inner, p_prev_outer = self.track.checkpoints[prev_cp]
        p_next_inner, p_next_outer = self.track.checkpoints[next_cp]
        p_next_next_inner, p_next_next_outer = self.track.checkpoints[next_next_cp]

        # Polygons are defined in a consistent order (e.g., CCW)
        local_poly1 = [p_prev_outer, p_prev_prev_outer, p_prev_prev_inner, p_prev_inner]
        local_poly2 = [p_next_outer, p_prev_outer, p_prev_inner, p_next_inner]
        local_poly3 = [p_next_next_outer, p_next_outer, p_next_inner, p_next_next_inner]

        for point in car_points:
            # A car point is valid if it's inside either of the two local polygons
            in_poly1 = self._point_in_polygon(point, local_poly1)
            in_poly2 = self._point_in_polygon(point, local_poly2)
            in_poly3 = self._point_in_polygon(point, local_poly3)

            if not in_poly1 and not in_poly2 and not in_poly3:
                return True  # Crashed: The point is outside all valid areas.

        return False  # Not crashed

    def _point_in_polygon(self, point: pygame.math.Vector2, polygon: List[pygame.math.Vector2]) -> bool:
        x, y = point.x, point.y
        n = len(polygon)
        inside = False
        p1 = polygon[0]
        for i in range(n + 1):
            p2 = polygon[i % n]
            if y > min(p1.y, p2.y):
                if y <= max(p1.y, p2.y):
                    if x <= max(p1.x, p2.x):
                        xinters = 0.0
                        if p1.y != p2.y:
                            xinters = (y - p1.y) * (p2.x - p1.x) / (p2.y - p1.y) + p1.x

                        if p1.x == p2.x or x <= xinters:
                            inside = not inside
            p1 = p2
        return inside

    def _get_car_corners(self, car: 'Car') -> List[pygame.math.Vector2]:
        center = car.position
        length = car.length
        width = car.width
        angle = car.angle

        half_len = length / 2
        half_wid = width / 2

        corners = [
            pygame.math.Vector2(-half_len, -half_wid),
            pygame.math.Vector2(half_len, -half_wid),
            pygame.math.Vector2(half_len, half_wid),
            pygame.math.Vector2(-half_len, half_wid),
        ]

        corners = [p.rotate(-angle) for p in corners]
        corners = [p + center for p in corners]

        return corners

    def _get_lidar_readings(self, car: 'Car') -> Tuple[List[float], List[pygame.math.Vector2]]:
        num_rays = 5
        ray_length = 300.0
        angles = np.linspace(-90, 90, num_rays)

        readings = []
        lidar_end_points = []

        track_lines = self._get_track_lines()

        for angle in angles:
            ray_angle = car.angle + angle
            start_pos = car.position
            end_pos_long = start_pos + pygame.math.Vector2(ray_length, 0).rotate(-ray_angle)

            closest_dist = ray_length
            closest_point = end_pos_long

            for line in track_lines:
                intersection = get_line_segment_intersection(
                    start_pos, end_pos_long, pygame.math.Vector2(line[0]), pygame.math.Vector2(line[1])
                )
                if intersection:
                    dist = start_pos.distance_to(intersection)
                    if dist < closest_dist:
                        closest_dist = dist
                        closest_point = intersection

            readings.append(closest_dist / ray_length)
            lidar_end_points.append(closest_point)

        return readings, lidar_end_points

    def _get_track_lines(self) -> List[Tuple[pygame.math.Vector2, pygame.math.Vector2]]:
        lines = []
        for i in range(len(self.track.outer_points)):
            lines.append((self.track.outer_points[i - 1], self.track.outer_points[i]))
        for i in range(len(self.track.inner_points)):
            lines.append((self.track.inner_points[i - 1], self.track.inner_points[i]))
        return lines

    def _calculate_reward(self, done: bool, checkpoint_reward: float, progress_reward: float) -> float:
        if done:
            return -100.0

        time_penalty = -0.1

        return checkpoint_reward + progress_reward + time_penalty

    def _get_state(self, racer: 'Racer') -> List[float]:
        lidar_readings, _ = self._get_lidar_readings(racer.car)

        # Normalize velocity into forward and sideways components relative to car's orientation
        velocity_vec = racer.car.position - racer.last_pos

        angle_rad = math.radians(racer.car.angle)
        forward_vec = pygame.math.Vector2(math.cos(angle_rad), -math.sin(angle_rad))

        # Project velocity vector onto forward vector to get forward speed
        forward_speed = velocity_vec.dot(forward_vec) / racer.car.max_velocity

        # Use the cross product (in 2D, this is a scalar) to find sideways speed
        sideways_speed = (forward_vec.x * velocity_vec.y - forward_vec.y * velocity_vec.x) / racer.car.max_velocity

        return [forward_speed, sideways_speed] + lidar_readings

    def draw(self, screen: pygame.Surface, racers: List['Racer']) -> None:
        self.track.draw(screen)
        for racer in racers:
            racer.car.draw(screen)
            self._draw_lidar(screen, racer.car)

    def _draw_lidar(self, screen: pygame.Surface, car: 'Car') -> None:
        _, lidar_end_points = self._get_lidar_readings(car)
        for point in lidar_end_points:
            pygame.draw.line(screen, (0, 255, 0), car.position, point, 1)
