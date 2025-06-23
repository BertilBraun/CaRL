import pygame
import math
import numpy as np
from .track import Track
from typing import List, Tuple, Dict
import numba

from utils.geometry import (
    get_line_segment_intersection_fast,
    get_lidar_readings_fast,
    point_in_polygon_fast,
)

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
        self.track_lines_np = self._get_track_lines()

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

        # Prevent checking for checkpoints that don't exist
        if racer.next_checkpoint >= len(self.track.checkpoints):
            return 0.0

        next_checkpoint_line = self.track.checkpoints[racer.next_checkpoint]
        p3 = next_checkpoint_line[0]
        p4 = next_checkpoint_line[1]

        p1_np = np.array([p1.x, p1.y], dtype=np.float32)
        p2_np = np.array([p2.x, p2.y], dtype=np.float32)
        p3_np = np.array([p3.x, p3.y], dtype=np.float32)
        p4_np = np.array([p4.x, p4.y], dtype=np.float32)

        if get_line_segment_intersection_fast(p1_np, p2_np, p3_np, p4_np) is not None:
            # If it's the final checkpoint, give a large reward
            if racer.next_checkpoint == len(self.track.checkpoints) - 1:
                racer.done = True  # Mark as finished
                reward = 1000.0
            else:
                reward = 50.0

            racer.next_checkpoint += 1
            racer.time_since_checkpoint = 0

        return reward

    def _check_collision(self, racer: 'Racer') -> bool:
        car_points = self._get_car_corners_np(racer.car)

        num_checkpoints = len(self.track.checkpoints)
        next_cp = racer.next_checkpoint

        # Define indices for the 3-segment window, clamping at the track ends
        prev_prev_cp = max(0, next_cp - 2)
        prev_cp = max(0, next_cp - 1)
        next_next_cp = min(num_checkpoints - 1, next_cp + 1)

        # Define the three local track segments using numpy
        p_prev_prev_inner = np.array(self.track.checkpoints[prev_prev_cp][0], dtype=np.float32)
        p_prev_prev_outer = np.array(self.track.checkpoints[prev_prev_cp][1], dtype=np.float32)
        p_prev_inner = np.array(self.track.checkpoints[prev_cp][0], dtype=np.float32)
        p_prev_outer = np.array(self.track.checkpoints[prev_cp][1], dtype=np.float32)
        p_next_inner = np.array(self.track.checkpoints[next_cp][0], dtype=np.float32)
        p_next_outer = np.array(self.track.checkpoints[next_cp][1], dtype=np.float32)
        p_next_next_inner = np.array(self.track.checkpoints[next_next_cp][0], dtype=np.float32)
        p_next_next_outer = np.array(self.track.checkpoints[next_next_cp][1], dtype=np.float32)

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

    def _get_car_corners_np(self, car: 'Car') -> np.ndarray:
        center = np.array([car.position.x, car.position.y], dtype=np.float32)
        angle_rad = np.radians(car.angle)

        half_len = car.length / 2
        half_wid = car.width / 2

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

    def _get_lidar_readings(self, car: 'Car') -> Tuple[List[float], List[pygame.math.Vector2]]:
        car_pos_np = np.array([car.position.x, car.position.y], dtype=np.float32)
        readings, end_points = get_lidar_readings_fast(car.angle, car_pos_np, self.track_lines_np)
        return readings.tolist(), [pygame.math.Vector2(x, y) for x, y in end_points]

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
