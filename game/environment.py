from __future__ import annotations

import pygame
import math
from dataclasses import dataclass
from typing import List, Tuple, Dict

from .track import Track

from utils.geometry import find_closest_point_on_segment, get_line_segment_intersection

# Guard imports for type-hinting to prevent circular dependencies
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agent.racer import Racer


@dataclass
class CarMetrics:
    forward_speed: float
    distance_to_centerline: float
    angle_to_centerline: float
    closest_point_on_centerline: pygame.Vector2


class GameEnvironment:
    ACTION_MAP: Dict[int, Tuple[float, float]] = {
        0: (0.0, 0.0),  # coast
        1: (1.0, 0.0),  # accelerate
        2: (-1.0, 0.0),  # brake
        3: (0.0, 1.0),  # steer left
        4: (0.0, -1.0),  # steer right
        5: (1.0, 1.0),  # accelerate and steer left
        6: (1.0, -1.0),  # accelerate and steer right
        7: (-1.0, 1.0),  # brake + left
        8: (-1.0, -1.0),  # brake + right
    }

    def __init__(self, track: Track, timeout_to_do_something: int) -> None:
        self.track = track
        self.timeout_to_do_something = timeout_to_do_something

    def step(self, racer: Racer, action: int) -> Tuple[List[float], float, bool]:
        # Update car physics
        acceleration_input, steering_input = self.ACTION_MAP[action]
        racer.car.update(acceleration_input, steering_input)

        # Pre-calculate metrics for reward and state
        metrics = self._calculate_car_metrics(racer)

        # Check for events
        collision = self.track.check_collision(racer.car, racer.next_checkpoint)
        checkpoint_reward = self._check_checkpoint_crossing(racer)

        # Determine if the episode is done and get terminal reward
        done, terminal_reward = self._is_done(racer, checkpoint_reward, collision, metrics.forward_speed)

        # Calculate reward
        reward = self._calculate_reward(racer, checkpoint_reward, done, terminal_reward, metrics)

        # Get next state
        state = self._get_state(racer, metrics)

        # Update racer for next step
        racer.last_pos = racer.car.position.copy()

        return state, reward, done

    def get_state(self, racer: Racer) -> List[float]:
        metrics = self._calculate_car_metrics(racer)
        return self._get_state(racer, metrics)

    def _calculate_car_metrics(self, racer: Racer) -> CarMetrics:
        """Calculates various metrics about the car's state and its relation to the track."""
        velocity_vec = racer.car.position - racer.last_pos

        # Toward next checkpoint
        next_checkpoint_line = self.track.checkpoints[racer.next_checkpoint]
        checkpoint_center = (next_checkpoint_line[0] + next_checkpoint_line[1]) / 2
        to_checkpoint_vec = checkpoint_center - racer.car.position
        if to_checkpoint_vec.length() != 0:
            to_checkpoint_dir = to_checkpoint_vec.normalize()
            forward_speed = velocity_vec.dot(to_checkpoint_dir)
        else:
            forward_speed = 0.0

        # Centerline projection
        prev_checkpoint = self.track.nodes[max(0, racer.next_checkpoint - 1)]
        next_checkpoint = self.track.nodes[racer.next_checkpoint]
        closest_point_on_centerline = find_closest_point_on_segment(
            racer.car.position, prev_checkpoint, next_checkpoint
        )
        distance_to_centerline = racer.car.position.distance_to(closest_point_on_centerline)

        # Angle to centerline
        angle_to_centerline = math.atan2(
            closest_point_on_centerline.y - racer.car.position.y,
            closest_point_on_centerline.x - racer.car.position.x,
        )

        return CarMetrics(
            forward_speed=forward_speed,
            distance_to_centerline=distance_to_centerline,
            angle_to_centerline=angle_to_centerline,
            closest_point_on_centerline=closest_point_on_centerline,
        )

    def _is_done(
        self, racer: Racer, checkpoint_reward: float, collision: bool, forward_speed: float
    ) -> Tuple[bool, float]:
        """Checks for terminal conditions and returns a terminal reward if applicable."""
        racer.time_since_checkpoint += 1
        if forward_speed < 0.8:
            racer.time_since_last_movement += 1
        else:
            racer.time_since_last_movement = 0

        if collision:
            return True, -500.0
        if racer.done:  # Racer finished the track
            return True, checkpoint_reward  # Final reward is given by checkpoint crossing
        if racer.time_since_checkpoint > self.timeout_to_do_something:
            return True, -100.0  # Penalty for being too slow
        if racer.time_since_last_movement > self.timeout_to_do_something:
            return True, -800.0  # Penalty for getting stuck

        return False, 0.0

    def _calculate_reward(
        self, racer: Racer, checkpoint_reward: float, done: bool, terminal_reward: float, metrics: CarMetrics
    ) -> float:
        """Calculates the reward for the current step."""
        if done:
            return terminal_reward

        # Calculate progress along the centerline
        prev_centerline_node = self.track.nodes[racer.next_checkpoint - 1]
        next_centerline_node = self.track.nodes[racer.next_checkpoint]
        closest_point_last_step = find_closest_point_on_segment(
            racer.last_pos, prev_centerline_node, next_centerline_node
        )
        progress_reward = metrics.closest_point_on_centerline.distance_to(closest_point_last_step)

        reward = progress_reward * 5.0 - 0.2 + checkpoint_reward
        return reward

    def _check_checkpoint_crossing(self, racer: Racer) -> float:
        reward = 0
        p1 = racer.last_pos
        p2 = racer.car.position

        # Prevent checking for checkpoints that don't exist
        if racer.next_checkpoint >= len(self.track.checkpoints):
            return 0.0

        next_checkpoint_line = self.track.checkpoints[racer.next_checkpoint]
        p3 = next_checkpoint_line[0]
        p4 = next_checkpoint_line[1]

        if get_line_segment_intersection(p1, p2, p3, p4) is not None:
            # If it's the final checkpoint, give a large reward
            if racer.next_checkpoint == len(self.track.checkpoints) - 2:
                racer.done = True  # Mark as finished
                reward = 1000.0
            else:
                reward = 10.0

            racer.next_checkpoint += 1
            racer.time_since_checkpoint = 0
            # assert that the racers position is in the polygon spanned by the current and next checkpoint
            # p5, p6 = self.track.checkpoints[racer.next_checkpoint]
            # assert point_in_polygon_fast(
            #     vector_to_numpy(racer.car.position),
            #     np.array(
            #         [
            #             vector_to_numpy(p6),
            #             vector_to_numpy(p4),
            #             vector_to_numpy(p3),
            #             vector_to_numpy(p5),
            #         ]
            #     ),
            # )

        return reward

    def _get_state(self, racer: Racer, metrics: CarMetrics) -> List[float]:
        lidar_readings, _ = self.track.get_lidar_readings(racer.car, racer.next_checkpoint)

        normalized_forward_speed = metrics.forward_speed / racer.car.max_velocity
        normalized_distance_to_centerline = metrics.distance_to_centerline / self.track.width
        normalized_angle_to_centerline = metrics.angle_to_centerline / math.pi

        return [
            normalized_forward_speed,
            normalized_distance_to_centerline,
            normalized_angle_to_centerline,
        ] + lidar_readings

    def draw(self, screen: pygame.Surface, racers: List[Racer]) -> None:
        self.track.draw(screen)

        for racer in racers:
            # Lerp between red and green based on velocity/max_velocity
            v = max(0.0, min(1.0, racer.car.velocity / racer.car.max_velocity))
            color = (
                int(255 * (1 - v)),  # R: 255->0
                int(255 * (v * 0.5 + 0.5)),  # G: 128->255
                0,  # B: always 0
            )
            if racer.done:
                color = (255, 0, 0)
            racer.car.draw(screen, color)

            # for debugging
            # Draw lidar readings
            # _, lidar_end_points = self.track.get_lidar_readings(racer.car, racer.next_checkpoint)
            # for point in lidar_end_points:
            #     pygame.draw.line(screen, (0, 255, 0), racer.car.position, point, 1)
