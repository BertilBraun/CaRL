import pygame
import math
from .track import Track
from typing import List, Tuple, Dict

from utils.geometry import get_line_segment_intersection

# Guard imports for type-hinting to prevent circular dependencies
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agent.racer import Racer


class GameEnvironment:
    MIN_PROGRESS_THRESHOLD = 0.001
    STALLED_WINDOW = 30
    MAX_STALLED_TIME = 60

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
        done = self.track.check_collision(racer.car, racer.next_checkpoint)

        checkpoint_reward = self._check_checkpoint_crossing(racer)

        # 2a. Check if the racer is stalled or moving backwards
        current_progress = self.track.get_progress_on_track(racer.car.position)
        racer.progress_history.append(current_progress)
        if len(racer.progress_history) > self.STALLED_WINDOW:
            racer.progress_history.pop(0)

        if len(racer.progress_history) == self.STALLED_WINDOW:
            progress_in_window = racer.progress_history[-1] - racer.progress_history[0]
            if progress_in_window < self.MIN_PROGRESS_THRESHOLD:
                racer.time_stalled += 1
            else:
                racer.time_stalled = 0

        if racer.time_stalled > self.MAX_STALLED_TIME:
            done = True
            progress_reward = -200.0

        # 3. Calculate speed for reward and state
        velocity_vec = racer.car.position - racer.last_pos
        angle_rad = math.radians(racer.car.angle)
        forward_vec = pygame.math.Vector2(math.cos(angle_rad), -math.sin(angle_rad))
        forward_speed = velocity_vec.dot(forward_vec)

        # 4. Combine rewards into a final reward value
        reward = self._calculate_reward(done, checkpoint_reward, progress_reward, forward_speed)

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

        if get_line_segment_intersection(p1, p2, p3, p4) is not None:
            # If it's the final checkpoint, give a large reward
            if racer.next_checkpoint == len(self.track.checkpoints) - 1:
                racer.done = True  # Mark as finished
                reward = 1000.0
            else:
                reward = 50.0

            racer.next_checkpoint += 1
            racer.time_since_checkpoint = 0

        return reward

    def _calculate_reward(
        self, done: bool, checkpoint_reward: float, progress_reward: float, forward_speed: float
    ) -> float:
        if done:
            return -100.0

        time_penalty = -0.1
        speed_reward = forward_speed * 0.1
        speed_penalty = -0.5 if forward_speed < 0.1 else 0

        return checkpoint_reward + progress_reward + time_penalty + speed_reward + speed_penalty

    def _get_state(self, racer: 'Racer') -> List[float]:
        lidar_readings, _ = self.track.get_lidar_readings(racer.car)

        velocity_vec = racer.car.position - racer.last_pos
        angle_rad = math.radians(racer.car.angle)
        forward_vec = pygame.math.Vector2(math.cos(angle_rad), -math.sin(angle_rad))

        # Normalize velocity into forward and sideways components relative to car's orientation
        # Project velocity vector onto forward vector to get forward speed
        forward_speed = velocity_vec.dot(forward_vec) / racer.car.max_velocity

        # Use the cross product (in 2D, this is a scalar) to find sideways speed
        sideways_speed = (forward_vec.x * velocity_vec.y - forward_vec.y * velocity_vec.x) / racer.car.max_velocity

        return [forward_speed, sideways_speed] + lidar_readings

    def draw(self, screen: pygame.Surface, racers: List['Racer']) -> None:
        self.track.draw(screen)

        for racer in racers:
            racer.car.draw(screen, (255, 0, 0) if racer.done else (0, 255, 0))

            # Draw lidar readings
            _, lidar_end_points = self.track.get_lidar_readings(racer.car)
            for point in lidar_end_points:
                pygame.draw.line(screen, (0, 255, 0), racer.car.position, point, 1)
