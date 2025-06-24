import pygame
import math
from .track import Track
from typing import List, Tuple, Dict

from utils.geometry import find_closest_point_on_segment, get_line_segment_intersection

# Guard imports for type-hinting to prevent circular dependencies
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agent.racer import Racer


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

    def step(self, racer: 'Racer', action: int) -> Tuple[List[float], float, bool]:
        # Update car physics
        acceleration_input, steering_input = self.ACTION_MAP[action]
        racer.car.update(acceleration_input, steering_input)

        # --- Reward Calculation ---
        # 1. Calculate reward for progressing towards the current next checkpoint
        next_checkpoint_line = self.track.checkpoints[racer.next_checkpoint]
        checkpoint_center = (next_checkpoint_line[0] + next_checkpoint_line[1]) / 2
        dist_before = racer.last_pos.distance_to(checkpoint_center)
        dist_now = racer.car.position.distance_to(checkpoint_center)
        progress_reward = dist_before - dist_now

        # 2. Check for events and sparse rewards
        collision = self.track.check_collision(racer.car, racer.next_checkpoint)

        checkpoint_reward = self._check_checkpoint_crossing(racer)

        # 3. Calculate speed for reward and state
        velocity_vec = racer.car.position - racer.last_pos
        next_checkpoint_line = self.track.checkpoints[racer.next_checkpoint]
        checkpoint_center = (next_checkpoint_line[0] + next_checkpoint_line[1]) / 2
        to_checkpoint_vec = checkpoint_center - racer.car.position
        if to_checkpoint_vec.length() != 0:
            to_checkpoint_dir = to_checkpoint_vec.normalize()
            forward_speed = velocity_vec.dot(to_checkpoint_dir)
        else:
            forward_speed = 0.0

        # 4. Combine rewards into a final reward value
        # reward = self._calculate_reward(collision, checkpoint_reward, progress_reward, forward_speed)

        # reward is just the distance traveled on the center line since the last step minus some time penalty
        closest_point_last_step = find_closest_point_on_segment(
            racer.last_pos, self.track.nodes[racer.next_checkpoint - 1], self.track.nodes[racer.next_checkpoint]
        )
        closest_point_this_step = find_closest_point_on_segment(
            racer.car.position, self.track.nodes[racer.next_checkpoint - 1], self.track.nodes[racer.next_checkpoint]
        )
        progress_reward = closest_point_this_step.distance_to(closest_point_last_step)
        checkpoint_reward = self._check_checkpoint_crossing(racer)
        reward = progress_reward * 5 - 0.2 + checkpoint_reward if not collision else -500.0

        # --- State and Updates ---
        state = self._get_state(racer)

        # Update racer state for the next step
        racer.last_pos = racer.car.position.copy()

        done = collision

        racer.time_since_checkpoint += 1

        if racer.time_since_checkpoint > self.timeout_to_do_something:
            done = True

        if forward_speed < 0.8:
            racer.time_since_last_movement += 1
        else:
            racer.time_since_last_movement = 0

        if racer.time_since_last_movement > self.timeout_to_do_something:
            done = True
            reward = -500.0

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
            if racer.next_checkpoint == len(self.track.checkpoints) - 2:
                racer.done = True  # Mark as finished
                reward = 1000.0
            else:
                reward = 10.0

            racer.next_checkpoint += 1
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
            racer.time_since_checkpoint = 0

        return reward

    def _calculate_reward(
        self, collision: bool, checkpoint_reward: float, progress_reward: float, forward_speed: float
    ) -> float:
        time_penalty = -1.0
        speed_reward = forward_speed * 0.02
        speed_penalty = -20 if forward_speed < 0.1 else 0
        collision_penalty = -100.0 if collision else 0

        return collision_penalty + checkpoint_reward + progress_reward + time_penalty + speed_reward + speed_penalty

    def _get_state(self, racer: 'Racer') -> List[float]:
        lidar_readings, _ = self.track.get_lidar_readings(racer.car, racer.next_checkpoint)

        velocity_vec = racer.car.position - racer.last_pos
        next_checkpoint_line = self.track.checkpoints[racer.next_checkpoint]
        checkpoint_center = (next_checkpoint_line[0] + next_checkpoint_line[1]) / 2
        to_checkpoint_vec = checkpoint_center - racer.car.position
        if to_checkpoint_vec.length() != 0:
            to_checkpoint_dir = to_checkpoint_vec.normalize()
            forward_speed = velocity_vec.dot(to_checkpoint_dir) / racer.car.max_velocity
        else:
            forward_speed = 0.0

        next_checkpoint = self.track.nodes[racer.next_checkpoint]
        prev_checkpoint = self.track.nodes[max(0, racer.next_checkpoint - 1)]

        # calculate distance to centerline
        closest_point_on_centerline = find_closest_point_on_segment(
            racer.car.position, prev_checkpoint, next_checkpoint
        )
        distance_to_centerline = racer.car.position.distance_to(closest_point_on_centerline) / self.track.width

        # calculate angle to centerline
        angle_to_centerline = (
            math.atan2(
                closest_point_on_centerline.y - racer.car.position.y,
                closest_point_on_centerline.x - racer.car.position.x,
            )
            / math.pi
        )

        return [forward_speed, distance_to_centerline, angle_to_centerline] + lidar_readings

    def draw(self, screen: pygame.Surface, racers: List['Racer']) -> None:
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
