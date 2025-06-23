import pygame
import math
import numpy as np
from .track import Track
from utils.geometry import get_line_segment_intersection


class GameEnvironment:
    def __init__(self, track_nodes=None):
        self.track = Track(nodes=track_nodes)
        self.action_map = {
            0: (0, 0),  # coast
            1: (1, 0),  # accelerate
            2: (-1, 0),  # brake
            3: (0, 1),  # steer left
            4: (0, -1),  # steer right
        }

    def step(self, racer, action):
        # Update car physics
        acceleration_input, steering_input = self.action_map[action]
        racer.car.update(acceleration_input, steering_input)

        # Check for events
        done = self._check_collision(racer.car)
        checkpoint_reward = self._check_checkpoint_crossing(racer)

        # Calculate reward and get new state
        reward = self._calculate_reward(done, checkpoint_reward)
        state = self._get_state(racer.car)

        # Update racer state
        racer.last_pos = racer.car.position.copy()
        racer.time_since_checkpoint += 1

        if racer.time_since_checkpoint > 300:
            done = True

        return state, reward, done

    def _check_checkpoint_crossing(self, racer):
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

    def _check_collision(self, car):
        car_corners = self._get_car_corners(car)
        for point in car_corners:
            if not self._point_in_polygon(point, self.track.outer_points) or self._point_in_polygon(
                point, self.track.inner_points
            ):
                return True
        return False

    def _point_in_polygon(self, point, polygon):
        x, y = point.x, point.y
        n = len(polygon)
        inside = False
        p1x, p1y = polygon[0].x, polygon[0].y
        for i in range(n + 1):
            p2x, p2y = polygon[i % n].x, polygon[i % n].y
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        xinters = 0.0
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x

                        if p1x == p2x or x <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y
        return inside

    def _get_car_corners(self, car):
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

    def _get_lidar_readings(self, car):
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

    def _get_track_lines(self):
        lines = []
        for i in range(len(self.track.outer_points)):
            lines.append((self.track.outer_points[i - 1], self.track.outer_points[i]))
        for i in range(len(self.track.inner_points)):
            lines.append((self.track.inner_points[i - 1], self.track.inner_points[i]))
        return lines

    def _calculate_reward(self, done, checkpoint_reward):
        if done:
            return -100

        time_penalty = -0.1

        return checkpoint_reward + time_penalty

    def _get_state(self, car):
        lidar_readings, _ = self._get_lidar_readings(car)
        return [car.velocity / car.max_velocity] + lidar_readings

    def draw(self, screen, racers):
        self.track.draw(screen)
        self.track.draw_checkpoints(screen)
        for racer in racers:
            racer.car.draw(screen)
            self._draw_lidar(screen, racer.car)

    def _draw_lidar(self, screen, car):
        _, lidar_end_points = self._get_lidar_readings(car)
        for point in lidar_end_points:
            pygame.draw.line(screen, (0, 255, 0), car.position, point, 1)
