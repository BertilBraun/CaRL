from typing import Tuple
import pygame
import math
import numpy as np
from utils.geometry import get_corners_numba


class Car:
    def __init__(self, x: float, y: float, angle: float = 0.0) -> None:
        self.position = pygame.math.Vector2(x, y)
        self.velocity = 0.0
        self.angle = angle

        # Default car properties
        self.length = 40
        self.width = 20
        self.max_velocity = 10
        self.acceleration_rate = 0.3
        self.braking_rate = 0.5
        self.friction = 0.05
        self.max_steering_angle = 5

    def update(self, acceleration_input: float = 0.0, steering_input: float = 0.0) -> None:
        """
        Update car state based on inputs.
        :param acceleration_input: float between -1 (brake) and 1 (accelerate)
        :param steering_input: float between -1 (right) and 1 (left)
        """

        # Apply acceleration and braking
        if acceleration_input > 0:
            self.velocity += self.acceleration_rate * acceleration_input
        elif acceleration_input < 0:
            self.velocity += self.braking_rate * acceleration_input  # Note: acceleration_input is negative

        # Apply friction
        if self.velocity > 0:
            self.velocity -= self.friction
        elif self.velocity < 0:  # for when braking makes it negative
            self.velocity = 0

        # Clamp velocity
        self.velocity = max(0, min(self.velocity, self.max_velocity))

        # Apply steering
        if self.velocity > 0.1:  # Can only steer when moving
            steering_effect = steering_input * self.max_steering_angle
            self.angle += steering_effect

        # Update position vector
        self.position.x += self.velocity * math.cos(math.radians(self.angle))
        self.position.y -= self.velocity * math.sin(math.radians(self.angle))

    def get_corners_np(self) -> np.ndarray:
        return get_corners_numba(self.position.x, self.position.y, self.angle, self.length, self.width)

    def draw(self, screen: pygame.Surface, color: Tuple[int, int, int] = (255, 0, 0)) -> None:
        car_surface = pygame.Surface((self.length, self.width), pygame.SRCALPHA)
        car_surface.fill(color)

        # Add a line to indicate front
        pygame.draw.line(car_surface, (0, 0, 0), (self.length / 2, self.width / 2), (self.length, self.width / 2), 3)

        rotated_car = pygame.transform.rotate(car_surface, self.angle)
        rect = rotated_car.get_rect(center=self.position)
        screen.blit(rotated_car, rect)

        # for debugging
        # for p in self.get_corners_np():
        #     pygame.draw.circle(screen, (0, 0, 0), numpy_to_vector(p), 5)
