import pygame
import math


class Car:
    def __init__(self, x: float, y: float, angle: float = 0.0) -> None:
        self.position = pygame.math.Vector2(x, y)
        self.velocity = 0.0
        self.angle = angle

        # Default car properties
        self.length = 40
        self.width = 20
        self.max_velocity = 5
        self.acceleration_rate = 0.1
        self.braking_rate = 0.2
        self.friction = 0.05
        self.max_steering_angle = 3

    def update(self, acceleration_input: int = 0, steering_input: int = 0) -> None:
        """
        Update car state based on inputs.
        :param acceleration_input: 1 for accelerate, -1 for brake, 0 for coast.
        :param steering_input: positive for left, negative for right.
        """

        # Apply acceleration and braking
        if acceleration_input == 1:
            self.velocity += self.acceleration_rate
        elif acceleration_input == -1:
            self.velocity -= self.braking_rate

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

        print(self.position, self.angle, self.velocity, acceleration_input, steering_input)

    def draw(self, screen: pygame.Surface) -> None:
        car_surface = pygame.Surface((self.length, self.width), pygame.SRCALPHA)
        car_surface.fill((255, 0, 0))  # Red car

        # Add a line to indicate front
        pygame.draw.line(car_surface, (0, 0, 0), (self.length / 2, self.width / 2), (self.length, self.width / 2), 3)

        rotated_car = pygame.transform.rotate(car_surface, self.angle)
        rect = rotated_car.get_rect(center=self.position)
        screen.blit(rotated_car, rect)
