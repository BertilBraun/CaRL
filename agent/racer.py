from __future__ import annotations

import random
from typing import TYPE_CHECKING, List

if TYPE_CHECKING:
    from game.track import Track


class Racer:
    def __init__(
        self, track: Track, progress_on_track: float, initial_velocity: float, initial_angle_variance: float
    ) -> None:
        from game.car import Car  # Local import to avoid circular dependency

        start_pos, start_angle, start_checkpoint = track.get_point_at_fraction(progress_on_track)

        start_angle += random.uniform(-initial_angle_variance, initial_angle_variance)

        self.track = track
        self.car = Car(x=start_pos.x, y=start_pos.y, angle=start_angle, velocity=initial_velocity)
        self.next_checkpoint = start_checkpoint
        self.total_reward = 0.0
        self.done = False
        self.last_pos = self.car.position.copy()
        self.current_state: List[float] = []

        # Checkpoint detection
        self.time_since_checkpoint = 0
        # Slow movement detection
        self.time_since_last_movement = 0
