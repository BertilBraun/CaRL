from game.car import Car
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from game.track import Track


class Racer:
    def __init__(self, track: 'Track') -> None:
        self.track = track

        # Create a new car for the racer
        start_pos = self.track.get_start_position()
        start_angle = self.track.get_start_angle()
        self.car = Car(start_pos[0], start_pos[1], angle=start_angle)

        # Reset racer's progress state
        self.last_pos = self.car.position.copy()
        self.next_checkpoint = 0
        self.lap_count = 0
        self.time_since_checkpoint = 0
        self.total_reward: float = 0.0
        self.done = False
