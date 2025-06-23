from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from game.track import Track


class Racer:
    def __init__(self, track: 'Track', progress_on_track: float) -> None:
        from game.car import Car  # Local import to avoid circular dependency

        start_pos, start_angle, start_checkpoint = track.get_point_at_fraction(progress_on_track)

        self.track = track
        self.car = Car(x=start_pos.x, y=start_pos.y, angle=start_angle)
        self.next_checkpoint = start_checkpoint
        self.total_reward = 0.0
        self.done = False
        self.last_pos = self.car.position.copy()
        self.time_since_checkpoint = 0
