from game.car import Car


class Racer:
    def __init__(self, car_config, track):
        self.track = track
        self.car_config = car_config
        self.reset()

    def reset(self):
        # Create a new car for the racer
        start_pos = self.track.get_start_position()
        start_angle = self.track.get_start_angle()
        self.car = Car(start_pos[0], start_pos[1], angle=start_angle, **self.car_config)

        # Reset racer's progress state
        self.last_pos = self.car.position.copy()
        self.next_checkpoint = 0
        self.lap_count = 0
        self.time_since_checkpoint = 0
        self.total_reward = 0
        self.done = False
