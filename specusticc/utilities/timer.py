import datetime as dt


class Timer:
    def __init__(self):
        self.start_dt = None
        self.end_dt = None

    def start(self):
        self.start_dt = dt.datetime.now()

    def stop(self):
        self.end_dt = dt.datetime.now()

    def print_time(self):
        time_taken = self.end_dt - self.start_dt
        print(f"[Timer] Time taken: {time_taken}")
