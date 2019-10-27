import datetime as dt


class Timer():

    def __init__(self):
        self.start_dt = None

    def start(self):
        self.start_dt = dt.datetime.now()

    def stop(self):
        end_dt = dt.datetime.now()
        print('Time taken: %s' % (end_dt - self.start_dt))


class DotConverter:

    @staticmethod
    def convert_dot_to_png(file: str) -> None:
        from subprocess import check_call
        check_call(['dot', '-Tpng', 'iris_drzewo.dot', '-o', 'iris_drzewo.png'])