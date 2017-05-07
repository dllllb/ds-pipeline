import time


class Timer:
    def __init__(self, block_name='block'):
        self.block_name = block_name

    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, *args):
        self.end = time.time()
        self.interval = self.end - self.start
        print('{block} execution time: {sec} sec'.format(block=self.block_name, sec=self.interval))
