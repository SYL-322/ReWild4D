import time


def visual_schedule(iterations, interval=5000, canonical_end=0):
    """Preview at startup, before the canonical phase ends, and before each interval."""
    if interval < 0:
        raise ValueError("visual_interval must be nonnegative")
    if interval == 0:
        return []
    steps = {1, canonical_end - 1}
    steps.update(step - 1 for step in range(interval, iterations + 1, interval))
    return sorted(step for step in steps if 1 <= step <= iterations)


def get_time_prefix():
    # Get the current time in the format [MM_DD_HH:MM:SS]
    start_time = time.gmtime()
    prefix = '[{:0>2}_{:0>2}_{:0>2}:{:0>2}:{:0>2}] '.format(
        str(start_time.tm_mon), str(start_time.tm_mday), 
        str(start_time.tm_hour), str(start_time.tm_min), str(start_time.tm_sec))
    return prefix


class AverageMeter(object):
    """Track the latest value and weighted running average."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
