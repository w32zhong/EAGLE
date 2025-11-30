import json
import time
import torch
import math
import statistics
from collections import defaultdict


class TimeStats():
    def __init__(self, disable=False):
        self.reset()
        self.disable = disable

    def reset(self):
        self._hist = defaultdict(list)
        self._start = defaultdict(float)

    @staticmethod
    def is_capturing():
        return torch.cuda.is_current_stream_capturing()

    def start(self, key='time'):
        if self.disable or self.is_capturing(): return
        torch.cuda.synchronize()
        self._start[key] = time.perf_counter_ns()

    def stop(self, key='time', verbose=False):
        if self.disable or self.is_capturing(): return
        torch.cuda.synchronize()

        dt = time.perf_counter_ns() - self._start[key]
        dt_ms = dt / 1_000_000
        self._hist[key].append(dt_ms)
        if verbose: print(key, dt_ms, 'ms')

    def f(self, func_name, hist, ignore_nan=False):
        if hasattr(statistics, func_name):
            func = getattr(statistics, func_name)
        else:
            func = eval(func_name)

        if ignore_nan:
            hist = list(filter(lambda x: not math.isnan(x), hist))

        if func_name == 'stdev' and len(hist) < 2:
            return float('nan')
        else:
            return func(hist)

    def report(self, lst=None, ignore_nan=False):
        use = lambda k: (lst is None or k in lst)
        return json.dumps({
            k: {
                f'cnt': self.f('len', self._hist[k], ignore_nan=ignore_nan),
                f'sum': self.f('sum',  self._hist[k], ignore_nan=ignore_nan),
                f'max': self.f('max',  self._hist[k], ignore_nan=ignore_nan),
                f'min': self.f('min',  self._hist[k], ignore_nan=ignore_nan),
                f'mean': self.f('mean', self._hist[k], ignore_nan=ignore_nan),
                f'stdev': self.f('stdev', self._hist[k], ignore_nan=ignore_nan),
            }
            for k in self._hist.keys() if use(k)
        }, indent=2)
