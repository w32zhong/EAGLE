import time
import json
import statistics
from collections import defaultdict


class TimeStats():
    def __init__(self):
        self.reset()

    def reset(self):
        self._hist = defaultdict(list)
        self._start = defaultdict(float)

    def start(self, key):
        self._start[key] = time.time_ns()

    def stop(self, key, verbose=False):
        dt = time.time_ns() - self._start[key]
        dt_ms = dt / 1_000_000
        self._hist[key].append(dt_ms)
        if verbose: print(key, dt_ms, 'ms')

    def f(self, func_name, hist):
        func = getattr(statistics, func_name)
        if func_name == 'stdev' and len(hist) < 2:
            return float('nan')
        else:
            return func(hist)

    def report(self):
        return json.dumps({
            k: {
                f'cnt': len(self._hist[k]),
                f'mean': self.f('mean', self._hist[k]),
                f'stdev': self.f('stdev', self._hist[k]),
            }
            for k in self._hist.keys()
        }, indent=2)
