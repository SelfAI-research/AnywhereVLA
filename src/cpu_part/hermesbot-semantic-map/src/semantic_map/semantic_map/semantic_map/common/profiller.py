import time

class Profiler:
    def __init__(self, log_every: int = 10, window_sec: float = 5.0, logger=None):
        self.log_every = log_every
        self.window_sec = window_sec
        self.logger = logger

        self._prof = {}
        self._count = 0
        self._last_t = None

    def t(self):
        return time.perf_counter_ns()

    def add(self, name: str, dt_sec: float):
        self._prof.setdefault(name, []).append(float(dt_sec * 1e-9))

    def add_period(self, t_now_ns: int):
        if self._last_t is not None:
            self.add("period", (t_now_ns - self._last_t) * 1e-9)
        self._last_t = t_now_ns

    def maybe_log(self):
        self._count += 1
        if self.log_every > 0 and (self._count % self.log_every) == 0:
            self._log_means()

    def _log_means(self):
        if not self._prof:
            return
        parts = []
        periods = self._prof.get("period", [])
        n = 0; sum_period = 0.0
        if periods:
            for v in reversed(periods):
                sum_period += v; n += 1
                if sum_period >= self.window_sec:
                    break
            if sum_period > 0.0:
                parts.append(f"rate={n/sum_period:.1f}Hz")
        for k, values in self._prof.items():
            tail = values[-n:] if n > 0 else values
            if tail:
                parts.append(f"{k}={(sum(tail)/len(tail))*1e3:.2f}ms")
        if self.logger:
            self.logger.info("Timing (last {:.1f}s):\n\t{}".format(self.window_sec, ", ".join(parts)))
        else:
            print("Profiler:", ", ".join(parts))
