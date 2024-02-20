import time
import os

class Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]

class TimingInfo(metaclass=Singleton):
    def __init__(self):
        self.info = dict()
        self.t = time.time()
        self.period = 3

    def update_statistics(self, func_name: str, delta_time: float) -> None:
        if func_name in self.info.keys():
            # min
            if delta_time < self.info[func_name]["min"]:
                self.info[func_name]["min"] = delta_time
            # max
            if delta_time > self.info[func_name]["max"]:
                self.info[func_name]["max"] = delta_time
            # mean
            n = self.info[func_name]["num_executions"]
            old_mean = self.info[func_name]["mean"]
            self.info[func_name]["mean"] = (old_mean * n + delta_time) / (n + 1)
            # num executions
            self.info[func_name]["num_executions"] += 1
        else:
            self.info[func_name] = {
                "min": delta_time,
                "mean": delta_time,
                "max": delta_time,
                "num_executions": 1,
            }

    def print_statistics(self) -> None:
        os.system("cls" if os.name == "nt" else "clear")
        print(f"In module {__name__}:")
        for func_name, func_info in self.info.items():
            print(f"{func_name}")
            print(f"\tmin={func_info['min']*1000:.3f} ms")
            print(f"\tmax={func_info['max']*1000:.3f} ms")
            print(f"\tmean={func_info['mean']*1000:.3f} ms")
            print(f"\tnum_executions={func_info['num_executions']}")
        print("-" * 40 + "\n")

    def periodic_print(self):
        now = time.time()
        if now - self.t >= self.period:
            self.print_statistics()
            self.t = now


def timethis(func, timing_info=TimingInfo()):
    def wrapper(*args, **kwargs):
        t1 = time.time()
        res = func(*args, **kwargs)
        t2 = time.time()
        timing_info.update_statistics(func.__name__, t2 - t1)
        timing_info.periodic_print()
        return res

    return wrapper
