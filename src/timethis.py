import time

def timethis(func, number=1000):
    def wrapper(*args, **kwargs):
        t1 = time.time()
        res = func(*args, **kwargs)
        t2 = time.time()
        print(f'{func.__name__} took {(t2 - t1)*1000:.3f} ms')
        return res
    return wrapper