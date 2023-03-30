from time import time
from typing import Generator
import pandas as pd

time_data = {}

def show_time_data():
    df = pd.DataFrame(time_data.values(), index=time_data.keys())
    print(df)
    

def TimeFunc(func):
    def wrapper(*args, **kwds):
        if not func.__name__ in time_data:
            time_data[func.__name__] = {'total_time': 0,
                                        'number_of_calls': 0}
        tic = time()
        res = func(*args, **kwds)
        tok = time()
        time_data[func.__name__]['total_time'] += tok - tic
        time_data[func.__name__]['number_of_calls'] += 1
        return res
    
    return wrapper
