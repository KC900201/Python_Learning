# -*- coding: utf-8 -*-
"""
Created on Fri Apr 21 2020

@author: Kwong Cheong Ng
@filename: multi_proc_ml.py
@coding: utf-8
@url: https://www.machinelearningplus.com/python/parallel-processing-python/
========================
Date          Comment
========================
04212020      First revision
"""

#from pathos.multiprocessing import ProcessingPool as Pool
import multiprocessing as mp
import time
import numpy as np
import pandas as pd

#result = []

# Row wise Operation
def hypotenuse(row):
    return round(row[1]**2 + row[2]**2, 2)**0.5

# Column wise Operation
def sum_of_squares(column):
    return sum([i**2 for i in column[1]])

def func(df):
    return df.shape

if __name__ == '__main__':
    df = pd.DataFrame(np.random.randint(3, 10, size=[500, 2]))
#    print(df.head())
    
    pool = mp.Pool(mp.cpu_count())
    cores = mp.cpu_count()
    df_split = np.array_split(df, cores, axis=0)
#    start_time = time.clock()
    start_time = time.process_time()
    '''
    with mp.Pool(4) as pool:
        result = pool.imap(hypotenuse, df.itertuples(), chunksize=10)
        output = [round(x, 2) for x in result]
    
    with mp.Pool(2) as pool:    
        result = pool.imap(sum_of_squares, df.iteritems(), chunksize=10) #  0.6361142000000002
        output = [x for x in result]
    '''
    # process the DataFrame by mapping function to each df across the pool
    df_out = np.vstack(pool.map(func, df_split)) # 1.1893037
    # close down the pool and join
    pool.close()
    pool.join()
#    pool.clear()
    print(df_out)
#    print(output)
    print("--- %s seconds ---" % (time.process_time() - start_time))