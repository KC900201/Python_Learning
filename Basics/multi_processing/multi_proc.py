# -*- coding: utf-8 -*-
"""
Created on Fri Apr 17 2020

@author: Kwong Cheong Ng
@filename: multi_proc.py
@coding: utf-8
========================
Date          Comment
========================
04172020      First revision
04192020      Example by problem solving
"""

import multiprocessing as mp
import numpy as np
import time 
#from time import time

results = []

# Solution without parallelization
def howmany_within_range(row, minimum=4, maximum=8):
    """Returns how many numbers lie within `maximum` and `minimum` in a given `row`"""
    count = 0
    for n in row:
        if minimum <= n <= maximum:
            count = count + 1
    return count

# Step 1: Redefine, to accept `i`, the iteration number
def howmany_within_range2(i, row, minimum, maximum):
    """Returns how many numbers lie within `maximum` and `minimum` in a given `row`"""
    count = 0
    for n in row:
        if minimum <= n <= maximum:
            count = count + 1
    return (i, count)

# Step 2: Define callback function to collect the output in `results`
def collect_result(result):
    global results
    results.append(result)

if __name__ == '__main__':
    print("Number of processors: ", mp.cpu_count())
    
    # Prepare data
    np.random.RandomState(100)
    arr = np.random.randint(0, 10, size=[200000, 5])
    data = arr.tolist()
    data[:5]

    start_time = time.process_time()
    for row in data: # 2.8619329 seconds
        results.append(howmany_within_range(row, minimum=4, maximum=8))

    print(results[:10])
    print("--- %s seconds ---" % (time.process_time() - start_time))
    
    # Parallelizing using Pool.apply()
    # Creating a pool of multiprocesses
    pool = mp.Pool(mp.cpu_count())
    # Step 1: Init multiprocessing.Pool()
    start_time_2 = time.time()
    print("Step 1")

    # Step 2: `pool.apply` the `howmany_within_range()`
    print("Step 2")
    #results = [pool.apply(howmany_within_range, args=(row, 4, 8)) for row in data] # 71.96873044967651 seconds
    #results = pool.map(howmany_within_range, [row for row in data]) # 0.3759949207305908 seconds
    #results = pool.starmap(howmany_within_range, [(row, 4, 8) for row in data]) # 0.3759949207305908 seconds
    
    #for i, row in enumerate(data):
    #    results = pool.apply_async(howmany_within_range2, args=(i, row, 4, 8), callback=collect_result)
    # Apply async without callback
    #result_objects = [pool.apply_async(howmany_within_range2, args=(i, row, 4, 8)) for i, row in enumerate(data)] # 42.54204702377319 seconds
    results = pool.starmap_async(howmany_within_range2, [(i, row, 4, 8) for i, row in enumerate(data)]).get() # 0.7360320091247559 seconds

    # Step 3: Don't forget to close
    print("Step 3")
    pool.close()
    pool.join()

#    results.sort(key = lambda x: x[0])
#    results_final = [r for i, r in results]    

#    print(results_final[:10])
    print(results[:10])
    print("--- %s seconds ---" % (time.time() - start_time_2))