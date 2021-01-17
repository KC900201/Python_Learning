# -*- coding: utf-8 -*-
"""
Created on Mon Jan 27 2020

@author: Kwong Cheong Ng
@filename: MultiProcessing.py
@coding: utf-8
@URL: https://pythonprogramming.net/multiprocessing-python-intermediate-python-tutorial/
========================
Date          Comment
========================
01272020      First revision
04282020      Further training
"""
import time
import multiprocessing
from multiprocessing import Pool #if multiprocessing returns value

def spawn():
    while True:        
        print('Spawned')

def spawn2(num): #with args
    print('Spawn # {}'.format(num))

def spawn3(num, num2):
    print('Spawn # {} {}'.format(num, num2))

def job(num):
    return num ** 2

if __name__ == '__main__':   
#    for i in range(5):
#        p = multiprocessing.Process(target=spawn3, args=(i, i+1))
#        p.start()
#        p.join()

    # Multiprocessing with Pooling
    p = Pool(processes=20) # set up Pool object, with n processes
    data = p.map(job, [i for i in range(20)]) # map job function to a list of parameters []
    p.close() # close the Pool object when job finishes, returning result
    print(data)
    time.sleep(10)
