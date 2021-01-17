# -*- coding: utf-8 -*-
"""
Created on Tue Feb 04 2020

@author: Kwong Cheong Ng
@filename: error_handling.py
@coding: utf-8
========================
Date          Comment
========================
02042020      First revision 
"""

import sys
import logging

def error_handling():
    return 'Error: {}. {}, line: {}'.format(sys.exc_info()[0],
                                             sys.exc_info()[1],
                                             sys.exc_info()[2].tb_lineno)

if __name__ == '__main__':
    try:
        a + b
    except:
        logging.error(error_handling())