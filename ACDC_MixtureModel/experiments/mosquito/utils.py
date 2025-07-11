#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 18 12:23:35 2022

@author: lijiawei
"""

import time
import os
import errno

def format_seconds(secs):
    if secs < 1e-3:
        t, u = secs * 1e6, 'microsec'
    elif secs < 1e0:
        t, u = secs * 1e3, 'millisec'
    else:
        t, u = secs, 'sec'
    return '%.03f %s' % (t, u)


class Timer:
    def __init__(self, descr=None):
        self.description = descr

    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, *args):
        self.end = time.time()
        self.interval = self.end - self.start
        if self.description is not None:
            time_str = format_seconds(self.interval)
            print('%s took %s to run' % (self.description, time_str))
            
            
def create_folder_if_not_exist(path):
    # create the output folder if it doesn't exist
    try:
        os.makedirs(path)
        print('Created output folder:', path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            print('Unknown error creating output directory', path)
            raise
            
          
def check_if_data_exists(path_to_data):
    return os.path.isfile(path_to_data) and os.access(path_to_data, os.R_OK)