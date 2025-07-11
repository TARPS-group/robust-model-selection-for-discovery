#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 18 12:37:48 2022

@author: jwli
"""
import numpy as np


def one_hot(labels):
    u = np.unique(labels)
    K = len(u)
    D = dict(zip(u, list(range(K))))
    A = np.array([[int(D[l] == k)  for k in range(K)] for l in labels])
    return A, u


def F_measure(a, b):
    ''' F measure between two sets of labels
    For each component in a, find the component in b that 
    maximizes the number of overlapping points relative to 
    the number of points from two components
     
    Parameters
    ----------
    a : array-like matrix, shape = n_data
         the ground truth labels for data
    b : array-like matrix, shape = n_data
         the inferred labels from algorithms for data.

    Returns
    -------
    f : float
        F metric between two collections of labels.

    '''
    assert len(a) == len(b)
    ii = a!=0
    aa, bb = a[ii], b[ii]
    A, ua = one_hot(aa)  # one-hot representation
    B, ub = one_hot(bb)
    nA, nB, n = np.sum(A, axis = 0), np.sum(B, axis = 0), np.sum(A)
    C = np.dot(A.T, B)
    R = np.divide(C, nA[:, np.newaxis])
    P = np.divide(C, nB[np.newaxis, :])
    F = np.divide(2* np.multiply(R, P), (R + P + np.finfo(float).eps))
    f = np.sum(np.multiply(nA/n, np.amax(F, axis = 1)))   
    return f