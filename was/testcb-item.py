# -*- coding: utf-8 -*-
"""
Created on Sat Jul 11 23:46:42 2020

@author: pyk12
"""
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity
import numpy as np

R =[
    [1,0,1,0],
    [1,0,1,0],
    [1,0,0,0],
    [0,1,0,0],
    [0,1,0,1]
    ]

select_query=[0,1,0,1]
sel_query = np.array(np.array(select_query)).reshape(1,-1)
r = np.array(R)
print(cosine_similarity(sel_query, r))


# =============================================================================
# q1 = np.array(np.array([0,0,0,1,1])).reshape(1,-1)
# q2 = np.array(np.array([0,0,0,0,1])).reshape(1,-1)
# 
# cosine_sim = cosine_similarity(q1, q2)
# print(cosine_sim)
# 
# =============================================================================


# =============================================================================
# R =[
#     [1,1,1,0,0],
#     [0,0,0,1,1],
#     [1,1,0,0,0],
#     [0,0,0,0,1]]
# 
# select_query=[1,1,1,0,0]
# sel_query = np.array(np.array(select_query)).reshape(1,-1)
# r = np.array(R)
# print(cosine_similarity(sel_query, r))
# =============================================================================
