# -*- coding: utf-8 -*-
"""
Created on Tue Oct 13 14:24:25 2020

Copyright (c) 2020 Connectivity Section, Department of Electronic Systems,
Aalborg University, Denmark

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.

@author: Josefine Holm <jho@es.aau.dk> 
         Federico Chiariotti <fchi@es.aau.dk>
         Andreas E. Kalør <aek@es.aau.dk>
"""

import numpy as np

    
# Monte Carlo simulation (histogram) with the given model and number of
# simulated packets
def histogram_mc(model, packets, mode):
    # Model parameters
    max_age = model.get_max_age()
    p_e = model.get_error()
    p_q = model.get_query_dist()
    p_b = model.get_token_prob()
    B = model.get_bucket_size()
    if (mode == 'QAoI'):
        policy = model.get_qaoi_policy()
    else:
        policy = model.get_aoi_policy()
    
    # Auxiliary variables
    aoi_dist = np.zeros(max_age)
    qaoi_dist = np.zeros(max_age)
    age = 1
    tokens = 0
    T_q = len(p_q)
    T_e = len(p_e)
    t_q = 1
    t_e = 0
    
    # Main simulation loop
    for t in range(packets):
        state = model.serialize(t_q, tokens, age, t_e)
        # Update age
        if (policy[state] == 1 and np.random.rand() > p_e[t_e]):
            age = 1
        else:
            age = np.min([max_age, age + 1])
        aoi_dist[age - 1] += 1
        # Update query state
        if (np.random.rand() > p_q[t_q]):
            t_q = np.min([t_q + 1, T_q - 1])
        else:
            t_q = 0
            qaoi_dist[age - 1] += 1
        # Update token bucket state
        if (np.random.rand() <= p_b):
            tokens = np.min([B - 1, tokens + 1 - policy[state]])
        else:
            tokens -= policy[state]
        # Update error index
        t_e = np.mod(t_e + 1, T_e)
        
    return aoi_dist, qaoi_dist

#  Monte Carlo simulation (trace) with the given model and number of
# simulated packets
def montecarlo_history(model, packets, mode):
    # Model parameters
    max_age = model.get_max_age()
    p_e = model.get_error()
    p_q = model.get_query_dist()
    p_b = model.get_token_prob()
    B = model.get_bucket_size()
    if (mode == 'QAoI'):
        policy = model.get_qaoi_policy()
    else:
        policy = model.get_aoi_policy()
    
    # Auxiliary variables
    aoi_history = np.zeros(packets)
    query_history = np.zeros(packets)
    age = 1
    tokens = 0
    T_q = len(p_q)
    T_e = len(p_e)
    t_q = 1
    t_e = 0

    for t in range(packets):
        state = model.serialize(t_q, tokens, age, t_e)
        if (policy[state] == 1 and np.random.rand() > p_e[t_e]):
            age = 1
        else:
            age = np.min([max_age, age + 1])
        # Query
        if (np.random.rand() > p_q[t_q]):
            t_q = np.min([t_q + 1, T_q - 1])
        else:
            t_q = 0
            query_history[t] = 1
        # Update token bucket
        if (np.random.rand() <= p_b):
            tokens = np.min([B, tokens + 1 - policy[state]])
        else:
            tokens -= policy[state]
        # Update error time
        t_e = np.mod(t_e + 1, T_e)
        # Register age
        aoi_history[t] = age
    
    return (aoi_history, query_history)
