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
import montecarlo as mc
import mdp

T_e = 1                             # error probability period                            
T_q = 10                            # inter-query time
M = 5                               # maximum number of packets we can lose
eps = np.linspace(0, 0.8, 5)        # error probabilities
B = 5                               # token bucket size
p_b = 0.2                           # token generation probability
gamma = 0.75                        # future reward discount
packets = 1000000                   # number of steps in Monte Carlo simulation
history = 1000                      # number of steps in AoI history

# Define query PDF
p_q = np.zeros(T_q)
p_q[-1] = 1
p_e = np.ones((len(eps), T_e))


aoi_pq = np.zeros((len(eps), M * T_q))
qaoi_pq = np.zeros((len(eps), M * T_q))
aoi_qapa = np.zeros((len(eps), M * T_q))
qaoi_qapa = np.zeros((len(eps), M * T_q))
av_aoi_pq = np.zeros(len(eps))
av_qaoi_pq = np.zeros(len(eps))
av_aoi_qapa = np.zeros(len(eps))
av_qaoi_qapa = np.zeros(len(eps))

pq_history = np.zeros((len(eps), history))
qapa_history = np.zeros((len(eps), history))
pq_query_history = np.zeros((len(eps), history))
qapa_query_history = np.zeros((len(eps), history))


for g in range(len(eps)):
    print(g)
    p_e[g, 0]=eps[g]
    # Create model and compute strategies
    model = mdp.model(p_e[g, :], p_q, p_b, B, M)
    model.compute(gamma)
    
    # Run Monte Carlo simulations
    aoi_pq[g, :], qaoi_pq[g, :] = mc.histogram_mc(model, packets, 'AoI')
    aoi_qapa[g, :], qaoi_qapa[g, :] = mc.histogram_mc(model, packets, 'QAoI')
    
    aoi_pq[g, :] /= np.sum(aoi_pq[g, :])
    qaoi_pq[g, :] /= np.sum(qaoi_pq[g, :])
    aoi_qapa[g, :] /= np.sum(aoi_qapa[g, :])
    qaoi_qapa[g, :] /= np.sum(qaoi_qapa[g, :])
    
    av_aoi_pq[g] = np.sum(np.asarray(aoi_pq[g, :]) * np.arange(1,  M * T_q + 1))
    av_qaoi_pq[g] = np.sum(np.asarray(qaoi_pq[g, :]) * np.arange(1,  M * T_q + 1))
    av_aoi_qapa[g] = np.sum(np.asarray(aoi_qapa[g, :]) * np.arange(1,  M * T_q + 1))
    av_qaoi_qapa[g] = np.sum(np.asarray(qaoi_qapa[g, :]) * np.arange(1, M * T_q + 1))
    
    pq_history[g, :], pq_query_history[g, :] = mc.montecarlo_history(model, history, 'AoI')
    qapa_history[g, :], qapa_query_history[g, :] = mc.montecarlo_history(model, history, 'QAoI')
    
    np.savez('results', eps, p_q, p_e, p_b, M, B, gamma, aoi_pq, qaoi_pq, aoi_qapa, qaoi_qapa, av_aoi_pq, av_qaoi_pq, av_aoi_qapa, av_qaoi_qapa, pq_history, qapa_history, pq_query_history, qapa_query_history)
