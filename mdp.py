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

# Class defining the MDP model and containing the modes necessary to
# find the optimal strategies for AoI or QAoI minimization
class model:
    T_e = 1                     # Error probability function period
    p_e = []                    # Error probability function values
    T_q = 1                     # Maximum inter-query interval
    p_q = []                    # Query probability
    p_b = 0                     # Token generation probability
    B = 0                       # Token bucket size
    M = 0                       # Maximum number of considered queries
    aoi_policy = []             # Policy to minimize AoI
    qaoi_policy = []            # Policy to minimize QAoI
    
    # Constructor
    def __init__(self, p_e, p_q, p_b, B, M):
        self.T_e = len(p_e)
        self.p_e = p_e
        self.T_q = len(p_q)
        self.p_q = p_q
        self.p_b = p_b
        self.B = B
        self.M = M
    
    # Expand the compact state representation into its constituting tuple
    def expand_state(self, s):
        t_q = s % self.T_q
        b = ((s - t_q) / self.T_q) % (self.B + 1)
        d = (s // (self.T_q * (self.B + 1)) + 1) % (self.M * self.T_q)
        if (d == 0):
            d = self.M * self.T_q
        t_e = s // (self.T_q * (self.B + 1) * self.M * self.T_q)
        return t_q, b, d, t_e

    # Serialize a state in tuple form into a compact value
    def serialize(self, t_q, b, d, t_e):
        return int(t_q + self.T_q * b + self.T_q * (self.B + 1) * (d - 1) + self.M * self.T_q * (self.B + 1) * self.T_q * t_e)
    
    # Compute the optimal policies for AoI and QAoI
    def compute(self, gamma):
        [self.aoi_policy, V, P, R] = self.policy_iteration(gamma, 'AoI')
        [self.qaoi_policy, V, P, R] = self.policy_iteration(gamma, 'QAoI')
        
    # Solve the MDP model in the given mode ('AoI' or 'QAoI') and discount using policy iteration
    def policy_iteration(self, gamma, mode):
        # Auxiliary variables
        states = np.arange((self.B + 1) * (self.T_q * self.M) * self.T_q * self.T_e)
        actions = np.array([0, 1])
        S_max = 8
        N_STATES = len(states)
        N_ACTIONS = len(actions)
        P = np.zeros((N_STATES, N_ACTIONS, S_max, 2)) # Transition probability matrix
        R = np.zeros(N_STATES) # Reward vector
        
        # Compute the possible future transitions and their probabilities
        for s in range(N_STATES):
            # Expand current state and find next values
            t_q, b, d, t_e = self.expand_state(s)
            next_e = np.mod(t_e + 1, self.T_e)
            if (d >= self.M * self.T_q - 1):
                d = self.M * self.T_q - 2
            # No transmission, token bucket is not full
            if (b < self.B):
                P[s, 0, 0, :] = self.serialize(0, b, d + 1, next_e), (1 - self.p_b) * self.p_q[t_q]
                P[s, 0, 1, :] = self.serialize(0, b + 1, d + 1, next_e), self.p_b * self.p_q[t_q]          
                P[s, 0, 2, :] = self.serialize(np.min([t_q + 1, self.T_q - 1]), b, d + 1, next_e), (1 - self.p_b) * (1 - self.p_q[t_q])
                P[s, 0, 3, :] = self.serialize(np.min([t_q + 1, self.T_q - 1]), b + 1, d + 1, next_e), self.p_b * (1 - self.p_q[t_q])
            # No transmission, token bucket is full
            else:
                P[s, 0, 0, :] = self.serialize(0, b, d + 1, next_e), self.p_q[t_q]
                P[s, 0, 1, :] = self.serialize(np.min([t_q + 1, self.T_q - 1]), b, d + 1, next_e), (1 - self.p_q[t_q])
            # Transmission
            if (b > 0):
                P[s, 1, 0, :] = self.serialize(0, b - 1, d + 1, next_e), (1 - self.p_b) * self.p_e[t_e] * self.p_q[t_q]
                P[s, 1, 1, :] = self.serialize(0, b, d + 1, next_e), self.p_b * self.p_e[t_e] * self.p_q[t_q]
                P[s, 1, 2, :] = self.serialize(0, b - 1, 1, next_e), (1 - self.p_b) * (1 - self.p_e[t_e]) * self.p_q[t_q]
                P[s, 1, 3, :] = self.serialize(0, b, 1, next_e), self.p_b * (1 - self.p_e[t_e]) * self.p_q[t_q]
                P[s, 1, 4, :] = self.serialize(np.min([t_q + 1, self.T_q - 1]), b - 1, d + 1, next_e), (1 - self.p_b) * self.p_e[t_e] * (1 - self.p_q[t_q])
                P[s, 1, 5, :] = self.serialize(np.min([t_q + 1, self.T_q - 1]), b, d + 1, next_e), self.p_b * self.p_e[t_e] * (1 - self.p_q[t_q])
                P[s, 1, 6, :] = self.serialize(np.min([t_q + 1, self.T_q - 1]), b - 1, 1, next_e), (1 - self.p_b) * (1 - self.p_e[t_e]) * (1 - self.p_q[t_q])
                P[s, 1, 7, :] = self.serialize(np.min([t_q + 1, self.T_q - 1]), b, 1, next_e), self.p_b * (1 - self.p_e[t_e]) * (1 - self.p_q[t_q])
                
        # reward for AoI
        if (mode == 'AoI'):
            for s in range(N_STATES):
                t_q, b, d, t_e = self.expand_state(s)
                R[s] = -d + 1

        # reward for QAoI
        if (mode == 'QAoI'):
            for s in range(N_STATES):
                t_q, b, d, t_e = self.expand_state(s)
                if (t_q == 0):
                    R[s] = -d + 1

        # Initialize policy and value to the minimum
        policy = [0 for s in range(N_STATES)]
        V = - (self.M + 1) * self.T_q * np.ones(N_STATES) / (1 - gamma)

        # Policy iteration mode
        is_value_changed = True
        iterations = 0
        while (is_value_changed):
            is_value_changed = False
            iterations += 1
            V_new = np.zeros(N_STATES)
            # Run value iteration for each state for the given policy (value step)
            for s in range(N_STATES):
                V_new[s] = sum([P[s, policy[s], s1, 1] * (R[int(P[s, policy[s], s1, 0])] + gamma * V[int(P[s, policy[s], s1, 0])]) for s1 in range(S_max)])
            V = V_new
            # Find optimal action for the given value (policy step)
            for s in range(N_STATES):
                q_best = V[s]
                for a in range(N_ACTIONS):
                    t_q, b, d, t_e = self.expand_state(s)
                    if (a == 0 or (b > 0 and self.p_e[t_e] < 1)):
                        q_sa = sum([P[s, a, s1, 1] * (R[int(P[s, a, s1, 0])] + gamma * V[int(P[s, a, s1, 0])]) for s1 in range(S_max)])
                        if (q_sa > q_best + 1e-3):
                            policy[s] = a
                            q_best = q_sa
                            is_value_changed = True
        return policy, V, P, R 
    
    # Get the maximum age
    def get_max_age(self):
        return self.M * self.T_q
    
    # Get the error probability function
    def get_error(self):
        return self.p_e
    
    # Get the query interval PDF
    def get_query_dist(self):
        return self.p_q
    
    # Get the token generation probability
    def get_token_prob(self):
        return self.p_b
    
    # Get the size of the token bucket
    def get_bucket_size(self):
        return self.B
    
    # Get the optimal policy for AoI
    def get_aoi_policy(self):
        return self.aoi_policy
    
    # Get the optimal policy for QAoI
    def get_qaoi_policy(self):
        return self.qaoi_policy
