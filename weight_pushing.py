# -*- coding: utf-8 -*-

import networkx as nx
import numpy as np
import random
from observations import find_revealing_edges

# Algorithm 3
# Take a directed weighted graph as input
# NB: graph's vertices must be labeled by 0,1,2,..., starting with source vertex and ending with destination vertex
def wp(G,weights_dict):
    K = G.number_of_nodes() - 1
    
    # Dynamic programming
    Hs={0:1}
    Hd={K:1}
    for k in range(1,K+1):
        
        Hd[K-k] = 0
        for v in G.successors(K-k):
            Hd[K-k]+= weights_dict[(K-k,v)] * Hd[v]
        Hs[k] = 0
        for v in G.predecessors(k):
            Hs[k] += weights_dict[(v,k)] * Hs[v]
    return Hs,Hd

# Algorithm 4
# Take a directed weighted graph as input
# NB: graph's vertices must be labeled by 0,1,2,..., starting with source vertex and ending with destination vertex
def wps(G,weights_dict ):
    K = G.number_of_nodes() - 1
    # Use WP Algorithm
    Hs, Hd = wp(G,weights_dict)
        
    # Sample a path from source to destination
    Q = [0]
    cur_vertex = 0
    while cur_vertex != K:
        successors = list(G.successors(cur_vertex))
        probs = []
        for s in successors:
            prob = weights_dict[(cur_vertex,s)] * Hd[s] / Hd[cur_vertex]
            probs.append(prob)
            
        # Sample a vertex using the probabilities defined above
        #print(sum(probs))
        r = random.uniform(0,1)
        i = 0
        while r > 0:
            if i > len(probs) - 1: # For possible edge cases due to rounding errors, since sum(probs) = 1 always in theory
                break
            r -= probs[i]
            i += 1
        next_vertex = successors[i-1]
        Q.append(next_vertex)
        cur_vertex = next_vertex
    
    # Explicitly compute a path
    path = []
    for i,v in enumerate(Q[:-1]):
        #print("HERE")
        path.append((v,Q[i+1]))
    return path

# Algorithm 2
def compute_reveal_prob(G,e,orig_weights):
    K = G.number_of_nodes() - 1
    # Initialise weights
    weights = orig_weights.copy()
    # Initialise q(e)
    q = 0
    # Find R(e)
    R = find_revealing_edges(G,e)
    # Compute H*
    Hsd,_ = wp(G,weights)
    Hsd = Hsd[K]
    for edge in R:
        Hs,Hd = wp(G,weights)
        K = Hs[edge[0]] * orig_weights[edge] * Hd[edge[1]]
        q += K / Hsd
        weights[edge] = 0
    return q