# -*- coding: utf-8 -*-

import networkx as nx
import numpy as np
from weight_pushing import wps, compute_reveal_prob
from observations import observe_path

# Sample a path based on weights
def get_next_path(G, weights={}):
    # Initialise weights if it was not given
    if not weights:
        weights = dict(zip(G.edges, [1] * len(G.edges)))
        
    # Use Algorithm 4
    path = wps(G, weights)
    return path

# A loss vector should be a dictionary with G's edges as keys
# For this application, it is assumed that the loss vector has estimated losses built-in
def update_weights(G,weights,path, loss, beta=0.1, eta = 0.5):
    # Find all the edges revealed by the last chosen oath
    revealed_edges = observe_path(G,path)
    
    estimated_loss = dict(zip(G.edges,np.zeros(len(G.edges))))
    for edge in revealed_edges:
        estimated_loss[edge] = loss[edge] / (compute_reveal_prob(G,edge,weights) + beta)
    for edge in list(set(G.edges) - set(revealed_edges)):
        estimated_loss[edge] = 0
            
    for edge in G.edges:
        weights[edge] = weights[edge] * np.exp( eta * estimated_loss[edge])
    return weights

