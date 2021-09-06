# -*- coding: utf-8 -*-
import networkx as nx

# Function for turning labeled edges to unlabeled ones
# Takes a tuple of integers as input
def unlabel(G,e):
    K=G.graph['K']
    b=G.graph['b']
    f =lambda x: 0 if x[0]+x[1]==0 else (1 + (K-1)*(b+1) if x[0]+x[1]==K+b else(
                                       (x[0]-1)*(b+1)+x[1]+1))
    return (f(e[0]),f(e[1]))

# Given a Blotto type graph G, and an edge e in G, find all the edges revealed
# by observing e
# Each vertex must come with a label (i,j)
def observe_edges(G,e):
    K=G.graph['K']
    b=G.graph['b']
    # Convert e into labeled format
    e = (G.nodes[e[0]]['label'],G.nodes[e[1]]['label'])
    # Relabel graph for convenience
    G=G.copy()
    G = nx.relabel_nodes(G, lambda x: G.nodes[x]['label'])
    
    v_from = e[0]
    v_to = e[1]
    n = v_to[1] - v_from[1]  # Number of doses allocated to a group
    i = v_from[0]
    edges = []
    # Find all side-observations
    if i == 0:
        for k in range(v_to[1],b+1):
            edges.append(((0,0),(1,k)))
    elif i == K-1:
        for k in range(v_from[1],-1,-1):
            edges.append(((K-1,k),(K,b)))
    else:
        for x in range(b-n+1):
            for y in range(x+n,b+1):
                edges.append(((i,x),(i+1,y)))
            
    return list(map(lambda x: unlabel(G,x),edges))

# Computes \mathbb{R}(r)
def find_revealing_edges(G,e):
    K=G.graph['K']
    b=G.graph['b']
    # Convert e into labeled format
    e = (G.nodes[e[0]]['label'],G.nodes[e[1]]['label'])
    # Relabel graph for convenience
    G=G.copy()
    G = nx.relabel_nodes(G, lambda x: G.nodes[x]['label'])
    v_from = e[0]
    v_to = e[1]
    n = v_to[1] - v_from[1]  # Number of doses allocated to a group
    i = v_from[0]
    
    R = []
    
    if i == 0:
        for k in range(v_to[1]+1):
            R.append(((0,0),(1,k))) # All the edges 'below' e
    elif i == K-1:
        for k in range(v_from[1],b+1): # All the edges 'above' e
            R.append(((i,k),(K,b)))
    else: # Here we need all the edges corresponding to allocating less than n doses
        for d in range(n+1):
            for x in range(b-d+1):
                R.append(((i,x),(i+1,x+d)))
    
    return list(map(lambda x: unlabel(G,x),R))
    
# Given a Blotto type graph G, and a path in G, find O(p)
def observe_path(G,p):
    edges = []
    for e in p:
        edges += observe_edges(G,e)
    return edges
    