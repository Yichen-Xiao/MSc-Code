# -*- coding: utf-8 -*-

import networkx as nx
import matplotlib.pyplot as plt

# INPUT of the game

# Given number of groups and number batches, generate an appropriate graph
def generate_graph(K,b):
    # Add K,b as labels to the graph to make things simple for other functions
    G = nx.DiGraph(K=K,b=b)
    
    s = (0,0)
    d= (K,b)
    
    # Nodes are indexed (i,j) in the way described by the paper
    # Weights are initialised to 1
    
    # Add source node
    G.add_node(s)
    
    # Add nodes and directed edges from source to first layer of graph
    for j in range(b+1):
        G.add_node((1,j))
        G.add_weighted_edges_from([(s,(1,j),1)])
    
    # Add nodes for each layer and directed edges
    for i in range(2,K):
        for j in range(b+1):
            G.add_node((i,j))
            G.add_weighted_edges_from([((i-1,y),(i,j),1) for y in range(j+1)])
    
    # Add destination node
    G.add_node(d)
    
    # Add directed edges to destination node
    for j in range(b+1):
        G.add_weighted_edges_from([((K-1,j),d,1)])
    
    # Upon further reflection, it would be better if nodes are defined 0,1,2,... 
    # but labels are attached to the nodes
    G_relabeled = G.copy()
    G_relabeled = nx.relabel_nodes(G_relabeled, 
                                   lambda x: 0 if x[0]+x[1]==0 else (1 + (K-1)*(b+1) if x[0]+x[1]==K+b else(
                                       (x[0]-1)*(b+1)+x[1]+1)))
    for n,m in zip(G_relabeled.nodes,G.nodes):
        G_relabeled.nodes[n]['label'] = m
    
    return G_relabeled

# Take a labeled Grotto graph and visualise it
def visualize_graph(G,label=False):
    if label:
        G=G.copy()
        G = nx.relabel_nodes(G, lambda x: G.nodes[x]['label'])
    pos = dict([(n,G.nodes[n]['label']) for n in G.nodes])
    nx.draw(G, pos,with_labels=True)
    plt.axis("off")
    plt.show()