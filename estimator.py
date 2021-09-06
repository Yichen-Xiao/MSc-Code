# -*- coding: utf-8 -*-
import networkx as nx
from observations import unlabel

# Loss is a dictionary with a list of edges as keys and the loss embedded on that edge
def estimate(G,loss,V,N,batch_size):
    K=G.graph['K']
    b=G.graph['b']
    
    # relabel the path
    path = loss.keys() # Relying on proper insertion order for dictionary
    #print([loss[e] for e in path])
    path = [(G.nodes[e[0]]['label'],G.nodes[e[1]]['label']) for e in path]
    for i,edge in enumerate(path):
        v_from = edge[0]
        v_to = edge[1]
        
        # find number of doses corresponding to the edge
        doses = v_to[1] - v_from[1]
        n = doses
        
        # pretend the current round didn't happen
        #V = V.copy()
        #V[i] = V[i] - doses
        
        # Assuming loss is inversely proportional to proportion of people vaccinated in group
        
        # Find constant of proportionality
        p = loss[unlabel(G,edge)] * (V[i]*batch_size/N[i])
        
        # Same structure as observe edges code
        if i == 0:
            for k in range(v_to[1],b+1):
                loss[unlabel(G,((0,0),(1,k)))] = p * (V[i] + (k-doses)*batch_size) / N[i]
        elif i == K-1:
            for k in range(v_from[1],-1,-1):
                loss[unlabel(G,((K-1,k),(K,b)))] = p * (V[i] + (K-k-doses)*batch_size) / N[i]
        else:
            for x in range(b-n+1):
                for y in range(x+n,b+1):
                    loss[unlabel(G,((i,x),(i+1,y)))] = p * (V[i] + (y-x-doses)*batch_size) / N[i]
    #print(loss)
    #input()
    return loss
            