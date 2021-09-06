# -*- coding: utf-8 -*-
from exp_oe import get_next_path, update_weights
import networkx as nx
import numpy as np
from estimator import estimate
from disease_model import seir_model
from initialise_graph import generate_graph
    
# Take a path through the graph and compute a vaccine allocation
# Path needs to be in the order of traversal from s to d
def compute_alloc(G,path):
    alloc = []
    for e in path:
        v_from = e[0]
        v_to = e[1]
        alloc.append(G.nodes[v_to]['label'][1] - G.nodes[v_from]['label'][1])
    return alloc
    
def ppp_wrapper(G,T,eta,d,env):
    # Draw Bernouilli variable
    p = 0.5
    B = np.random.binomial(1,p)
    
    # Initial weight with equal weighting for all paths
    weights_init = {}
    for e in list(G.edges):
        weights_init[e] = 1
        
    # Weight at time t
    weights_cur = weights_init
    
    # Initial path
    path = get_next_path(G,weights_cur)
    
    # Variable to store collected losses
    # Dictionary (indexed by t) of dictionaries (indexed by edges) of observed
    # losses
    losses = {0:{}} # We see nothing on t=0
    
    # Observations from the 'environment'
    observer = env
    
    # Keep track of update and stay rounds
    if B == 1:
        round_history = ['u'] # update round
    else:
        round_history = ['s'] # stay round
    
    for t in range(1,T+1):
        if round_history[t-1] == 'u':  # Draw round
            path = get_next_path(G,weights_cur)
            
            #Ad hoc code to block vaccination of <10 year olds
            # COMMENT THIS OUT WHEN NOT IN USE
            #while compute_alloc(G,path)[0] != 0:
            #    path = get_next_path(G,weights_cur)
            
            round_history.append('r')  
            # Advance time
            vac_alloc = compute_alloc(G,path)
            observer.take_timestep(vac_alloc)
            # Collect loss for the current round
            losses[t] = observer.get_loss(G)
            # Estimate losses on edges in the observation graph
            losses[t] = estimate(G,losses[t],observer.V,observer.N,observer.batch_size)
            #print("Next path: ",path)
            
        elif 'u' in round_history[t-2*d+1:-2]:  # Stay round
            round_history.append('s') # In stay round just use the last path
            # Advance time
            vac_alloc = compute_alloc(G,path)
            observer.take_timestep(vac_alloc)
            # Collect loss for the current round
            losses[t] = observer.get_loss(G)
            # Estimate losses on edges in the observation graph
            losses[t] = estimate(G,losses[t],observer.V,observer.N,observer.batch_size)
            
        else:
            
            # Update round #
            if B == 1: 
                round_history.append('u')
                # Advance time
                vac_alloc = compute_alloc(G,path)
                observer.take_timestep(vac_alloc)
                # Collect loss for the current round
                losses[t] = observer.get_loss(G)
                # Estimate losses on edges in the observation graph
                losses[t] = estimate(G,losses[t],observer.V,observer.N,observer.batch_size)
                #print(losses)
                
                # Find composite loss
                composite_loss = {}
                for t in range(max(t-d+1,0),t+1):
                    for edge in losses[t].keys():
                        composite_loss[edge] = composite_loss.get(edge,0) + losses[t][edge]
                # Turn sum into average
                if t-d+1 >0 :
                    composite_loss = {e : r/(2*d) for e,r in composite_loss.items()}
                else:
                    composite_loss = {e : r/(2*t) for e,r in composite_loss.items()}
                #print("losses of previous path: ", [composite_loss[e] for e in path])
                
                
                # Feed observed loss into SOPPP
                #print(composite_loss)
                weights_cur = update_weights(G,weights_cur,path,composite_loss,beta=0.01,eta=eta)
                
            else:  # Stay round
                round_history.append('s')
                # Advance time
                vac_alloc = compute_alloc(G,path)
                observer.take_timestep(vac_alloc)
                # Collect loss for the current round
                losses[t] = observer.get_loss(G)
                # Estimate losses on edges in the observation graph
                losses[t] = estimate(G,losses[t],observer.V,observer.N,observer.batch_size)
        #print(observer.feedback[t:t+20])
        print(vac_alloc)
        # Draw a new Bernoulli random variable
        B = np.random.binomial(1,p)
    #print(path)
    return(losses,weights_cur,observer)
