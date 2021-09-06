# -*- coding: utf-8 -*-

from wrapper import ppp_wrapper
import numpy as np
import networkx as nx
from observations import unlabel
from initialise_graph import generate_graph
from disease_model import seir_model
import matplotlib.pyplot as plt

def do_nothing():
    model = seir_model()
    for _ in range(300):
        model.take_timestep()
    print(model.infection_history)
    print(sum(model.S.values()))
    #print(model.hospital_history)
    #print(model.death_history)
    plt.plot(model.infection_history.values())
    
def main_analysis():
    model = seir_model()
    N = model.N_total
    # Calibrate vaccine batch size
    batch_size = 2000
    model.batch_size = batch_size
    #model.batch_size = 0
    
    
    start_time = 0
    # vaccination begins when 10% of the population is exposed or worse
    while sum(model.S.values())/model.N_total > 0.9:
        start_time += 1
        model.take_timestep()
        
    G = generate_graph(8,16)
    
    #loss,weights,fin_model = ppp_wrapper(G,100,1.2,6,model)
    loss,weights,fin_model = ppp_wrapper(G,100,0.01,6,model)
    
    infection_history = np.array(list(fin_model.infection_history.values()))/N
    death_history = np.array(list(fin_model.death_history.values()))/N
    
    #input()
    # Benchmark
    bench_model = benchmark_analysis(batch_size)
    b_infection_history = np.array(list(bench_model.infection_history.values()))/N
    b_death_history = np.array(list(bench_model.death_history.values()))/N
    
    plt.plot(b_infection_history,'k--',label = 'Infected (Benchmark)')
    
    # Using Algorithm
    plt.plot(infection_history,'k-',label = 'Infected (Algorithm)')
    plt.xlabel("Days")
    plt.ylabel("Proportion of original population")
    plt.legend()
    plt.title('Simulation Results Using Hospitalisation as Feedback')
    plt.show()
    plt.clf()
    #plt.axvline(x=start_time, label = 'Vaccination Start Date')
    plt.plot(death_history,'r-',label = 'Deceased (Algorithm)')
    plt.plot(b_death_history,'r--',label = 'Deceased (Benchmark)')
    plt.xlabel("Day")
    plt.ylabel("Proportion of original population")
    plt.legend()
    plt.title('Simulation Results Using Incidence Feedback - Deaths')
    plt.show()
    
    
    
    print(sum(infection_history))
    print(sum(b_infection_history))
    print(fin_model.V)
    #972366.1151563375
    return infection_history
    
def benchmark_analysis(batch_size):
    model = seir_model()
    
    # Calibrate vaccine batch size
    model.batch_size = batch_size
    
    start_time = 0
    # vaccination begins when 10% of the population is infected
    while sum(model.S.values())/model.N_total > 0.9:
        start_time += 1
        model.take_timestep()
    
    counter = 7
    
    for _ in range(100):
        print(model.S)
        if model.S[counter]<= 0 and counter > 0:
            counter -= 1
        alloc = [0,0,0,0,0,0,0,0]
        alloc[counter] = 16
        model.take_timestep(alloc)
    
    return model

def batch_analysis():
    for b in [0,2000,4000,6000,8000]:
        model = seir_model()
        N = model.N_total
        model.batch_size = b
        for _ in range(70):
            model.take_timestep()
        G = generate_graph(8,16)
        loss,weights,fin_model = ppp_wrapper(G,80,1.2,10,model)
        infection_history = np.array(list(fin_model.infection_history.values())[70:])/N
        lab ='Batch size: ' + str(b)
        plt.plot(infection_history,label=lab)
        plt.legend()
        plt.ylabel("Proportion of original population")
        plt.xlabel("Days since vaccination allocation")
        plt.title('Comparison of Batch Sizes')
        
def budget_analysis():
    start_time = 0
    model = seir_model(sliding_budget = True, budget_gradient = 250, budget_max = 4000)
    model.batch_size = 500
    N = model.N_total
    # vaccination begins when 15% of the population is infected
    while sum(model.S.values())/model.N_total > 0.9:
        start_time += 1
        model.take_timestep()
    G = generate_graph(8,16)
    
    #loss,weights,fin_model = ppp_wrapper(G,100,1.2,6,model)
    loss,weights,fin_model = ppp_wrapper(G,60,0.1,6,model)
    
    infection_history = np.array(list(fin_model.infection_history.values()))/N
    infection_history = infection_history[start_time:]
    
    for b in [4000,4500,5500,6000,6500,7000]:
        model = seir_model(sliding_budget = True, budget_gradient = (b-500)/14, budget_max = b)
        model.batch_size = 500
        while sum(model.S.values())/model.N_total > 0.9:
            model.take_timestep()
        counter = 7
        for _ in range(60):
            print(model.S)
            if model.S[counter]<= 0 and counter > 0:
                counter -= 1
            alloc = [0,0,0,0,0,0,0,0]
            alloc[counter] = 16
            model.take_timestep(alloc)
        b_infection_history = np.array(list(model.infection_history.values()))/N
        b_infection_history = b_infection_history[start_time:]
        lab = 'Benchmark Max Daily Budget: ' + str(b*16)
        plt.plot(b_infection_history,label = lab)

    plt.plot(infection_history,'k-',label = 'Algorithm Max Daily Budget: 64000')
    plt.xlabel("Days since vaccination start")
    plt.ylabel("Proportion of original population")
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), fancybox=True, shadow=True)
    plt.title('Budget Analysis')
    plt.show()