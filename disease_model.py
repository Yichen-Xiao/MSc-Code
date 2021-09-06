# -*- coding: utf-8 -*-
import numpy as np
import networkx as nx
from observations import unlabel

class seir_model:
    
    def __init__(self,loss_type='hospitalised',sliding_budget = False, budget_gradient = 0,budget_max = 0):
        
        self.loss_type = loss_type # incidence, hospitalisation, death
        self.sliding_budget = sliding_budget # Controls if the batch size is fixed
        self.budget_gradient = budget_gradient # if the batch size isn't fixed, then this controls how much the budget increases
        self.budget_max = budget_max
        
        # parameters of the model
        self.contact_matrix = np.loadtxt(open("contact_matrix.csv"),delimiter=",",skiprows=1,usecols=(1,2,3,4,5,6,7,8))
        self.sus = np.array([0.4,0.38,0.79,0.86,0.8,0.82,0.88,0.74])/5
        self.age_dist = [0.116397452,0.12922296,0.127145681,0.147143385,0.143472335,0.12477855,0.096813849,0.115025787]
        self.N_total = 3000000
        self.d = 1/5.1
        self.gamma = 0.2
        
        # Chance of hospitalisation by age
        self.hospital_chance = [0,0.001,0.01,0.034,0.043,0.082,0.118,0.17] 
        
        # Chance of death by age 
        self.death_chance = np.array([0,0.2/4,0.21,0.3/4,1.18/4,3.2/4,10.8/4,11.9375])/100
        
        age_dist = self.age_dist # spaghetti code
        
        self.current_time = 0
        
        # compartments
        # each compartment (S_i,I_i,etc) is a dictionary with i as keys
        self.S = {}
        self.E = {}
        self.I = {}
        self.R = {}
        self.N = {}
        self.V = {} # Keep track of number of people vaccinated
        
        # Metrics
        self.death_history = {}
        self.infection_history = {}
        
        # Initialise compartments
        for i, d in enumerate(age_dist):
            self.S[i] = np.round(self.N_total * d)
            self.E[i] = 0
            self.I[i] = 0
            self.R[i] = 0
            self.V[i] = 0
            
            self.N[i] = self.S[i] # assume changes in total pop is negligible
        
        # Seed 10 people from 20 to 50
        for _ in range(10):
            r = np.random.randint(2,4)
            self.S[r] -= 1
            self.E[r] += 1
        
        # Keep track of feedback
        # List of lists (with integer keys)
        self.feedback = [[0 for i in range(len(self.age_dist))] for j in range(2*365)] # This is basically a calendar
        
        # History of each compartment
        self.history = []
        
        # Current vaccination plan
        self.vac_plan = [[0] for _ in range(len(self.age_dist))]
        self.batch_size = 300
        
    def merge_comparts(self):
        return (sum(self.S.values()),sum(self.E.values()),sum(self.I.values()),sum(self.R.values()))
        
    def take_timestep(self,vaccine_dist=[0,0,0,0,0,0,0,0]):
        # Do budget calculations
        if self.sliding_budget == True and self.batch_size<self.budget_max and vaccine_dist != [0,0,0,0,0,0,0,0]:
            self.batch_size += self.budget_gradient
        #print(self.batch_size)
        
        # Write to history
        self.history.append((self.S.copy(),self.E.copy(),self.I.copy(),self.R.copy()))
        if len(vaccine_dist) == 8:
            self.vac_plan = vaccine_dist
        else:
            self.vac_plan = [0] * (8-len(vaccine_dist)) + vaccine_dist
            vaccine_dist = self.vac_plan
        
        # Do the update
        self.current_time += 1
        
        new_infections = [] # Keep track of newly infected people for feedback purposes
        hospitalised = []
        deaths = {}
        for i in range(len(self.age_dist)):
            dS = 0
            dE = 0
            for j in range(len(self.age_dist)):
                dS+= self.contact_matrix[i][j] * self.I[j] / self.N[j]
                dE+= self.contact_matrix[i][j] * self.I[j] / self.N[j]
            dS = dS * -1 * self.sus[i] * self.S[i]
            dE = dE * self.sus[i] * self.S[i] - self.d * self.E[i]
            dI = self.d * self.E[i] - self.gamma * self.I[i]
            dR = self.gamma * self.I[i]
            
            if self.S[i]+dS > vaccine_dist[i] * self.batch_size:
                dS -= vaccine_dist[i] * self.batch_size
                dR += vaccine_dist[i] * self.batch_size
            else:
                dS = -self.S[i]
                dR = self.S[i]
            
            new_infections.append(self.d * self.E[i])
            self.S[i] = self.S[i]+dS
            self.E[i] = self.E[i]+dE
            self.I[i] = self.I[i]+dI
            self.R[i] =  self.R[i]+dR
            
            # keep track of total num of vaccinations
            self.V[i] += vaccine_dist[i]
            
            hospitalised.append(self.hospital_chance[i] * new_infections[-1])
            
            # keep track of deaths 
            deaths[i] = max(0,(self.gamma * self.I[i]) * self.death_chance[i]) # some proportion of natural recoveries are deaths
            
        # record metrics
        self.death_history[self.current_time] = sum(deaths.values())
        self.infection_history[self.current_time] = sum(self.I.values())
        
        print(self.merge_comparts())
        # here is a more complicated feedback mechanism
        '''
        # Add feedback to the schedule
        if self.loss_type == 'incidence':
            for i in range(len(self.age_dist)):
                newly_infected = new_infections[i]
                for _ in range(int(np.round(newly_infected/2))): # Half of people with symptoms get tested in 5 days
                    r = np.random.randint(6)
                    self.feedback[self.current_time+r][i] += 1
        elif self.loss_type == 'hospitalised':
            for i in range(len(self.age_dist)):
                newly_hospitalised = hospitalised[i]
                for _ in range(int(np.round(newly_hospitalised))):
                    r = np.random.randint(6)
                    self.feedback[self.current_time+r][i] += 1
        elif self.loss_type == 'deaths':
            self.feedback[self.current_time][i] = np.round(deaths[i])
        '''
        
        for i in range(len(self.age_dist)):
            if self.loss_type == 'incidence':
                newly_infected = new_infections[i]
                self.feedback[self.current_time+1][i] += newly_infected
            elif self.loss_type == 'hospitalised':
                newly_hospitalised = hospitalised[i]
                self.feedback[self.current_time+1][i] += newly_hospitalised
            elif self.loss_type == 'deaths':
                self.feedback[self.current_time+1][i] = deaths[i]
                    
                
    def get_feedback(self):
        return self.feedback[self.current_time]
    
    # Use feedback and current vac plan to embed observed loss on current path
    def get_loss(self,G):
        K=G.graph['K']
        b=G.graph['b']
        
        feedback = self.get_feedback()
        
        # embed loss of current path
        if sum(self.vac_plan) != 0:
            losses = {}
            total_vac = 0
            for i,v in enumerate(self.vac_plan):
                if i == 0:
                    start = (i,i)
                else:
                    start = (i,total_vac)
                    
                if  i == len(self.vac_plan)-1:
                    end = (K,b)
                else:
                    end = (i+1,v+total_vac)
                    total_vac+=v
                
                losses[(start,end)] = feedback[i] 
            
            # rename dictionary keys to be compatible with graph
            losses = {unlabel(G,k):v for (k,v) in losses.items()}
            
            return losses
        else:
            return None
        
        
