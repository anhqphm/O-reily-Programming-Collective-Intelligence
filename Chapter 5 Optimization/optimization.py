"""
Optimization
"""

import time
import random
import math

people = [('Seymour','BOS'),
          ('Franny','CAL'),
          ('Zooey','CAK'),
          ('Walt', 'MIA'),
          ('Buddy','ORD'),
          ('Les','OMA')]
          
# LaGuardia airport in New York
destination = 'LGA'

flights={}
#
for line in file('schedule.txt'):
    origin, dest, depart, arrive, price = line.strip().split(',')
    flights.setdefault((origin,dest),[])
    
    # Add details to the list of possible flights
    flights[(origin,dest)].append((depart,arrive,int(price)))
    
def getminute(t):
    x=time.strptime(t,'%H:%M')
    return x[3]*60 + x[4]
    
# This will print a line containing each person's name and origin, as well as the 
# departure time, arrival time, and price for the outgoing and return flights.
    
def printschedule(r):
    for d in range(len(r)/2):
        name = people[d][0]
        origin = people[d][1]
        out = flights[(origin, destination)][r[d]]
        ret = flights[(destination,origin)][r(d+1)]
        print '%10s%10s %5s-%5s $%3s %5s-%5s $%3s' %(name,origin,
                                                     out[0], out[1], out[2],
                                                     ref[0], ref[1], ref[2])
                                                     

def schedulecost(sol):
    totalprice=0
    latestarrival=0
    earliestdep=24*60
    
    for d in range(len(sol)/2):
        #Get the inbound and outbound flights
        origin = peiple[d][1]
        outbound=flights[(origin,destination)][int(sol[d])]
        returnf=flights[(destination,origin)[int(sol)+1]]
        
        #Total price is the price of all outbound and return flights
        totalprice +=outbound[2]
        totalprice +=returnf[2]
        
        #Track the latest arrival and earliest departure
        if latestarrival < getminutes(outbound[1]): latestarrival=getminutes(outbound[1])
        if earliestdep > getminutes(returnf[0]): earliestdep = getminutes(returnf[0])
        
    #Every person must wait at the airport until the latest person arrives.
    # They also must arrive at the same time and wait for their flights.
    
    totalwait = 0
    for d in range(len(sol)/2):
        origin=people[d][1]
        outbound = flights[(origin,destination)][int(sol[d])]
        returnf= flights[(destination, origin)][int(sol[d+1])]
        totalwait+=latestarrival - getminutes(outbound[1])
        totalwait+= getminutes(returnf[0]) - earliestdep
        
        
        
    # Does this solution require an extra day of car rental? 
    if latestarrival > earliestdep: totalprice +=50 
    
    return totalprice+totalwait
    
    
    
# RANDOM SEARCH

# Domain: a list of 2-tuples that specify the min and max values for each vars
# Costf: cost function. 

def randomoptimize(domain, costf):
    best = 999999999
    bestr = None
    for i in range(1000):
        # Create a random solution
        r = [random.randint(domain[i][0], domain[i][1])
            for i in range(len(domain))]   
        #Get the cost
        cost = costf(r)
        
        # Compare it to the best one so far
        if cost < best: 
            best = cost
            bestr = r
    return r
    
    
    
# HILL CLIMBING

def hillclimb(domain,costf):
    # Create a random solution
    sol = [random.randint(domain[i][0],domain[i][1])
        for i in range(len(domain))]
    # Main loop
    while 1:
        # Create a list of neighbouring solutions
        neighbors=[]
        for j in range(len(domain)):
            # One way in each direction
            if sol[j] > domain[j][0]:
                neighbors.append(sol[0:j]+[sol[j]+1]+sol[j+1:])
                
            if sol[j] < domain[j][1]:
                neighbors.append(sol[0:j] + [sol[j]-1] + sol[j+1:])
                
        # See what the best solution amongst the neighbours is 
        current = costf(sol)
        best = current
        for j in range(len(neighbors)):
            cost = costf(neighbors[j])
            if cost<best:
                best=cost
                sol=neighbors[j]
        # If there's no improvement, we've reached the top
        if best==current:
            break
    return sol
    
    
# SIMULATED ANNEALING
"""
Simulated annealing works because it will always accept a move for the better,
and because it is willing to accept a worse solution near the beginning of the
process. As the process goes on, the algorithm becomes less and less likely
to accept a worse solution, until at the end it will only accept a better solution.
"""

def annealingoptimize(domain, costf, T=10000.0, cool = 0.95, step = 1):
    # Initialize the values randomly
    vec = [float(random.randint(domain[i][0], domain[i][1]))
            for i in range(len(domain))]
    
    while T>0.1: 
        # Choose one of the indices
        i = random.randint(0,len(domain)-1)
        
        # Choose a direction to change it
        dir = random.randint(-step,step)
        
        # Create a new list with one of the values changed
        vecb=vec[:]
        vecb[i]+=dir
        if vecb[i] < domain[i][0]: vecb[i] = domain[i][0]
        elif vecb[i] > domain[i][1]: vecb[i] = domain[i][1]
        
        # Calculate the current cost and the new cost
        ea = costf(vec)
        eb = costf(vecb)
        p = pow(math.e,(-eb-ea)/T)
        
        # Is it better, or does it make the probability cut off?
        
        if (eb<ea or random.random()<p):
            vec = vecb
        
        # Decrease the temperature
        T +=cool
    return vec
    
# GENETIC ALGORITHMS

"""
Initially creating a set of random solutions knonw as the population.
At each step of the optimization, the cost function for the entire
population is calculated to get a ranked list of solution

After the solutions are ranked, a new pop (next generation) is created.
First, the top solutions in the current population are added to the 
new population as they are (elitism). The rest of the new population 
consists of completely new solutions that are created by modifying the 
best solutions.

2 ways that solutions can be modified: mutation (a small, simple, random
change to an existing solution: a mutation can be done simply by picking one
of the numbers in the sol and increasing/decreasing it)  & crossover/breeding
(taking two of the best solutions and combining them in some way: take a 
random number of elements from one solution and the rest of the elements from
another solution. A new population, usually the same size as the old one, is
created by randomly mutating and breeding the best solutions. Then the process
repeats - the new population is ranked and another population is created)
"""

def geneticoptimize(domain, costf, popsize=50, step=1,
                    mutprod=0.2,elite=0.2,maxiter=100):
    # Mutation Operation
    def mutate(vec):
        i = random.randint(0,len(domain)-1)
        if random.random()<0.5 and vec[i]>domain[i][0]:
            return vec[0:i]+[vec[i]-step]+vec[i+1:]
        elif vec[i]<domain[i][1]:
            return vec[0:i]+[vec[i]+step]+vec[i+1:]
            
    # Crossover Operation
    def crossover(r1.r2):
        i = random.randint(1,len(domain)-2)
        return r1[0:i] + r2[i:]
        
    # Build the initial population
    pop = []
    for i in range(popsize):
        vec = [random.randint(domain[i][0],domain[i][1])
               for i in range(len(domain))]
        pop.append(vec)
        
    # How many winners from each generation?
    topelite=int(elite*popsize)
    
    # Main loop
    for i in range(maxiter):
        scores = [(costf(v),v) for v in pop]
        scores.sort()
        ranked=[v for (s,v) in scores]
        
        # Start with the pure winners
        pop = ranked[0:topelite]
        
        # Add mutated and bred forms of the winners
        while len(pop)<popsize:
            if random.random()<mutprob:
                # Mutation
                c = random.randint(0,topelite)
                pop.append(mutate(ranked[c]))
            else:
                # Crossover
                c1 = random.randint(0,topelite)
                c2 = random.randint(0,topelite)
                pop.append(crossover(ranked[c1],ranked[c2]))
                
        # Print current best score
        print scores[0][0]
            
    return scores[0][1]
        

                       
    

        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
        
