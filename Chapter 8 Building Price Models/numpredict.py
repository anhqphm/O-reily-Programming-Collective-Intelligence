"""

 In this section, we will create a dataset of wine prices based on a simple artificial model
 The prices are based on a combination of the rating and the age of the wine.
 The mode; assumes that wine has a peak age. which is older for good wines and almost immediate for bad wines.
 A high-rated wine will start at a high price and increase in value until its peak age, and a low-rated wine
 will start cheap and get cheaper"""


from random import random, randint
import math
from pylab import *

def wineprice(rating, age):
    peak_age = rating - 50

    # Calculate price based on rating
    price = rating/2
    if age>peak_age:
        # Past its peak, goes bad in 5 years
        price = price*(5 - (age - peak_age))
    else:
        # Incrases to 5x original values as it approaces its peak
        price = price*(5*((age + 1)/peak_age))
    if price < 0:
        price = 0
    return price


def wineset1():
    rows = []
    for i in range(300):
        # Create a random age and rating
        rating = random()*50 + 50
        age = random()*50

        # Get referemce price
        price = wineprice(rating,age)

        # Add some noise
        price*=(random()*0.4 + 0.8)

        # Add to the dataset
        rows.append({'input':(rating,age),
                     'result':price})
    return rows


# KNN
def euclidean(v1,v2):
    d = 0.0
    for i in range(len(v1)):
        d += (v1[i] - v2[i])**2
    return math.sqrt(d)

def getdistances(data, vec1):
    distancelist = []
    for i in range (len(data)):
        vec2 = data[i]['input']
        distancelist.append((euclidean(vec1,vec2),i))
    distancelist.sort()
    return distancelist

def knnestimate(data, vec1, k =3):
    # Get sorted distances
    dlist = getdistances(data,vec1)
    avg = 0.0

    # Take the advantage of the top k results
    for i in range(k):
        idx = dlist[i][1]
        avg +=data[idx]['result']
    avg = avg/k
    return avg
    
# Weighted Neighbours

# 1. Using inverse function

def inverseweight(dist, num = 1.0, const = 0.1):
    return num/(dist + const)
    
# 2. Subtraction Function
def subtractweight(dist, const = 1.0):
    if dist > const:
        return 0
    else:
        return const - dist
        
# 3. Gaussian Function
def gaussian(dist, sigma=10.0):
    return math.e**(-dist**2/(2*sigma**2))
    
# Weighted kNN

def weightedknn(data,vec1,k=5,weighf=gaussian):
    # Get distances
    dlist = getdistances(data,vec1)
    avg = 0.0
    totalweight = 0.0
    
    # Get weighted average
    for i in range(k):
        dist=dlist[i][0]
        idx = dlist[i][1]
        weight = weighf(dist)
        avg +=weight*data[idx]['result']
        totalweight+=weight
    avg = avg/totalweight
    return avg
    
    
# Cross-Validation
# a set of techniques that divide up data into training sets and test sets. 
# The training set is given to the algorithm, along with the correct answers 
# (in this case, prices), and becomes the set used to make predictions. The
# algorithm is then asked to make predictions for each item in the test set. 
# The answers it give are compared to the correct answers, and an overall 
# score for how well the algorithm did is calculated.

# Usually this procedure is performed several times, dividing the data up
# differently each time. 

def dividedata(data,test=0.05):
    trainset=[]
    testset=[]
    for row in data:
        if random()<test:
            testset.append(row)
        else:
            trainset.append(row)
    return trainset,testset
    
def testalgorithm(algf, trainset, testset):
    error= 0.0
    for row in testset:
        guess = algf(trainset, row['input'])
        error+=(row['result']-guess)**2
    return error/len(testset)
    
def crossvalidate(algf, data, trials=100, test=0.05):
    error=0.0
    for i in range(trials):
        trainset,testset=dividedata(data,test)
        error+=testalgorithm(algf, trainset,testset)
    return error/trials
    
# Heterogeneous Variables

def wineset2():
    rows=[]
    for i in range(300):
        rating=random()*50 + 50
        age = random()*50
        aisle = float(randint(1,20))
        bottlesize=[375.0,750.0,1500.0,3000.0][randint(0,3)]
        price = wineprice(rating,age)
        price*=(bottlesize/750)
        price*=(random()*0.9 + 0.2)
        rows.append({'input':(rating,age,aisle,bottlesize),
                     'result':price})
    return rows
        

    
def rescale(data,scale):
    scaleddata=[]
    for row in data:
        scaled=[scale[i]*row['input'][i] for i in range(len(scale))]
        scaleddata.append({'input':scaled, 'result':row['result']})
    return scaleddata
    
    
# Optimising the Scale
# Recall that optimization simply requires you to specify a domain 
# that gives the number of variables, a range and a cost function. 

def createcostfunction(algf, data):
    def costf(scale):
        sdata=rescale(data,scale)
        return crossvalidate(algf, sdata, trials = 10)
    return costf
  
weightdomain = [(0,20)]*4


# An advantage of optimizing the variable scales: you immediately
# see which variables are important and how important they are. 
  
  
# Uneven Distributions
# some discount information was not reflected in the dataset
    
def wineset3():
    rows=wineset1()
    for row in rows:
        if random() < 0.5:
            # Wine was bought at a discount store
            row['result']*=0.6
    return rows
    


# Estimating the Probability Density

# Rather than taking the weighted average of the neighbors and getting
# a single price estimate, it might be interesting in this case to 
# know the probability that an item falls within a certain price range.

# This function first calculates the weights of the neighbors within 
# that range, and then calculates the weights of all the neighbors. 


def probguess(data, vec1, low, high, k = 5, weightf = gaussian):
    dlist=getdistances(data, vec1)
    nweight = 0.0
    tweight = 0.0
    
    for i in range(k):
        dist = dlist[i][0]
        idx = dlist[i][1]
        weight = weightf(dist)
        v = data[idx]['result']
        
        # Is this point in the range?
        if v >= low and v <= high:
            nweight += weight
        tweight+=weight
    if tweight==0:
        return 0
    
    # The probability is the weights in the range
    # divided by all the weights
    return nweight/tweight
    
    
# Graphing the Probabilities 
    
def cumulativegraph(data,vec1,high,k=5,weightf=gaussian):
    t1 = arange(0.0,high,0.1)
    cprob = array([probguess(data,vec1,0,v,k,weightf) for v in t1])
    plot(t1,cprob)
    show()
    
def probabilitygraph(data,vec1,high,k=5, weightf=gaussian, ss=5.0):
    # Make a range for the prices
    t1 = arange(0.0, high, 0.1)
    
    # Get the probabilities for the entire range
    probs = [probguess(data,vec1,v,v+0.1,k,weightf) for v in t1]
    
    # Smooth them by adding the gaussian of the nearby probabilities
    smoothed=[]
    for i in range(0,len(probs)):
        sv=0.0
        for j in range(0,len(probs)):
            dist=abs(i-j)*0.1
            weight=gaussian(dist,sigma=ss)
            sv+=weight*probs[j]
        smoothed.append(sv)
    smoothed= array(smoothed)
    
    plot(t1,smoothed)
    show()
    
    
        
           




