################################################################################
# DelayDiscounting.py
# Author: John Piaszynski
# Last updated: 12/30/21
################################################################################
#
################################################################################

import csv
import random
import math
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys

################################################################################
#
################################################################################

class DelayDiscounting:
    """
    Class that takes data from the delay discounting test and computes indifference points,
    hyperbolic and exponential discounting curves, and classifies the subject as either a
    hyperbolic or exponential discounter.
    """

    # populates data fields and runs subsequent functions to produce results
    # param: subID: subject ID
    # param: times: list of delay lengths
    # param: amounts: list of immediate reward amounts
    # param: defaultAmount: the amount of money received at the delay period
    # param: record: optionally include choices for an already-administered delay discounting test
    #  format should be a .csv with delay length as the first row and delay amount as the first column
    # param: indiffs: optionally include a list of indifference points from a delay discounting test
    #  should be a dictionary keyed by delay lengths and their associated indiff points
    #  NOTE: if a record is provided, it will override the given indiff points
    # return: None
    def __init__(self, subID = "test",
                 times = [0,7,30,90,180,365],
                 amounts = [0,2.5,5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95,100,105],
                 defaultAmount = 100, record=None, indiffs=None):
        self.subject_ID = subID
        self.defaultAmount = defaultAmount
        if not record:
            self.times = times
            self.amounts = amounts
        else:
            self.times = []
            self.amounts = []
        self.responses = {} # will contain pair:response mapping to the individual test prompts
        self.askedResponses = False
        self.indiffs = {} # indifference points; the point at which the subject has no preference for delay or immediate
        self.hyperbolic_k = 0
        self.hyperbolic_RSS = 0
        self.exponential_k = 0
        self.exponential_RSS = 0
        self.actual_AUC = 0
        self.hyperbolic_AUC = 0
        self.exponential_AUC = 0
        self.discounting_type = None

        # if a record or set of indiff points is given, uses those,
        # if not it prompts the user to take the test
        if record and not indiffs:
            self.readData(record)
            self.findIndiffs()
        elif indiffs:
            self.indiffs = indiffs
        else:
            self.getResponses()
            self.findIndiffs()
        self.findKvalues()
        self.computeAUCs()

    # reads in data from an already-taken delay discounting test in .csv format
    #  delay lengths are in the first row and immediate reward amounts are in first column
    # param: data: file location of test data
    # return: None: sets self.responses, self.times, and self.amounts fields
    def readData(self, data):
        responses = {}
        with open(data, 'r') as file:
            reader = csv.reader(file)
            info = [row for row in reader]
            for time in info[0][1:]:
                self.times.append(int(time))
            for amount in [info[i][0] for i in range(1,len(info))]:
                self.amounts.append(float(amount))
            for amount in range(1,len(info)):
                for time in range(1,len(info[0])):
                    responses[(int(info[0][time]),float(info[amount][0]))] = int(info[amount][time])
        self.responses = responses

    # administers delay discounting test to a subject through the command prompt
    # return: None: updates self.responses field with subject choices
    def getResponses(self):
        self.askedResponses = True
        pairs = []
        # creates (delay length, immediate amount) pairs
        for i in self.times:
            for j in self.amounts:
                pairs.append((i,j))
        random.shuffle(pairs)
        responses = {}
        print("Enter '0' to choose the immediate reward, and '1' to choose the delay reward")
        for pair in pairs:
            if pair[0] == 0 and pair[1] == 100:
                continue
            else:
                statement = "Would you rather get ${} now, or $100 in {} days?\n".format(pair[1],pair[0])
                response = input(statement)
                while response not in ("0","1"):
                    print("Select 0 or 1\n")
                    response = input(statement)
                responses[pair] = int(response)
        self.responses = responses

    # computes indifference points for a set of subject data
    # return: None: updates self.indiffs field
    def findIndiffs(self):
        responsesForEachTime = {} # maps delay lengths to subject response at every amount
        indiffs = {}
        for level in self.times:
            responsesForEachTime[level] = [None for i in range(len(self.amounts))]
            indiffs[level] = None
        indexToAmount = {} # maps list index to the immediate reward amount it represents
        amountToIndex = {} # maps immediate reward amount to its list index
        index=0
        # needs higher dollar values at lower list indexes for the indexing to work properly
        if self.askedResponses:
            self.amounts.reverse()
        for level in self.amounts:
            amountToIndex[level] = index
            indexToAmount[index] = level
            index += 1
        # recovers proper order of immediate amounts
        for record in self.responses.keys():
            responsesForEachTime[record[0]][amountToIndex[record[1]]] = self.responses[record]

        # finds indifference points by taking the average of the lowest amount preferred for
        # the immediate reward and the highest amount where the delay was preferred
        for time in indiffs.keys():
            responses = responsesForEachTime[time] # computes for each delay length
            high0 = 0 # relative to the internal list object
            low1 = len(responses)-1
            # assumes the highest dollar value gets a response of 1 at every level
            for i in range(len(responses)-2,0,-1):
                if responses[i] == 1:
                    # if a response is bracketed by two opposing responses, it is rejected as erroneous
                    if responses[i-1] != 0 or responses[i+1] != 0:
                        low1 = i
            # assumes the lowest dollar value gets a response of 0 at every level
            for i in range(1,len(responses)-1):
                if responses[i] == 0:
                    if responses[i-1] != 1 or responses[i+1] != 1:
                        high0 = i
            if responses[len(responses)-1] == 0 and responses[len(responses)-2] == 0:
                high0 = len(responses)-1
            indiffs[time] = ((self.amounts[high0] + self.amounts[low1]) / 2)
        self.indiffs = indiffs

    # computes k-values (rate of discounting) and RSS for both hyperbolic and exponential curves
    # return: None: sets self.hyperbolic_K, self.hyperbolic_RSS, self.exponential_K, self.exponential_RSS
    def findKvalues(self):
        
        # helper functions

        # estimates indiff point based on the exponential curve for a given rate of discounting and time frame
        # exponetial curve has the form:
        #  f(D) = A * e^(-k*D)
        #  where D = delay length, A = delay reward, k = rate of discounting
        # param: k: rate of discounting (calculated for an exponential curve)
        # param: x: length of delay
        # return: float representing the indifference point at delay length "x" based on the curve given by the above form
        def ex(k,x):
            return (self.defaultAmount)*math.exp(-k*x)

        # estimates indiff point based on the hyperbolic curve for a given k and delay length
        # hyperbolic curve has the form:
        #  f(D) = A / (1 + k*D)
        #  where A = delay reward, D = delay length, k = rate of discounting
        # param: k: rate of discounting (estimated for a hyperbolic curve)
        # param: x: delay length
        # return: float of indifference point for given k and delay length
        def hyp(k,x):
            return (self.defaultAmount)/(1+k*x)
        # pair of functions that compute the Residual Sum of Squares for the exponential and hyperbolic curves, respectively
        # param: k: rate of discounting (for respective curve)
        # param: data: actual indifference points of subject test
        # return: float representing RSS of estimated curve from actual data
        def squareDifExp(k,data):
            return (ex(k,0)-data[0])**2 + (ex(k,7)-data[7])**2 + (ex(k,30)-data[30])**2  + (ex(k,90)-data[90])**2 + (ex(k,180)-data[180])**2 + (ex(k,365)-data[365])**2
        def squareDifHyp(k,data):
            return (hyp(k,0)-data[0])**2 + (hyp(k,7)-data[7])**2 + (hyp(k,30)-data[30])**2  + (hyp(k,90)-data[90])**2 + (hyp(k,180)-data[180])**2 + (hyp(k,365)-data[365])**2

        # estimates rate of discounting based on a hyperbolic curve using a quasi-brute force algorithm,
        # finds the best value of k at a high level, then takes ten even spaced values centered on that k
        #  and repeats for several offsets, using RSS as the difference parameter
        #  estimating k is complicated by the fact that the RSS is not symmetric about a given estimate of k
        # param: indiffs: set of indifference points
        # param: lo: lower bound to start checking values
        # param: hi: upper bound to start checking values
        # return: tuple of estimated value of k and its associated RSS as floats
        def findHyperbolicK(indiffs, lo=0,hi=1):
            k = 0
            dif = 9999999 # crude, but the difference should never be this high
            # initially finds the best value of k between 0 and 1 by increments of 0.1
            for i in np.arange(float(lo), float(hi), 0.1):
                current_difference = squareDifHyp(i,indiffs)
                # takes the value of k that gives the lowest RSS
                if current_difference < dif:
                    k = i
                    dif = current_difference
            # offsets to incrementally check around the previous best value of k
            offsets = [0.1,0.01,0.001,0.0001,0.00001,0.000001,0.0000001,0.00000001,0.000000001,0.0000000001,0.00000000001,0.000000000001,0.0000000000001,0.00000000000001,0.000000000000001,0.0000000000000001,0.00000000000000001]
            for offset in range(len(offsets)-1):
                # checks ten evenly-spaced values around the previous best k estimate
                for i in np.arange(k-offsets[offset], k+offsets[offset], offsets[offset+1]):
                    current_difference = squareDifHyp(i,indiffs)
                    if current_difference < dif:
                        k = i
                        dif = current_difference
            return (k,dif)

        # estimates rate of discounting and RSS based on an exponential curve
        # uses same algorithm as findHyperbolicK()
        # param: indiffs: actual indifference points
        # param: lo: lower bound to start searching from
        # param: hi: upper bound to start searching to
        # return: tuple of estimated k and its RSS as floats
        def findExponentialK(indiffs, lo=0,hi=1):
            k = 0
            dif = 9999999 # crude, but the difference should never be this high
            for i in np.arange(float(lo), float(hi), 0.1):
                current_difference = squareDifExp(i,indiffs)
                if current_difference < dif:
                    k = i
                    dif = current_difference
            offsets = [0.1,0.01,0.001,0.0001,0.00001,0.000001,0.0000001,0.00000001,0.000000001,0.0000000001,0.00000000001,0.000000000001,0.0000000000001,0.00000000000001,0.000000000000001,0.0000000000000001,0.00000000000000001]
            for offset in range(len(offsets)-1):
                for i in np.arange(k-offsets[offset], k+offsets[offset], offsets[offset+1]):
                    current_difference = squareDifExp(i,indiffs)
                    if current_difference < dif:
                        k = i
                        dif = current_difference
            return (k,dif)

        # runs above helper functions and populates fields with the results
        hyperbolic = findHyperbolicK(self.indiffs)
        exponential = findExponentialK(self.indiffs)
        self.hyperbolic_K = hyperbolic[0]
        self.hyperbolic_RSS = hyperbolic[1]
        self.exponential_K = exponential[0]
        self.exponential_RSS = exponential[1]


    # calculates trapezoidal Area Under the Curve for both estimated curves
    # return: None: sets self.hyperbolic_AUC and self.exponential_AUC, and categorizes discounter type
    def computeAUCs(self):
        actual = 0
        hyperbolic = 0
        exponential = 0
        for dx in range(len(self.times)-1):
            # AUC pieces use the equation: (x2 - x1) * (y1 + y2)/2
            actual += ((self.times[dx+1] - self.times[dx])*((self.indiffs[self.times[dx]]+self.indiffs[self.times[dx+1]])/2))
            hyperbolic += ((self.times[dx+1] - self.times[dx])*((self.recoverIndiff("hyperbolic",self.times[dx])+self.recoverIndiff("hyperbolic",self.times[dx+1]))/2))
            exponential += ((self.times[dx+1] - self.times[dx])*((self.recoverIndiff("exponential",self.times[dx])+self.recoverIndiff("exponential",self.times[dx+1]))/2))

        # sets values
        self.actual_AUC = actual
        self.hyperbolic_AUC = hyperbolic
        self.exponential_AUC = exponential

        # categorizes discounting type based on which estimated curve's AUC is closer to the actual
        # uses squared AUC difference to remove the effect of sign when subtracting
        if (actual - hyperbolic)**2 < (actual - exponential)**2:
            self.discounting_type = "hyperbolic"
        else:
            self.discounting_type = "exponential"

    # helper function that gives the location of either curve at a given delay length
    # left as a public function in case it becomes useful for plotting purposes
    # param: curve: which curve to calculate, either "hyperbolic" or "exponential"
    # param: time: delay length (x-axis value)
    # return: float indicating amount of indifference point (y-axis value)
    def recoverIndiff(self, curve, time):
        if curve == "hyperbolic":
            return (self.defaultAmount)/(1+self.hyperbolic_K*time)
        elif curve == "exponential":
            return (self.defaultAmount)*math.exp(-1*self.exponential_K*time)

    # gives printout of subject results
    # return: None: prints information to command line
    def printSummary(self):
        print("Subject ID: {}".format(self.subject_ID))
        print("Indiff points: {}".format(self.indiffs))
        print("At delay amount ${}:".format(self.defaultAmount))
        print("Actual AUC: {}".format(self.actual_AUC))
        print("--------------------")
        print("Hyperbolic Discounting:")
        print("K-value: {}".format(self.hyperbolic_K))
        print("RSS: {}".format(self.hyperbolic_RSS))
        print("AUC: {}".format(self.hyperbolic_AUC))
        print("--------------------\nExponential Discounting")
        print("K-value: {}".format(self.exponential_K))
        print("RSS: {}".format(self.exponential_RSS))
        print("AUC: {}".format(self.exponential_AUC))
        print("--------------------\nDiscounting Type: {}".format(self.discounting_type))

    def plotCurve(self, curve, filename):
        plt.figure()
        plt.xlabel("Delay Length (days)")
        plt.ylabel("Immediate Reward Amount ($)")
        actualY = [self.indiffs[i] for i in self.times]
        if curve == 'exponential':
            data = [self.recoverIndiff('exponential',i) for i in list(range(366))]
            plt.title("Exponential Discounting Curve")
        else:
            data = [self.recoverIndiff('hyperbolic',i) for i in list(range(366))]
            plt.title("Hyperbolic Discounting Curve")
        X = list(range(366))
        plt.plot(X,data,color="blue",label='{} curve'.format(curve))
        plt.plot(self.times,actualY,color="green",label='actual')
        plt.legend()
        plt.xlim(0,365)
        plt.ylim(0,105)
        plt.savefig(filename)
        
        

                        





def main():
    data = DelayDiscounting("test",record="./test.csv")
    data.printSummary()
    data.plotCurve("exponential","./exponentialPlot_{}.png".format(data.subject_ID))
    data.plotCurve("hyperbolic","./hyperbolicPlot_{}.png".format(data.subject_ID))

if __name__ == '__main__':
    main()
            
        

