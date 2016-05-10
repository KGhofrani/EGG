# Step 1, find the make a grid of different variables -- done
# Step 2, calculate the distance between the data points and the points in the grid done
# Step 3, assign a weight to each of the data points using a gaussian distribution done
# Step 4, calculate a weighted average between all the data points for each grid points done
# Step 5, grid interpolate between the data points
# Step 6, Likelihood function generated

import numpy as np
from scipy.stats import norm
from scipy.interpolate import griddata


class VarSpace (object):

    def __init__(self, varspace):
        # varspace should be a dictionary with the following format: {
        # 'name':'varname' , 'space': [variable space] }
        parnum = len(varspace)  # extracting the number of parameters

        self.space = []
        self.one = []
        self.longspace = []
        self.names = []

        for i in range(0, parnum, 1):
            self.names.append(varspace[i]['name']) #array containing the names
            self.space.append({(varspace[i]['name']): varspace[i]['space']}) #dictionanary with variable spaces (grid axis)
            self.one.append({(varspace[i]['name']): np.ones(np.array(varspace[i]['space']).shape)})  # dictionaru with
            # array of ones the same size as the variable space

        for i in range(0, parnum, 1):  # creating an array with different permutations of variables

            alpha = 1
            for j in range(0, parnum, 1): # tensoring spaces with ones of length of other spaces
                if i != j:
                    alpha = np.kron(alpha, self.one[j][varspace[j]['name']])
                else:
                    alpha = np.kron(alpha, self.space[j][varspace[j]['name']])
            self.longspace.append({(varspace[i]['name']): alpha})


class DataSpace (object): #this class contains the processed data and grid information

    def __init__(self, data, space):
        # data should be a dictionary with the following format: { 'name':'varname' , 'space': [variable timeseries] }
        # first var name should be target and contain a vector of ones a zeros corresponding to a classification.
        # space should be a object from class 'space things'

        parnum = len(space.names)  # extracting the number of parameters

        self.bigspace = []
        self.bigdata = []
        self.names = space.names

        alpha = 1
        beta = 1

        for i in range(0, parnum, 1):
            alpha = np.kron(alpha, space.one[i][space.names['name']]) # creating a vector of ones with length of grid spaces
            beta = np.kron(beta, np.ones(np.array(data[i]['space'])))  # creating a vector of ones with length of data points

        onespacelong = alpha
        onedatalong = np.transpose(beta)

        for i in range(0, parnum + 1, 1):
            # this is the data from the data processing people
            self.bigdata.append({data[i]['name']: np.kron(onespacelong, data[i]['space'])}) #dic of big data arrays

        lonspacearray = []
        for i in range(0, parnum, 1):
            self.bigspace.append({data[i + 1]['name']: np.kron(onedatalong, space.longspace[i]['name'])})  # this is a
            lonspacearray = np.concatenate(lonspacearray,
            self.bigspace[i][data[i + 1]['name']])

        self.longspacearray = lonspacearray

        # grid that we calculate the averages for


class Likelihood(object):
    # dataspace is an object from the class DataSpace

    def __init__(self, dataspace):
        self.p = likelihoodgrid(dataspace)
        self.dataspace = dataspace

    def likelihood(self, point):
        return griddata(self.dataspace.lonspacearray, self.p, point, 'cubic') # function interpolates the likelihood
        #value using the grid


def likelihoodgrid(dataspace):
    # dataspace is an object from class DataSpace

    parnum = len(dataspace.names) #number of variables

    a = 0
    for i in range(0, parnum, 1):
        a += (dataspace.bigdata[i + 1][dataspace.names[i]] - dataspace.bigspace[i][dataspace.names[i]]) ** 2 #calculating
        # the distance between the grid points and the data points. Needs to be normalized by the average of the data.

    w = norm.pdf(np.sqrt(a), 0, 1) #Calculating weight of each point
    p = np.multiply(dataspace.bigdata[0], w) #multiplying the outcomes (one or zero) by the distribution
    p = np.sum(p, 0) #summing to complete the weighted average
    return p


def main():
    m = []
    m.append({'name': 'var1', 'space': [1, 2, 3, 4]})
    m.append({'name': 'var2', 'space': [1, 2, 3, 4, 5, 6]})
    print(m)
    h = VarSpace(m)
    print(h.longspace)

main()
