# this is where I try different things out and decide how to get things to work


#importing libraries
import numpy as np
import matplotlib.pyplot as plt


Materials = []

Materials.append({'E': 17e4, 'var2': 'C:\\data file path', 'var3': [1, 2, 3, 4]})
Materials.append({'E': 17e4, 'var2': 'C:\data file path', 'var3':[1, 2,3, 4]})

print(Materials['E'])

a=np.array([[1,2, 7],[4,6, 5]])

print(np.transpose(a).shape)