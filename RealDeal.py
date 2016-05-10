#this stuff should be functional

#importing libraries
import numpy as np
import matplotlib.pyplot as plt


def rolfour(timeseries, time, window, n=50):
    length = np.size(timeseries)  # length of the time series
    A = np.ones((length, length - window + 1))  # create a matrix of ones
    B = np.triu(A, 1)
    C = np.tril(A, -window)
    Proj = A - B - C  # this matrix selects the sections to fourier transform
    M = np.kron(np.reshape(timeseries, [np.size(timeseries), 1]), np.ones(length - window + 1))
    M = np.reshape(M.ravel()[np.flatnonzero(Proj)], (window, length - window + 1))
    transform = np.fft.rfft(M, n, 0)
    frequency = np.reshape(np.fft.rfftfreq(n, d=(time[2] - time[1])), (n, 1))
    return DataThings(frequency, transform)

class DataThings(object):
    def __init__(self, x, y):
        self.x = x
        self.y = y