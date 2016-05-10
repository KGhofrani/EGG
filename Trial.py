# this is where I try different things out and decide how to get things to work


# importing libraries
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

def main():
    # closing previous plots
    plt.close("all")
    time = np.linspace(0, 5, 2000)

    a=(data(5, 1, 2000)) #2*np.sin(2*time*np.pi*2)+np.sin(8*time*np.pi*2)+np.sin(15*time*np.pi*2)+np.sin(20*time*np.pi*2)
    b = rolfour(a.y, a.x, 2000, 100)
    #print(a.f, a.fstd, a.noisestd)

    plt.figure(1)
    plt.plot(b.x, np.multiply(b.y[:,0],np.conj(b.y[:,0])))
    plt.figure(2)
    plt.plot(a.y)
    plt.show()
    plt.draw()


class DataThings(object):
    def __init__(self, x, y):
        self.x = x
        self.y = y


def rolfour(timeseries, time, window, n=50):
    length = np.size(timeseries)  # length of the time series
    A = np.ones((length, length - window + 1))  # create a matrix of ones
    B = np.triu(A, 1)
    C = np.tril(A, -window)
    Proj = A - B - C  # this matrix selects the sections to fourier transform (ie, the rolling window)
    M = np.kron(np.reshape(timeseries, [np.size(timeseries), 1]), np.ones(length - window + 1)) # this selects the appropriete window
    M = np.reshape(M.ravel()[np.flatnonzero(Proj)], (window, length - window + 1))


    transform = np.fft.rfft(M, None, 0)
    frequency = np.fft.rfftfreq(window, d=(time[2] - time[1]))
    return DataThings(frequency, transform)


def EEG(time, n):
    f1=5
    f2=13
    f3=25

def data(f,t,n):  #this function generates a noisy time series there is noise in frequency and phase in addition to white noise.
    nch=10
    stdf=np.random.rand(1,1)*f/3
    stdN=np.random.rand(1,1)*nch/3
    t=np.linspace(0, t, n)
    fr=np.linspace(norm.ppf(0.01,f,stdf)[0],norm.ppf(0.99,f,stdf)[0],nch) #vector of different frequencies
    frM=np.kron(np.reshape(fr, (nch,1)),np.ones([1,n]))
    tM=np.kron(np.ones([nch,1]), t)
    phiM=2*np.random.rand(nch,n)
    a=np.sin(2*np.pi*frM*tM+phiM)
    w=norm.pdf(fr, f, stdf) + np.random.randn(1,nch)
    noise=stdN*np.random.randn(nch,n)
    wM=np.kron(np.reshape(w,(nch,1)),np.ones([1,n]))
    outM=np.multiply(wM,a)+noise
    out=np.sum(outM,0)
    return FakeData(t, out, f, stdf, stdN )

class FakeData(object):
    def __init__(self, x, y, f, fstd, noisestd):
        self.x=x
        self.y=y
        self.f=f
        self.fstd=fstd
        self.noisestd=noisestd
main()
