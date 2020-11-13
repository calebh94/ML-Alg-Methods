"""
Implementation of the Algorithm for Hidden Markov Models

Caleb Harris
"""
import matplotlib.pyplot as plt
import numpy as np


def forward(X, ls, ks, pi, A, B):
    # init
    alphas = np.zeros((X.shape))
    # initialization
    # alphas[0,0] = np.prod(np.power(B[0,:], X[0,:])) * pi[0]
    # alphas[0,1] = np.prod(np.power(B[1,:], X[0,:])) * pi[1]
    alphas[0, :] = np.prod(np.power(B[:, :], X[0, :]), axis=1) * pi[:]
    # iteration
    for t in range(1, X.shape[0]):
        alphas[t,:] = np.prod(np.power(B[:, :], X[t, :]), axis=1) * np.sum(alphas[t-1, :] * A[:,:] , axis=1)
    return alphas


def backward(X, ls, ks, pi, A, B):
    betas = np.zeros((X.shape))
    # initialization
    betas[-1, :] = 1
    # iteration
    for t in range(X.shape[0]-1-1, 0, -1):
        betas[t,:] = np.sum( A[:,:] * np.power(B[:,:], X[t+1,:])* betas[t+1, :] , axis=0)
    return betas


def algo(q, Y):
    # init
    p = 0.0
    fig, ax = plt.subplots()

    # Problem Assumptions
    ls = [-1,1]  # "Decrease" and "Increase"
    ks = [0,1]  # "Bad" and "Good"
    pi = np.array([0.8,0.2])
    A = np.array([[0.8,0.2],[0.2,0.8]])
    B = np.array([[q, 1-q],[1-q, q]])

    Y_temp= np.ones((Y.shape[0], len(ks)))
    Y_onehot = Y_temp.copy()
    for i in range(0, len(ls)):
        Y_onehot[:,i] = np.multiply(Y_temp[:,i], np.transpose(Y==ls[i]))

    alphas = forward(Y_onehot, ls, ks, pi, A, B)
    betas = backward(Y_onehot, ls, ks, pi, A, B)

    P = np.sum(alphas[-1, :])
    p = alphas[-1, 1] * betas[-1, 1] / P
    ps = np.zeros((39))
    ps[0] = (alphas[0,1] * betas[0,1]) / (np.sum(alphas[0,:]))
    for n in range(1, ps.shape[0]):
        P_tmp = np.sum(alphas[n, :])
        ps[n] = alphas[n, 1] * betas[n, 1] / P_tmp
    plt.plot(np.linspace(1,39,39), ps)
    plt.show()

    return p, fig