
"""MONTE CARLO SIMULATION FOR GENERATING CORRELATED RANDOM VECTORS"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

n_samples=2000
x_m=np.array([ [20],[30]])          #mean matrix
x_sd=np.array([ [5],[6]])           #standard deviation matrix
x_r=np.array([ [1,-0.5],[-0.5,1]])  #correlation matrix

Cv_x=x_sd @ np.transpose(x_sd)*x_r  #calculating covariance matrix from correlation matrix

L,V = np.linalg.eig(Cv_x)
gen_values=np.empty((2,n_samples))
z=np.empty((2,1)) 

for x in range(0,n_samples):
    z[0,0]=np.random.normal(loc=0,scale=1)
    z[1,0]=np.random.normal(loc=0,scale=1)

    # generating correlated random vectors using the eigenvector and eigenvalue decomposition

    gen_values[:, x] = (x_m + V @ np.sqrt(np.diag(L)) @ z)[:, 0]

for_plot_gen=np.transpose(gen_values)

plt.scatter(for_plot_gen[:,0],for_plot_gen[:,1],s=3)
plt.grid()
plt.title("Randomly Generated vectors with correlated random vectors")
plt.show()
