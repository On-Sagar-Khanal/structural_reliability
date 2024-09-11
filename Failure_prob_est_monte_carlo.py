"""CODE TO CALCULATE FAILURE PROBABILITY OF A SHALLOW FOUNDATION USING MONTE CARLO SIMULATION"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

# Function to calculate bearing capacity using Hansen's method
def Hansen(x):
    B = 2                   # Width of foundation (in meters)
    Df = 0.5                # Depth of foundation (in meters)
    gs = 20                 # Unit weight of soil (in kN/m^3)
    c = x[0]                # Cohesion of soil (in kPa)
    phi = x[1] * np.pi/180  # Unit Conversion
    
    # Calculating bearing capacity factors Nc, Nq, and Ng
    Nq = (math.tan(np.pi/4 + phi/2))**2 * np.exp(np.pi * math.tan(phi))
    Ng = 1.8 * (Nq - 1) * math.tan(phi)
    Nc = (Nq - 1) * (math.tan(phi))**(-1)
   
    # Returning the total bearing capacity
    return 0.5 * gs * B * Ng + c * Nc + gs * Df * Nq

n_samples = 1000  # Number of samples for Monte Carlo simulation

x_m = np.array([20, 30])          # mean of cohesion (kPa) and friction angle (degrees)

x_sd = np.array([[5], [6]])       # standard deviations of cohesion and friction angle

# Correlation matrix for cohesion and friction angle
x_r = np.array([[1, -0.5],        
                [-0.5, 1]])

# Covariance matrix calculated from standard deviations and correlation matrix
Cv_x = x_sd @ np.transpose(x_sd) * x_r

# Generating random samples using multivariate normal distribution
x = np.random.multivariate_normal(x_m, Cv_x, n_samples)

# Initializing arrays
qu = np.zeros(n_samples)
I = np.zeros(n_samples)

# Loop to calculate bearing capacity and failure indicator
for sn, row in enumerate(x):
    qu[sn] = Hansen(row)       # Calculate bearing capacity using Hansen's method
    F = qu[sn] / 500           # Load factor (assuming applied load is 500 kN)
    
    # If load factor F < 1, foundation is safe; otherwise, failure occurs
    
    if F < 1:
        I[sn] = 1              
    else:
        I[sn] = 0            

# mean of failure indicators
pf = np.mean(I)

# COV of failure probability
COV = np.sqrt((1 - pf) / (n_samples * pf))

# Print the probability of failure and its COV
print("The mean probability of failure and COV for the shallow foundation are respectively: ", round(pf,2) , round(COV,2))
