"""OPTIMIZATION PROBLEM FOR FINDING RELIABILITY INDEX"""

import numpy as np
from scipy.optimize import minimize # for minimization optimization
import math

# our objective function for optimization
def objective(A):
    return np.sqrt(np.dot(A,np.transpose(A)))

# contraints for optimization
def constraint(A):
    return (A[0]*2+10)*(A[1]*2+3)-(A[2]*6+25)
    
# intial guess
Init_guess=np.zeros(3)
#######################

b=(2,50)
bnds=(b,b,b)
con={'type':"ineq","fun":constraint}
sol=minimize(objective,Init_guess,method="SLSQP",bounds=bnds,constraints=con)  #optimizing
print("The final opitmization result is:\n\n",sol)
