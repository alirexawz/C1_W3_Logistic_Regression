import numpy as np
import matplotlib.pyplot as plt
from utils import *
import copy
import math


# load dataset
X_train, y_train = load_data("ex2data1.txt")

# print("First five elements in X_train are:\n", y_train[:5])
# print("Type of X_train:",type(X_train))

# print("First five elements in y_train are:\n", y_train[:5])
# print("Type of y_train:",type(y_train))

# print ('The shape of X_train is: ' + str(X_train.shape))
# print ('The shape of y_train is: ' + str(y_train.shape))
# print ('We have m = %d training examples' % (len(y_train)))


# Plot examples
# plot_data(X_train, y_train[:], pos_label="Admitted", neg_label="Not admitted")

# Set the y-axis label
# plt.ylabel('Exam 2 score') 
# Set the x-axis label
# plt.xlabel('Exam 1 score') 
# plt.legend(loc="upper right")
# plt.show()


def compute_cost(X, y, w, b, lambda_= 1):
 m, n = X.shape

 ### START CODE HERE ###
 loss_sum = 0 

 # Loop over each training example
 for i in range(m): 

     # First calculate z_wb = w[0]*X[i][0]+...+w[n-1]*X[i][n-1]+b
     z_wb = 0 
     # Loop over each feature
     for j in range(n): 
         # Add the corresponding term to z_wb
         z_wb_ij = w[j]*X[i][j] # Your code here to calculate w[j] * X[i][j]
         z_wb += z_wb_ij # equivalent to z_wb = z_wb + z_wb_ij
     # Add the bias term to z_wb
     z_wb += b # equivalent to z_wb = z_wb + b

     f_wb = sig(z_wb) # Your code here to calculate prediction f_wb for a training example
     loss =  -y[i] * np.log(f_wb) - (1 - y[i]) * np.log(1 - f_wb)

     loss_sum += loss # equivalent to loss_sum = loss_sum + loss

 total_cost = (1 / m) * loss_sum  
 ### END CODE HERE ### 

 return total_cost